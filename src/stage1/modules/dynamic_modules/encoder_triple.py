import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.nn_modules import AttnBlock, Downsample, ResnetBlock, group_norm
from src.utils.util_modules import instantiate_from_config


class TripleGrainEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        router_config=None,
        update_router=True,
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.update_router = update_router

        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.Sequential()
        for i_level in range(self.num_resolutions):
            block = nn.Sequential()
            attn = nn.Sequential()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        def _make_grain_branch(block_in):
            branch = nn.Sequential(
                ResnetBlock(
                    in_channels=block_in, out_channels=block_in, dropout=dropout
                ),
                AttnBlock(block_in),
                ResnetBlock(
                    in_channels=block_in, out_channels=block_in, dropout=dropout
                ),
                group_norm(block_in),
                nn.SiLU(),
                nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1),
            )
            return branch

        self.coarse_branch = _make_grain_branch(block_in)
        block_in_median = block_in // (ch_mult[-1] // ch_mult[-2])
        self.median_branch = _make_grain_branch(block_in_median)
        block_in_fine = block_in_median // (ch_mult[-2] // ch_mult[-3])
        self.fine_branch = _make_grain_branch(block_in_fine)

        self.router = instantiate_from_config(router_config)

    def forward(self, x, x_entropy):
        assert x.shape[2] == x.shape[3] == self.resolution

        hs, h_fine, h_median = self._downsample(x)
        h_coarse = hs[-1]

        h_coarse = self.coarse_branch(h_coarse)
        h_median = self.median_branch(h_median)
        h_fine = self.fine_branch(h_fine)

        return self._dynamic_routing(h_coarse, h_median, h_fine, x_entropy)

    def _downsample(self, x):
        hs = [self.conv_in(x)]
        h_fine = None
        h_median = None
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if self.down[i_level].attn:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level == self.num_resolutions - 2:
                h_median = h
            elif i_level == self.num_resolutions - 3:
                h_fine = h
        return hs, h_fine, h_median

    def _dynamic_routing(self, h_coarse, h_median, h_fine, x_entropy):
        gate = self.router(
            h_fine=h_fine, h_median=h_median, h_coarse=h_coarse, entropy=x_entropy
        )
        if self.update_router and self.training:
            gate = F.gumbel_softmax(gate, tau=1, dim=-1, hard=True)

        gate = gate.permute(0, 3, 1, 2)
        indices = gate.argmax(dim=1)

        h_coarse = h_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        h_median = h_median.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        indices_repeat = (
            indices.repeat_interleave(4, dim=-1)
            .repeat_interleave(4, dim=-2)
            .unsqueeze(1)
        )

        h_triple = torch.where(indices_repeat == 0, h_coarse, h_median)
        h_triple = torch.where(indices_repeat == 1, h_median, h_triple)
        h_triple = torch.where(indices_repeat == 2, h_fine, h_triple)

        if self.update_router and self.training:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(4, dim=-1).repeat_interleave(
                4, dim=-2
            )
            h_triple = h_triple * gate_grad

        coarse_mask = 0.0625 * torch.ones_like(indices_repeat, device=h_triple.device)
        median_mask = 0.25 * torch.ones_like(indices_repeat, device=h_triple.device)
        fine_mask = 1.0 * torch.ones_like(indices_repeat, device=h_triple.device)
        codebook_mask = torch.where(indices_repeat == 0, coarse_mask, median_mask)
        codebook_mask = torch.where(indices_repeat == 1, median_mask, codebook_mask)
        codebook_mask = torch.where(indices_repeat == 2, fine_mask, codebook_mask)

        return {
            "h_triple": h_triple,
            "indices": indices,
            "codebook_mask": codebook_mask,
            "gate": gate,
        }
