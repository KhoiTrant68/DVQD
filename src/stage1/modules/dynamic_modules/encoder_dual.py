import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.nn_modules import AttnBlock, Downsample, ResnetBlock, group_norm
from src.utils.utils_modules import instantiate_from_config


class DualGrainEncoder(nn.Module):
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
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
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

        # middle and end for the coarse grain
        self.mid_coarse = nn.Sequential(
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            AttnBlock(block_in),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
        )
        self.norm_out_coarse = group_norm(block_in)
        self.conv_out_coarse = nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

        block_in_finegrain = block_in // (ch_mult[-1] // ch_mult[-2])
        # middle and end for the fine grain
        self.mid_fine = nn.Sequential(
            ResnetBlock(
                in_channels=block_in_finegrain,
                out_channels=block_in_finegrain,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            AttnBlock(block_in_finegrain),
            ResnetBlock(
                in_channels=block_in_finegrain,
                out_channels=block_in_finegrain,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
        )
        self.norm_out_fine = group_norm(block_in_finegrain)
        self.conv_out_fine = nn.Conv2d(
            block_in_finegrain, z_channels, kernel_size=3, stride=1, padding=1
        )

        self.router = instantiate_from_config(router_config)
        self.update_router = update_router

    def forward(self, x, x_entropy):
        assert (
            x.shape[2] == x.shape[3] == self.resolution
        ), f"{x.shape[2]}, {x.shape[3]}, {self.resolution}"

        hs, h_fine = self._downsample(x)
        h_coarse = hs[-1]

        h_coarse = self._process_coarse(h_coarse)
        h_fine = self._process_fine(h_fine)

        return self._dynamic_routing(h_coarse, h_fine, x_entropy)

    def _downsample(self, x):
        hs = [self.conv_in(x)]
        h_fine = None
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], None)
                if self.down[i_level].attn:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level == self.num_resolutions - 2:
                h_fine = h
        return hs, h_fine

    def _process_coarse(self, h):
        h = self.mid_coarse(h)
        h = self.norm_out_coarse(h)
        h = F.silu(h)
        return self.conv_out_coarse(h)

    def _process_fine(self, h):
        h = self.mid_fine(h)
        h = self.norm_out_fine(h)
        h = F.silu(h)
        return self.conv_out_fine(h)

    def _dynamic_routing(self, h_coarse, h_fine, x_entropy):
        gate = self.router(h_fine=h_fine, h_coarse=h_coarse, entropy=x_entropy)
        if self.update_router:
            gate = F.gumbel_softmax(gate, dim=-1, hard=True)
        gate = gate.permute(0, 3, 1, 2)
        indices = gate.argmax(dim=1)

        h_coarse = h_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        indices_repeat = (
            indices.repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .unsqueeze(1)
        )
        h_dual = torch.where(indices_repeat == 0, h_coarse, h_fine)

        if self.update_router:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(2, dim=-1).repeat_interleave(
                2, dim=-2
            )
            h_dual.mul_(gate_grad)

        coarse_mask = 0.25 * torch.ones_like(indices_repeat, device=h_dual.device)
        fine_mask = torch.ones_like(indices_repeat, device=h_dual.device)
        codebook_mask = torch.where(indices_repeat == 0, coarse_mask, fine_mask)

        return {
            "h_dual": h_dual,
            "indices": indices,
            "codebook_mask": codebook_mask,
            "gate": gate,
        }
