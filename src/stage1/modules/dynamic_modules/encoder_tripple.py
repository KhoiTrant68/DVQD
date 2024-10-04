import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.nn_modules import AttnBlock, Downsample, ResnetBlock, group_norm
from src.utils.util_modules import instantiate_from_config


class TripleGrainEncoder(nn.Module):
    """
    Triple Grain Encoder for processing input images at different resolutions.
    """

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: list,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        router_config: dict = None,
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels

        # Downsampling layers
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.down = self._build_downsampling_layers(
            ch, ch_mult, num_res_blocks, attn_resolutions, dropout, resamp_with_conv
        )

        # Mid-grain processing layers
        self.mid_blocks, self.norm_out, self.conv_out = self._build_mid_grain_layers(
            ch, ch_mult, dropout
        )

        # Router for combining outputs
        self.router = instantiate_from_config(router_config)

    def _build_downsampling_layers(
        self,
        ch: int,
        ch_mult: tuple,
        num_res_blocks: int,
        attn_resolutions: list,
        dropout: float,
        resamp_with_conv: bool,
    ):
        """
        Build downsampling layers for the encoder.
        """
        down_layers = nn.ModuleList()
        curr_res = self.resolution
        in_ch_mult = [
            1,
        ] + ch_mult

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
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
            down_layers.append(down)

        return down_layers

    def _build_mid_grain_layers(self, ch: int, ch_mult: tuple, dropout: float):
        """
        Build mid-grain processing layers for the encoder.
        """
        mid_blocks = nn.ModuleList()
        norm_out = nn.ModuleList()
        conv_out = nn.ModuleList()

        for i in range(3):
            block_in = ch * ch_mult[-(i + 1)]
            mid_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            in_channels=block_in, out_channels=block_in, dropout=dropout
                        ),
                        AttnBlock(block_in),
                        ResnetBlock(
                            in_channels=block_in, out_channels=block_in, dropout=dropout
                        ),
                    ]
                )
            )
            norm_out.append(group_norm(block_in))
            conv_out.append(
                nn.Conv2d(block_in, self.z_channels, kernel_size=3, stride=1, padding=1)
            )

        return mid_blocks, norm_out, conv_out

    def forward(self, x: torch.Tensor, x_entropy: torch.Tensor = None) -> dict:
        """
        Forward pass through the encoder.
        """
        assert (
            x.shape[2] == x.shape[3] == self.resolution
        ), f"Input resolution mismatch: {x.shape[2]}, {x.shape[3]} != {self.resolution}"

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level].block:
                h = block(hs[-1])
                if self.down[i_level].attn:
                    h = self.down[i_level].attn[0](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h_coarse, h_median, h_fine = hs[-3], hs[-2], hs[-1]

        # Process mid-grain blocks
        h_outputs = [
            self._process_mid_block(
                h, self.mid_blocks[i], self.norm_out[i], self.conv_out[i]
            )
            for i, h in enumerate([h_coarse, h_median, h_fine])
        ]

        # Router and gate processing
        gate = self.router(
            h_fine=h_outputs[2],
            h_median=h_outputs[1],
            h_coarse=h_outputs[0],
            entropy=x_entropy,
        )
        if self.training:
            gate = F.gumbel_softmax(gate, tau=1, dim=-1, hard=True)

        indices = gate.argmax(dim=1).unsqueeze(1)
        h_triple = self._combine_outputs(h_outputs, indices)

        if self.training:
            gate_grad = (
                gate.max(dim=1, keepdim=True)[0]
                .repeat_interleave(4, dim=-1)
                .repeat_interleave(4, dim=-2)
            )
            h_triple *= gate_grad

        # Masks
        codebook_mask = self._create_codebook_mask(indices)

        return {
            "h_triple": h_triple,
            "indices": indices.squeeze(1),
            "codebook_mask": codebook_mask,
            "gate": gate,
        }

    def _process_mid_block(
        self,
        h: torch.Tensor,
        mid_block: nn.ModuleList,
        norm: nn.Module,
        conv: nn.Module,
    ) -> torch.Tensor:
        """
        Process a mid-grain block.
        """
        for block in mid_block:
            h = block(h)
        h = norm(h)
        return F.relu(conv(h))

    def _combine_outputs(self, h_outputs: list, indices: torch.Tensor) -> torch.Tensor:
        """
        Combine outputs based on indices.
        """
        h_triple = torch.where(indices == 0, h_outputs[0], h_outputs[1])
        h_triple = torch.where(indices == 1, h_outputs[1], h_triple)
        return torch.where(indices == 2, h_outputs[2], h_triple)

    def _create_codebook_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Create a codebook mask based on indices.
        """
        masks = torch.tensor([0.0625, 0.25, 1.0], device=indices.device)
        codebook_mask = masks[indices]
        return codebook_mask
