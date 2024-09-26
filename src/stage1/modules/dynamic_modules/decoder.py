import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from diffusers import UNet2DModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import AttentionBlock
from diffusers.models.upsample import Upsample2D

class Decoder(nn.Module):
    def __init__(self, 
                 ch: int, in_ch: int, out_ch: int, ch_mult: List[int], num_res_blocks: int, resolution: int,
                 attn_resolutions: List[int], dropout: float = 0.0, resamp_with_conv: bool = True, give_pre_end: bool = False):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.give_pre_end = give_pre_end

        # Compute block_in and curr_res at lowest res
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, in_ch, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        self.conv_in = nn.Conv2d(in_ch, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Sequential(
            ResnetBlock2D(in_channels=block_in, out_channels=block_in),
            AttentionBlock(block_in),
            ResnetBlock2D(in_channels=block_in, out_channels=block_in)
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                layers = [ResnetBlock2D(in_channels=block_in, out_channels=block_out)]
                block_in = block_out
                if curr_res in attn_resolutions:
                    layers.append(AttentionBlock(block_in))
                self.up.append(nn.Sequential(*layers))
            if i_level != 0:
                self.up.append(Upsample2D(channels=block_in, use_conv=resamp_with_conv))
                curr_res *= 2

        # End
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, h: torch.Tensor, grain_indices: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(h)
        h = self.mid(h)

        for layer in self.up:
            h = layer(h)

        if self.give_pre_end:
            return h

        h = self.conv_out(self.norm_out(nn.SiLU()(h)))
        return h