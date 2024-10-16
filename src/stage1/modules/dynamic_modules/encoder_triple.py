import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.nn_modules import AttnBlock, Downsample, ResnetBlock, group_norm
from src.utils.util_modules import instantiate_from_config


class TripleGrainEncoder(nn.Module):
    """
    Triple Grain Encoder for image processing.
    """

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
        """
        Initializes the TripleGrainEncoder.
        Args:
            ch (int): Base number of channels.
            ch_mult (tuple): Multipliers for the number of channels at each resolution level.
            num_res_blocks (int): Number of resnet blocks per resolution level.
            attn_resolutions (list): Resolutions at which to apply attention.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            resamp_with_conv (bool, optional): Whether to use convolution for resampling. Defaults to True.
            in_channels (int): Number of input channels.
            resolution (int): Initial resolution of the input.
            z_channels (int): Number of channels in the latent space.
            router_config (dict, optional): Configuration for the dynamic router. Defaults to None.
            update_router (bool, optional): Whether to update the router during training. Defaults to True.
        """
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.Sequential()
        for i_level in range(self.num_resolutions):
            block = nn.Sequential()
            attn = nn.Sequential()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
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

        # Middle and end for each grain
        def _make_grain_branch(block_in):
            return nn.Sequential(
                ResnetBlock(block_in, block_in, dropout=dropout),
                AttnBlock(block_in),
                ResnetBlock(block_in, block_in, dropout=dropout),
                group_norm(block_in),
                nn.SiLU(),
                nn.Conv2d(block_in, z_channels, 3, 1, 1),
            )

        self.mid_coarse = _make_grain_branch(block_in)
        block_in_median = block_in // (ch_mult[-1] // ch_mult[-2])
        self.mid_median = _make_grain_branch(block_in_median)
        block_in_fine = block_in_median // (ch_mult[-2] // ch_mult[-3])
        self.mid_fine = _make_grain_branch(block_in_fine)

        self.router = instantiate_from_config(router_config)
        self.update_router = update_router

    def forward(self, x, x_entropy):
        """
        Forward pass of the encoder.
        Args:
            x (torch.Tensor): Input tensor to the encoder.
            x_entropy (torch.Tensor): Entropy map for dynamic routing.

        Returns:
            dict: A dictionary containing the triple grain output, indices, codebook mask, and gate values.
        """
        assert x.shape[2] == x.shape[3] == self.resolution

        hs, h_fine, h_median = self._downsample(x)
        h_coarse = hs[-1]

        h_coarse = self.mid_coarse(h_coarse)
        h_median = self.mid_median(h_median)
        h_fine = self.mid_fine(h_fine)

        return self._dynamic_routing(h_coarse, h_median, h_fine, x_entropy)

    def _downsample(self, x):
        """
        Downsamples the input tensor through multiple resolution levels.

        Args:
            x (torch.Tensor): Input tensor to be downsampled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the downsampled features,
            fine grain features, and median grain features.
        """
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
            if i_level == self.num_resolutions - 2:  # Extract median grain
                h_median = h
            elif i_level == self.num_resolutions - 3:  # Extract fine grain
                h_fine = h
        return hs, h_fine, h_median

    def _dynamic_routing(self, h_coarse, h_median, h_fine, x_entropy):
        """
        Performs dynamic routing between coarse, median, and fine grain features.

        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_median (torch.Tensor): Median grain features.
            h_fine (torch.Tensor): Fine grain features.
            x_entropy (torch.Tensor): Entropy map for dynamic routing.

        Returns:
            dict: A dictionary containing the triple grain output, indices, codebook mask, and gate values.
        """

        gate = self.router(
            h_fine=h_fine, h_median=h_median, h_coarse=h_coarse, entropy=x_entropy
        )
        if self.update_router:
            gate = F.gumbel_softmax(gate, dim=-1, hard=True)
        gate = gate.permute(0, 3, 1, 2)
        indices = gate.argmax(dim=1)

        # Upsample and select based on indices (more efficient implementation)
        h_coarse_upsampled = h_coarse.repeat_interleave(4, dim=-1).repeat_interleave(
            4, dim=-2
        )
        h_median_upsampled = h_median.repeat_interleave(2, dim=-1).repeat_interleave(
            2, dim=-2
        )

        indices_repeat = (
            indices.repeat_interleave(4, dim=-1)
            .repeat_interleave(4, dim=-2)
            .unsqueeze(1)
        )

        h_triple = torch.where(
            indices_repeat == 0, h_coarse_upsampled, h_median_upsampled
        )
        h_triple = torch.where(indices_repeat == 2, h_fine, h_triple)

        if self.update_router:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(4, dim=-1).repeat_interleave(
                4, dim=-2
            )
            h_triple.mul_(gate_grad)

        # Mask generation (simplified)
        coarse_mask_value = 0.0625
        median_mask_value = 0.25
        fine_mask_value = 1.0

        codebook_mask = torch.where(
            indices_repeat == 0, coarse_mask_value, median_mask_value
        )
        codebook_mask = torch.where(indices_repeat == 2, fine_mask_value, codebook_mask)

        return {
            "h_triple": h_triple,
            "indices": indices,
            "codebook_mask": codebook_mask,
            "gate": gate,
        }
