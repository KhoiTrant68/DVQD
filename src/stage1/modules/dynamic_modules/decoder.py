import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchtune.modules import RotaryPositionalEmbeddings

from src.utils.nn_modules import AttnBlock, ResnetBlock, Upsample, group_norm


class PositionEmbedding2DLearned(nn.Module):
    """
    2D Learned Positional Embedding.
    """

    def __init__(self, n_row, feats_dim, n_col=None):
        """
        Initializes the PositionEmbedding2DLearned.

        Args:
            n_row (int): Number of rows for the embedding.
            feats_dim (int): Dimension of the feature embeddings.
            n_col (int, optional): Number of columns for the embedding. Defaults to n_row.
        """
        super().__init__()
        n_col = n_col or n_row
        self.row_embed = nn.Embedding(n_row, feats_dim)
        self.col_embed = nn.Embedding(n_col, feats_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the weights of the embeddings using a truncated normal distribution."""
        trunc_normal_(self.row_embed.weight)
        trunc_normal_(self.col_embed.weight)

    def forward(self, x):
        """
        Forward pass to add positional embeddings to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (..., H, W).

        Returns:
            torch.Tensor: Input tensor with added positional embeddings.
        """
        h, w = x.shape[-2:]
        device = x.device
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)

        x_emb = self.col_embed(i).unsqueeze(0).expand(h, -1, -1)
        y_emb = self.row_embed(j).unsqueeze(1).expand(-1, w, -1)

        pos = (
            (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        )

        if x.dim() == 5:
            pos = pos.unsqueeze(-3)

        return x + pos


class Decoder(nn.Module):
    """
    Decoder module for image generation.
    """

    def __init__(
        self,
        ch,
        in_ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        resolution,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        give_pre_end=False,
        latent_size=32,
        window_size=2,
        position_type="relative",
    ):
        """
        Initializes the Decoder.
        Args:
            ch (int): Base number of channels.
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            ch_mult (list): Multipliers for the number of channels at each resolution level.
            num_res_blocks (int): Number of resnet blocks per resolution level.
            resolution (int): Initial resolution of the input.
            attn_resolutions (list): Resolutions at which to apply attention.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            resamp_with_conv (bool, optional): Whether to use convolution for resampling. Defaults to True.
            give_pre_end (bool, optional): Whether to return intermediate features. Defaults to False.
            latent_size (int, optional): Size of the latent space. Defaults to 32.
            window_size (int, optional): Size of the window for relative positional embeddings. Defaults to 2.
            position_type (str, optional): Type of positional embeddings to use. Defaults to "relative".
        """
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.temb_ch = 0
        self.ch = ch
        self.give_pre_end = give_pre_end

        # compute block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, in_ch, curr_res, curr_res)

        self.conv_in = nn.Conv2d(in_ch, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
            ),
            AttnBlock(block_in),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
            ),
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.Sequential()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.add_module(
                    f"block_{i_block}",
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    ),
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        # relative pos embeddings
        self.position_type = position_type
        if self.position_type == "learned":
            self.position_bias = PositionEmbedding2DLearned(
                n_row=latent_size, feats_dim=in_ch
            )
        elif self.position_type == "learned+relative":
            self.position_bias = PositionEmbedding2DLearned(
                n_row=window_size, feats_dim=in_ch
            )
            self.window_size = window_size
            self.window_num = latent_size // window_size
        elif self.position_type == "rope":
            self.position_bias = RotaryPositionalEmbeddings(dim=latent_size)
        elif self.position_type == "rope+learned":
            self.position_bias_rope = RotaryPositionalEmbeddings(dim=latent_size)
            self.position_bias_learned = PositionEmbedding2DLearned(
                n_row=latent_size, feats_dim=in_ch
            )
        else:
            raise NotImplementedError()

    def forward(self, h):
        """
        Forward pass of the decoder.
        Args:
            h (torch.Tensor): Input tensor to the decoder.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """
        if self.position_type in ["learned", "rope"]:
            h = self.position_bias(h)
        elif self.position_type == "learned+relative":
            h = rearrange(
                h,
                "B C (n1 nH) (n2 nW) -> B C (n1 n2) nH nW",
                n1=self.window_num,
                nH=self.window_size,
                n2=self.window_num,
                nW=self.window_size,
            )
            h = self.position_bias(h)
            h = rearrange(
                h,
                "B C (n1 n2) nH nW -> B C (n1 nH) (n2 nW)",
                n1=self.window_num,
                nH=self.window_size,
            )
        elif self.position_type == "rope+learned":
            h = self.position_bias_rope(h)
            h = self.position_bias_learned(h)

        # z to block_in
        h = self.conv_in(h)

        # middle
        h = self.mid(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h = self.up[i_level].block(h)
            for attn_block in self.up[i_level].attn:
                h = attn_block(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
