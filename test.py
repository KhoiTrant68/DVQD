import torch

from src.stage1.modules.dynamic_modules.decoder import (
    Decoder, PositionEmbedding2DLearned)


def test_PositionEmbedding2DLearned():
    n_row, feats_dim, n_col = 10, 64, 10
    model = PositionEmbedding2DLearned(n_row, feats_dim, n_col)
    x = torch.randn(1, feats_dim, n_row, n_col)
    output = model(x)
    print(f"PositionEmbedding2DLearned output shape: {output.shape}")


def test_Decoder():
    ch, in_ch, out_ch = 64, 3, 3
    ch_mult = [1, 2, 4]
    num_res_blocks = 2
    resolution = 32
    attn_resolutions = [16]
    latent_size = 32
    window_size = 2
    position_type = "rope"

    model = Decoder(
        ch,
        in_ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        resolution,
        attn_resolutions,
        latent_size=latent_size,
        window_size=window_size,
        position_type=position_type,
    )
    h = torch.randn(1, in_ch, resolution, resolution)
    output = model(h)
    print(f"Decoder output shape: {output.shape}")


if __name__ == "__main__":
    test_PositionEmbedding2DLearned()
    test_Decoder()
