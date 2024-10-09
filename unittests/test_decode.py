import sys

sys.path.append("..")
import torch

from src.stage1.modules.dynamic_modules.decoder import Decoder


def run_decoder():
    decoder = Decoder(
        ch=128,
        in_ch=256,
        out_ch=3,
        ch_mult=[1, 1, 2, 2],
        num_res_blocks=2,
        resolution=256,
        attn_resolutions=[32],
        latent_size=32,
        window_size=2,
        position_type="rope+learned",
    )

    dummy_input = torch.rand(1, 256, 32, 32)
    output = decoder(dummy_input)
    print(output.shape)


if __name__ == "__main__":
    run_decoder()
