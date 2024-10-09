import torch

from src.stage1.modules.dynamic_modules.encoder_dual import DualGrainEncoder
from src.stage1.modules.dynamic_modules.encoder_triple import \
    TripleGrainEncoder
from src.stage1.modules.dynamic_modules.router_dual import (
    DualGrainDynamicEntropyRouter, DualGrainFeatureRouter)
from src.stage1.modules.dynamic_modules.router_triple import (
    TripleGrainDynamicEntropyRouter, TripleGrainFeatureRouter)

# Dual grain router configuration
dual_entropy_router_config = DualGrainDynamicEntropyRouter(
    fine_grain_ratio_min=0.01, fine_grain_ratio_max=0.99
)
dual_feat_router_config = DualGrainFeatureRouter(
    num_channels=256,
    num_groups=32,
)

# Triple grain router configuration
triple_entropy_router_config = TripleGrainDynamicEntropyRouter(
    fine_grain_ratio_min=0.01, fine_grain_ratio_max=0.99
)
triple_feat_router_config = TripleGrainFeatureRouter(
    num_channels=256,
    num_groups=32,
)


# DualGrainEncoder forward pass
def run_dual_grain_encoder():
    encoder = DualGrainEncoder(
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16, 32],
        dropout=0.0,
        in_channels=3,
        resolution=256,
        z_channels=256,
        router_config=triple_feat_router_config,
    )

    x = torch.randn(4, 3, 256, 256)
    x_entropy = torch.randn(4, 3, 256, 256)

    output = encoder(x, x_entropy)
    print("DualGrainEncoder Output Shapes:")
    print("h_dual shape:", output["h_dual"].shape)
    print("indices shape:", output["indices"].shape)
    print("codebook_mask shape:", output["codebook_mask"].shape)


# TripleGrainEncoder forward pass
def run_triple_grain_encoder():
    encoder = TripleGrainEncoder(
        ch=128,
        ch_mult=[1, 1, 2, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
        in_channels=3,
        resolution=256,
        z_channels=256,
        router_config=triple_feat_router_config,
    )

    x = torch.randn(1, 3, 256, 256)
    x_entropy = torch.randn(1, 3, 256, 256)

    output = encoder(x, x_entropy)

    print("TripleGrainEncoder Output Shapes:")
    print("h_triple shape:", output["h_triple"].shape)
    print("indices shape:", output["indices"].shape)
    print("codebook_mask shape:", output["codebook_mask"].shape)
    print("gate shape:", output["gate"].shape)


if __name__ == "__main__":
    run_dual_grain_encoder()
    run_triple_grain_encoder()
