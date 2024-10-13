import sys

import torch

sys.path.append("..")
from src.stage1.modules.losses.vq_lpips_multidisc import \
    VQLPIPSWithDiscriminator


def test_vq_lpips_with_discriminator():
    disc_config = {
        "target": "src.stage1.modules.discriminator.model.NLayerDiscriminator",
        "params": {"input_nc": 3, "ndf": 64, "n_layers": 3, "use_actnorm": False},
    }
    budget_loss_config = {
        "target": "src.stage1.modules.dynamic_modules.budget.BudgetConstraint_RatioMSE_DualGrain",
        "params": {
            "target_ratio": 0.5,
            "gamma": 10.0,
            "min_grain_size": 16,
            "max_grain_size": 32,
            "calculate_all": True,
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = VQLPIPSWithDiscriminator(
        disc_start=0,
        disc_config=disc_config,
        disc_init=True,
        disc_factor=1.0,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_conditional=False,
        disc_adaptive_loss=True,
        disc_loss="hinge",
        disc_weight_max=0.75,
        budget_loss_config=budget_loss_config,
    ).to(device)

    loss_fn.training = False

    batch_size = 4
    channels = 3
    height = 256
    width = 256

    codebook_loss = torch.rand(batch_size, device=device)
    inputs = torch.rand(batch_size, channels, height, width, device=device)
    reconstructions = torch.rand(batch_size, channels, height, width, device=device)
    optimizer_idx = torch.tensor(0, device=device)
    global_step = torch.tensor(0, device=device)
    last_layer = torch.rand(
        channels, height // 2, 3, 3, device=device, requires_grad=True
    )
    gate = torch.rand(batch_size, 2, height // 16, width // 16, device=device)

    loss, log_dict = loss_fn(
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer,
        cond=None,
        split="train",
        gate=gate,
    )

    print(loss, log_dict)

    print(f"Generator loss shape: {loss}")
    for key, value in log_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value}")
        else:
            print(f"{key}: {value} (type: {type(value)})")

    optimizer_idx = torch.tensor(1, device=device)
    d_loss, d_log_dict = loss_fn(
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer,
        cond=None,
        split="train",
        gate=gate,
    )

    print(f"Discriminator loss shape: {d_loss}")
    for key, value in d_log_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value}")
        else:
            print(f"{key}: {value} (type: {type(value)})")


if __name__ == "__main__":
    test_vq_lpips_with_discriminator()
