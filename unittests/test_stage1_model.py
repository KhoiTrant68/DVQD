import sys

sys.path.append("..")
import torch
import yaml
from omegaconf import OmegaConf

from src.utils.util_modules import instantiate_from_config


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def test_grain_vq_model(config_path):
    # Load the configuration
    config = OmegaConf.load(config_path)

    # Extract model configuration
    model = instantiate_from_config(config.model)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 256, 256)

    # Perform a forward pass
    output = model(dummy_input)

    # Print the output shapes
    print("Decoded Image Shape:", output[0].shape)
    print("Quantization Loss:", output[1])
    print("Grain Indices Shape:", output[2].shape)
    print("Gate Values Shape:", output[3].shape)


# Run the test
# test_grain_vq_model("./src/configs/stage1/dual_feat_imagenet.yml")


# test_grain_vq_model("./src/configs/stage1/dual_dynamic_entropy_imagenet.yml")


# test_grain_vq_model("./src/configs/stage1/dual_fixed_entropy_imagenet.yml")


# test_grain_vq_model("./src/configs/stage1/triple_feat_imagenet.yml")

# test_grain_vq_model("./src/configs/stage1/triple_fixed_entropy_imagenet.yml")

test_grain_vq_model("./src/configs/stage1/triple_dynamic_entropy_imagenet.yml")
