import sys

import torch

sys.path.append("..")
from src.stage2.modules.triple_permuter import TripleGrainSeparatePermuter

def test_triple_grain_separate_permuter():
    test_code = 1
    position_order = "region-first"
    if test_code == 1:
        # Test code 1
        x1 = torch.randint(0, 1024, (2, 16, 16))  # Input tensor for coarse grain
        x2 = 4 * torch.ones_like(x1)  # Input tensor for medium grain
        x3 = 8 * torch.ones_like(x1)  # Input tensor for fine grain

        grain_indices = torch.randint(0, 3, (2, 4, 4))  # Grain indices for coarse, medium, fine
        grain_indices_repeat = grain_indices.repeat_interleave(
            2, dim=-1
        ).repeat_interleave(2, dim=-2)

        # Convert to Float type before interpolation
        grain_indices_repeat = grain_indices_repeat.float()
        grain_indices_repeat = torch.nn.functional.interpolate(grain_indices_repeat.unsqueeze(1), size=(16, 16), mode='nearest').squeeze(1)

        original_indices = (
            x1 * (grain_indices_repeat == 0) +
            x2 * (grain_indices_repeat == 1) +
            x3 * (grain_indices_repeat == 2)
        )
        print("original_indices-------------", original_indices.shape)
        print("grain_indices-------------", grain_indices.shape)

        permuter = TripleGrainSeparatePermuter(
            coarse_size=8,
            medium_size=16,
            fine_size=32,
            content_pad_code=1024,
            content_eos_code=1025,
            coarse_position_pad_code=128,
            coarse_position_eos_code=129,
            medium_position_pad_code=256,
            medium_position_eos_code=257,
            fine_position_pad_code=1024,
            fine_position_eos_code=1025,
            position_order=position_order,
        )

        out = permuter(original_indices, grain_indices)
        print("original_indices:", original_indices)
        print("grain_indices:", grain_indices)

        coarse_content, medium_content, fine_content, coarse_position, medium_position, fine_position = (
            out["coarse_content"],
            out["medium_content"],
            out["fine_content"],
            out["coarse_position"],
            out["medium_position"],
            out["fine_position"],
        )
        print("coarse_content", coarse_content.shape)
        print("medium_content", medium_content.shape)
        print("fine_content", fine_content.shape)
        print("coarse_position", coarse_position.shape)
        print("medium_position", medium_position.shape)
        print("fine_position", fine_position.shape)

        target_fine = permuter.forward_back(
            coarse_content, medium_content, fine_content, coarse_position, medium_position, fine_position
        )
        print(target_fine)
        print(torch.all(target_fine == original_indices))

# Call the test function
test_triple_grain_separate_permuter()