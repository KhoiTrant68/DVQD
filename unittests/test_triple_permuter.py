import sys

import torch

sys.path.append("..")
from src.stage2.modules.triple_permuter import TripleGrainSeparatePermuter

if __name__ == "__main__":
    test_code = 1
    position_order = "region-first"
    if test_code == 1:
        # test code 1
        x1 = torch.randint(0, 1024, (2, 16, 16))  # .cuda()
        x2 = 4 * torch.ones_like(x1)  # .cuda()
        x3 = 8 * torch.ones_like(x1)  # .cuda()

        grain_indices = torch.randint(0, 3, (2, 4, 4))  # .cuda()
        grain_indices_repeat_medium = grain_indices.repeat_interleave(
            2, dim=-1
        ).repeat_interleave(2, dim=-2)
        grain_indices_repeat_fine = grain_indices.repeat_interleave(
            4, dim=-1
        ).repeat_interleave(4, dim=-2)

        original_indices = (
            x1 * (grain_indices_repeat_fine == 0)
            + x2 * (grain_indices_repeat_fine == 1)
            + x3 * (grain_indices_repeat_fine == 2)
        )
        print("original_indices-------------", original_indices.shape)
        print("grain_indices-------------", grain_indices.shape)

        permuter = TripleGrainSeparatePermuter(
            coarse_size=4,
            medium_size=8,
            fine_size=16,
            content_pad_code=1024,
            content_eos_code=1025,
            coarse_position_pad_code=256,
            coarse_position_eos_code=257,
            medium_position_pad_code=1024,
            medium_position_eos_code=1025,
            fine_position_pad_code=2048,
            fine_position_eos_code=2049,
            position_order=position_order,
        )
        out = permuter(original_indices, grain_indices)

        (
            coarse_content,
            medium_content,
            fine_content,
            coarse_position,
            medium_position,
            fine_position,
        ) = (
            out["coarse_content"],
            out["medium_content"],
            out["fine_content"],
            out["coarse_position"],
            out["medium_position"],
            out["fine_position"],
        )
        target_fine = permuter.forward_back(
            coarse_content,
            medium_content,
            fine_content,
            coarse_position,
            medium_position,
            fine_position,
        )
        print(target_fine)
        print(torch.all(target_fine == original_indices))
