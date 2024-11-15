import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


class TripleGrainSeparatePermuter(nn.Module):
    """
    A module to represent all-grain positions using fine positions,
    separating coarse, medium, and fine positions.

    Args:
        coarse_size (int): Size of the coarse grid.
        medium_size (int): Size of the medium grid.
        fine_size (int): Size of the fine grid.
        content_pad_code (int): Padding code for content.
        content_eos_code (int): End-of-sequence code for content.
        coarse_position_pad_code (int): Padding code for coarse positions.
        coarse_position_eos_code (int): End-of-sequence code for coarse positions.
        medium_position_pad_code (int): Padding code for medium positions.
        medium_position_eos_code (int): End-of-sequence code for medium positions.
        fine_position_pad_code (int): Padding code for fine positions.
        fine_position_eos_code (int): End-of-sequence code for fine positions.
        position_order (str): Order of positions, either "row-first" or "region-first".
    """

    def __init__(
        self,
        coarse_size=8,
        medium_size=16,
        fine_size=32,
        content_pad_code=1024,
        content_eos_code=1025,
        coarse_position_pad_code=256,
        coarse_position_eos_code=257,
        medium_position_pad_code=512,
        medium_position_eos_code=513,
        fine_position_pad_code=1024,
        fine_position_eos_code=1025,
        position_order="region-first",
    ) -> None:
        super(TripleGrainSeparatePermuter, self).__init__()

        # Granularity sizes
        self.hw1 = coarse_size
        self.hw2 = medium_size // coarse_size
        self.hw3 = fine_size // medium_size
        self.fine_size = fine_size

        # Padding and EOS codes for content and positions
        self.content_pad_code = content_pad_code
        self.content_eos_code = content_eos_code
        self.coarse_position_pad_code = coarse_position_pad_code
        self.coarse_position_eos_code = coarse_position_eos_code
        self.medium_position_pad_code = medium_position_pad_code
        self.medium_position_eos_code = medium_position_eos_code
        self.fine_position_pad_code = fine_position_pad_code
        self.fine_position_eos_code = fine_position_eos_code

        # EOS tensors for content and positions
        self.content_eos_tensor = torch.tensor(
            [self.content_eos_code], dtype=torch.long
        )
        self.coarse_position_eos_tensor = torch.tensor(
            [self.coarse_position_eos_code], dtype=torch.long
        )
        self.medium_position_eos_tensor = torch.tensor(
            [self.medium_position_eos_code], dtype=torch.long
        )
        self.fine_position_eos_tensor = torch.tensor(
            [self.fine_position_eos_code], dtype=torch.long
        )

        self.position_order = position_order
        assert self.position_order in ["row-first", "region-first"]

        # Generate position sequences
        self.position_sequence_coarse = torch.arange(coarse_size**2, dtype=torch.long)
        self.position_sequence_medium = torch.arange(
            medium_size**2, dtype=torch.long
        ).view(medium_size, medium_size)
        self.position_sequence_fine = torch.arange(
            fine_size**2, dtype=torch.long
        ).view(fine_size, fine_size)

        if self.position_order == "region-first":
            self.position_sequence_medium = rearrange(
                self.position_sequence_medium,
                "(h1 h2) (w1 w2) -> h1 w1 (h2 w2)",
                h1=self.hw1,
                h2=self.hw2,
                w1=self.hw1,
                w2=self.hw2,
            )
            self.position_sequence_fine = rearrange(
                self.position_sequence_fine,
                "(h2 h3) (w2 w3) -> h2 w2 (h3 w3)",
                h2=medium_size,
                h3=self.hw3,
                w2=medium_size,
                w3=self.hw3,
            )

    def forward(self, indices, grain_indices):
        """
        Forward pass to extract coarse, medium, and fine sequences.
        """
        batch_size = indices.size(0)
        device = indices.device

        # Reshape indices into nested grids
        indices = rearrange(
            indices,
            "B (h1 h2 h3) (w1 w2 w3) -> B h1 w1 (h2 w2) (h3 w3)",
            h1=self.hw1,
            h2=self.hw2,
            h3=self.hw3,
            w1=self.hw1,
            w2=self.hw2,
            w3=self.hw3,
        )

        original_indices = indices.clone()

        # Coarse-grain sequence
        coarse_content = indices[:, :, :, 0, 0]
        coarse_content_list = [
            torch.cat(
                [
                    coarse_content[i][(grain_indices[i] == 0)],
                    self.content_eos_tensor.to(device),
                ]
            )
            for i in range(batch_size)
        ]
        coarse_content_tensor = pad_sequence(
            coarse_content_list, batch_first=True, padding_value=self.content_pad_code
        )

        coarse_position_list = [
            torch.cat(
                [
                    self.position_sequence_coarse[grain_indices[i].view(-1).cpu() == 0],
                    self.coarse_position_eos_tensor.to(device),
                ]
            )
            for i in range(batch_size)
        ]
        coarse_position_tensor = pad_sequence(
            coarse_position_list,
            batch_first=True,
            padding_value=self.coarse_position_pad_code,
        )
        coarse_segment_tensor = torch.zeros_like(coarse_content_tensor).to(device)

        # Medium-grain sequence
        medium_content_list = [
            torch.cat(
                [
                    indices[i][(grain_indices[i] == 1)].view(-1),
                    self.content_eos_tensor.to(device),
                ]
            )
            for i in range(batch_size)
        ]
        medium_content_tensor = pad_sequence(
            medium_content_list, batch_first=True, padding_value=self.content_pad_code
        )

        medium_position_list = [
            torch.cat(
                [
                    self.position_sequence_medium[grain_indices[i] == 1].view(-1),
                    self.medium_position_eos_tensor.to(device),
                ]
            )
            for i in range(batch_size)
        ]
        medium_position_tensor = pad_sequence(
            medium_position_list,
            batch_first=True,
            padding_value=self.medium_position_pad_code,
        )
        medium_segment_tensor = torch.ones_like(medium_content_tensor).to(device)

        # Fine-grain sequence
        fine_content_list = [
            torch.cat(
                [
                    original_indices[i][(grain_indices[i] == 2)].view(-1),
                    self.content_eos_tensor.to(device),
                ]
            )
            for i in range(batch_size)
        ]
        fine_content_tensor = pad_sequence(
            fine_content_list, batch_first=True, padding_value=self.content_pad_code
        )

        fine_position_list = [
            torch.cat(
                [
                    self.position_sequence_fine[grain_indices[i] == 2].view(-1),
                    self.fine_position_eos_tensor.to(device),
                ]
            )
            for i in range(batch_size)
        ]
        fine_position_tensor = pad_sequence(
            fine_position_list,
            batch_first=True,
            padding_value=self.fine_position_pad_code,
        )
        fine_segment_tensor = torch.ones_like(fine_content_tensor).to(device) * 2

        return {
            "coarse_content": coarse_content_tensor,
            "medium_content": medium_content_tensor,
            "fine_content": fine_content_tensor,
            "coarse_position": coarse_position_tensor,
            "medium_position": medium_position_tensor,
            "fine_position": fine_position_tensor,
            "coarse_segment": coarse_segment_tensor,
            "medium_segment": medium_segment_tensor,
            "fine_segment": fine_segment_tensor,
        }

    def reverse(
        self,
        coarse_content,
        medium_content,
        fine_content,
        coarse_position,
        medium_position,
        fine_position,
    ):
        """
        Reverse operation for the TripleGrainSeparatePermuter.

        Args:
            coarse_content (torch.Tensor): Coarse content tensor.
            medium_content (torch.Tensor): Medium content tensor.
            fine_content (torch.Tensor): Fine content tensor.
            coarse_position (torch.Tensor): Coarse position tensor.
            medium_position (torch.Tensor): Medium position tensor.
            fine_position (torch.Tensor): Fine position tensor.

        Returns:
            torch.Tensor: The reconstructed target indices tensor.
        """
        batch_size, coarse_length = coarse_content.size()
        device = coarse_content.device
        medium_length = medium_content.size(1)
        fine_length = fine_content.size(1)

        # Initialize target index tensors
        target_coarse_idx = torch.zeros(batch_size, self.hw1**2, dtype=torch.long).to(
            device
        )
        target_medium_idx = torch.zeros(
            batch_size, (self.hw1 * self.hw2) ** 2, dtype=torch.long
        ).to(device)
        target_idx = torch.zeros(batch_size, self.fine_size**2, dtype=torch.long).to(
            device
        )

        # Reconstruct coarse level
        for i in range(batch_size):
            for current_position in range(coarse_length):
                if (
                    coarse_position[i, current_position]
                    == self.coarse_position_eos_code
                ):
                    break
                else:
                    target_coarse_idx[i, coarse_position[i, current_position]] = (
                        coarse_content[i, current_position]
                    )

        # Expand coarse level indices to medium level grid
        target_medium_expanded = target_coarse_idx.repeat_interleave(
            self.hw2, dim=-1
        ).repeat_interleave(self.hw2, dim=-2)
        target_medium_expanded = rearrange(
            target_medium_expanded,
            "B (h1 w1 h2 w2) -> B (h1 h2) (w1 w2)",
            h1=self.hw1,
            w1=self.hw1,
            h2=self.hw2,
            w2=self.hw2,
        )

        # Reconstruct medium level
        for i in range(batch_size):
            for current_position in range(medium_length):
                if (
                    medium_position[i, current_position]
                    == self.medium_position_eos_code
                ):
                    break
                else:
                    target_medium_idx[i, medium_position[i, current_position]] = (
                        medium_content[i, current_position]
                    )

        # Combine coarse and medium levels to generate fine-level base
        target_fine_expanded = target_medium_idx.repeat_interleave(
            self.hw3, dim=-1
        ).repeat_interleave(self.hw3, dim=-2)
        target_fine_expanded = rearrange(
            target_fine_expanded,
            "B (h2 w2 h3 w3) -> B (h2 h3) (w2 w3)",
            h2=self.hw2,
            w2=self.hw2,
            h3=self.hw3,
            w3=self.hw3,
        )

        # Reconstruct fine level
        for i in range(batch_size):
            for current_position in range(fine_length):
                if fine_position[i, current_position] == self.fine_position_eos_code:
                    break
                else:
                    target_idx[i, fine_position[i, current_position]] = fine_content[
                        i, current_position
                    ]

        # Combine coarse, medium, and fine indices into the original shape
        target_idx = rearrange(
            target_idx,
            "B (h1 h2 h3 w1 w2 w3) -> B (h1 h2 h3) (w1 w2 w3)",
            h1=self.hw1,
            h2=self.hw2,
            h3=self.hw3,
            w1=self.hw1,
            w2=self.hw2,
            w3=self.hw3,
        )

        return target_idx
