import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class DualGrainSeparatePermuter(nn.Module):
    """A dual-grain permutation module that handles both coarse and fine-grained positions.

    This module processes input indices at two granularity levels:
    - Coarse grain: One code per region
    - Fine grain: Multiple codes per region

    Args:
        coarse_size (int): Height/width size for coarse grain (default: 16)
        fine_size (int): Height/width size for fine grain (default: 32)
        content_pad_code (int): Padding code for content (default: 1024)
        content_eos_code (int): End-of-sequence code for content (default: 1025)
        coarse_position_pad_code (int): Padding code for coarse positions (default: 256)
        coarse_position_eos_code (int): End-of-sequence code for coarse positions (default: 257)
        fine_position_pad_code (int): Padding code for fine positions (default: 1024)
        fine_position_eos_code (int): End-of-sequence code for fine positions (default: 1025)
        position_order (str): Ordering strategy for fine positions ("region-first" or "row-first")
    """

    def __init__(
        self,
        coarse_size=16,
        fine_size=32,
        content_pad_code=1024,
        content_eos_code=1025,
        coarse_position_pad_code=256,
        coarse_position_eos_code=257,
        fine_position_pad_code=1024,
        fine_position_eos_code=1025,
        position_order="region-first",
    ):
        super().__init__()

        # Validate inputs
        if fine_size % coarse_size != 0:
            raise ValueError("fine_size must be divisible by coarse_size")
        if position_order not in ["region-first", "row-first"]:
            raise ValueError("position_order must be 'region-first' or 'row-first'")

        # Initialize dimensions
        self.coarse_size = coarse_size
        self.fine_size = fine_size
        self.ratio_coarse_fine = fine_size // coarse_size
        self.ratio_coarse_fine_square = self.ratio_coarse_fine * self.ratio_coarse_fine

        # Store codes as buffer tensors for efficiency
        self.register_buffer(
            "content_eos_tensor", torch.tensor([content_eos_code], dtype=torch.long)
        )
        self.register_buffer(
            "coarse_position_eos_tensor",
            torch.tensor([coarse_position_eos_code], dtype=torch.long),
        )
        self.register_buffer(
            "fine_position_eos_tensor",
            torch.tensor([fine_position_eos_code], dtype=torch.long),
        )

        # Store configuration
        self.content_pad_code = content_pad_code
        self.content_eos_code = content_eos_code
        self.coarse_position_pad_code = coarse_position_pad_code
        self.coarse_position_eos_code = coarse_position_eos_code
        self.fine_position_pad_code = fine_position_pad_code
        self.fine_position_eos_code = fine_position_eos_code
        self.position_order = position_order

        # Pre-compute position sequences
        self._initialize_position_sequences()

    def _initialize_position_sequences(self):
        """Initialize position sequences for both coarse and fine grains."""
        coarse_positions = torch.arange(self.coarse_size**2, dtype=torch.long)
        self.register_buffer("position_sequence_coarse", coarse_positions)

        fine_positions = torch.arange(self.fine_size**2, dtype=torch.long).view(
            self.fine_size, self.fine_size
        )
        if self.position_order == "region-first":
            fine_positions = rearrange(
                fine_positions,
                "(h1 h2) (w1 w2) -> h1 w1 (h2 w2)",
                h1=self.coarse_size,
                h2=self.ratio_coarse_fine,
                w1=self.coarse_size,
                w2=self.ratio_coarse_fine,
            )
        self.register_buffer("position_sequence_fine", fine_positions)

    def _process_content_and_position(
        self, indices, grain_indices, original_indices=None
    ):
        """Process content and position sequences for both grains."""
        batch_size = len(indices)

        # Process coarse grain
        coarse_content = indices[:, :, :, 0]
        coarse_content_list = [
            torch.cat(
                [coarse_content[i][grain_indices[i] == 0], self.content_eos_tensor]
            )
            for i in range(batch_size)
        ]

        coarse_position_list = [
            torch.cat(
                [
                    self.position_sequence_coarse[grain_indices[i].view(-1) == 0],
                    self.coarse_position_eos_tensor,
                ]
            )
            for i in range(batch_size)
        ]

        # Process fine grain
        if self.position_order == "region-first":
            fine_content_list = [
                torch.cat(
                    [
                        indices[i][grain_indices[i] == 1].view(-1),
                        self.content_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]
            fine_position_list = [
                torch.cat(
                    [
                        self.position_sequence_fine[grain_indices[i] == 1].view(-1),
                        self.fine_position_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]
        else:  # row-first
            fine_grain_indices = grain_indices.repeat_interleave(
                2, dim=-1
            ).repeat_interleave(2, dim=-2)
            fine_content_list = [
                torch.cat(
                    [
                        original_indices[i][fine_grain_indices[i] == 1].view(-1),
                        self.content_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]
            fine_position_list = [
                torch.cat(
                    [
                        self.position_sequence_fine[fine_grain_indices[i] == 1],
                        self.fine_position_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]

        return self._pad_sequences(
            coarse_content_list,
            coarse_position_list,
            fine_content_list,
            fine_position_list,
        )

    def _pad_sequences(
        self,
        coarse_content_list,
        coarse_position_list,
        fine_content_list,
        fine_position_list,
    ):
        """Pad sequences and create segment tensors."""
        coarse_content_tensor = pad_sequence(
            coarse_content_list, batch_first=True, padding_value=self.content_pad_code
        )
        coarse_position_tensor = pad_sequence(
            coarse_position_list,
            batch_first=True,
            padding_value=self.coarse_position_pad_code,
        )
        fine_content_tensor = pad_sequence(
            fine_content_list, batch_first=True, padding_value=self.content_pad_code
        )
        fine_position_tensor = pad_sequence(
            fine_position_list,
            batch_first=True,
            padding_value=self.fine_position_pad_code,
        )

        # Create segment tensors
        coarse_segment_tensor = torch.zeros_like(coarse_content_tensor)
        fine_segment_tensor = torch.ones_like(fine_content_tensor)

        return {
            "coarse_content": coarse_content_tensor,
            "fine_content": fine_content_tensor,
            "coarse_position": coarse_position_tensor,
            "fine_position": fine_position_tensor,
            "coarse_segment": coarse_segment_tensor,
            "fine_segment": fine_segment_tensor,
        }

    def forward(self, indices, grain_indices):
        """Forward pass for the permuter.

        Args:
            indices: Input tensor of shape [B, H, W]
            grain_indices: Binary tensor indicating grain type (0=coarse, 1=fine)

        Returns:
            dict: Contains processed tensors for both coarse and fine grains
        """
        original_indices = indices.clone()
        indices = rearrange(
            indices,
            "B (h1 h2) (w1 w2) -> B h1 w1 (h2 w2)",
            h1=self.coarse_size,
            h2=self.ratio_coarse_fine,
            w1=self.coarse_size,
            w2=self.ratio_coarse_fine,
        )

        return self._process_content_and_position(
            indices, grain_indices, original_indices
        )

    def forward_back(
        self, coarse_content, fine_content, coarse_position, fine_position
    ):
        """Reverse the permutation process.

        Args:
            coarse_content: Tensor of coarse grain content
            fine_content: Tensor of fine grain content
            coarse_position: Tensor of coarse grain positions
            fine_position: Tensor of fine grain positions

        Returns:
            torch.Tensor: Reconstructed indices tensor
        """
        batch_size = len(coarse_content)
        device = coarse_content.device

        target_coarse_idx = torch.zeros(
            batch_size, self.coarse_size**2, device=device, dtype=torch.long
        )
        target_idx = torch.zeros(
            batch_size, self.fine_size**2, device=device, dtype=torch.long
        )

        for i in range(batch_size):
            # Process coarse positions
            coarse_end = (coarse_position[i] == self.coarse_position_eos_code).nonzero(
                as_tuple=True
            )[0]
            if len(coarse_end) > 0:
                coarse_end = coarse_end[0]
                for pos in range(coarse_end):
                    target_coarse_idx[i, coarse_position[i, pos]] = coarse_content[
                        i, pos
                    ]
                target_idx[i] = target_coarse_idx[i].repeat_interleave(4)
                target_idx[i] = rearrange(
                    target_idx[i],
                    "(h1 w1 h2 w2) -> (h1 h2 w1 w2)",
                    h1=self.coarse_size,
                    h2=self.ratio_coarse_fine,
                    w1=self.coarse_size,
                    w2=self.ratio_coarse_fine,
                )

            # Process fine positions
            fine_end = (fine_position[i] == self.fine_position_eos_code).nonzero(
                as_tuple=True
            )[0]
            if len(fine_end) > 0:
                fine_end = fine_end[0]
                for pos in range(fine_end):
                    target_idx[i, fine_position[i, pos]] = fine_content[i, pos]

        return rearrange(
            target_idx,
            "B (h1 h2 w1 w2) -> B (h1 h2) (w1 w2)",
            h1=self.coarse_size,
            h2=self.ratio_coarse_fine,
            w1=self.coarse_size,
            w2=self.ratio_coarse_fine,
        )
