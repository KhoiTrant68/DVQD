import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class TripleGrainSeparatePermuter(nn.Module):
    def __init__(
        self,
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
        position_order="region-first",
    ):
        super().__init__()
        if fine_size % medium_size != 0 or medium_size % coarse_size != 0:
            raise ValueError(
                "Each grain size must be divisible by the previous grain size"
            )
        if position_order not in ["region-first", "row-first"]:
            raise ValueError("position_order must be 'region-first' or 'row-first'")

        self.coarse_size = coarse_size
        self.medium_size = medium_size
        self.fine_size = fine_size
        self.ratio_coarse_medium = medium_size // coarse_size
        self.ratio_medium_fine = fine_size // medium_size
        self.ratio_coarse_fine = fine_size // coarse_size

        self.register_buffer(
            "content_eos_tensor", torch.tensor([content_eos_code], dtype=torch.long)
        )
        self.register_buffer(
            "coarse_position_eos_tensor",
            torch.tensor([coarse_position_eos_code], dtype=torch.long),
        )
        self.register_buffer(
            "medium_position_eos_tensor",
            torch.tensor([medium_position_eos_code], dtype=torch.long),
        )
        self.register_buffer(
            "fine_position_eos_tensor",
            torch.tensor([fine_position_eos_code], dtype=torch.long),
        )

        self.content_pad_code = content_pad_code
        self.content_eos_code = content_eos_code
        self.coarse_position_pad_code = coarse_position_pad_code
        self.coarse_position_eos_code = coarse_position_eos_code
        self.medium_position_pad_code = medium_position_pad_code
        self.medium_position_eos_code = medium_position_eos_code
        self.fine_position_pad_code = fine_position_pad_code
        self.fine_position_eos_code = fine_position_eos_code
        self.position_order = position_order

        self._initialize_position_sequences()

    def _initialize_position_sequences(self):
        coarse_positions = torch.arange(self.coarse_size**2, dtype=torch.long)
        self.register_buffer("position_sequence_coarse", coarse_positions)

        medium_positions = torch.arange(self.medium_size**2, dtype=torch.long).view(
            self.medium_size, self.medium_size
        )
        if self.position_order == "region-first":
            medium_positions = rearrange(
                medium_positions,
                "(h1 h2) (w1 w2) -> h1 w1 (h2 w2)",
                h1=self.coarse_size,
                h2=self.ratio_coarse_medium,
                w1=self.coarse_size,
                w2=self.ratio_coarse_medium,
            )
        self.register_buffer("position_sequence_medium", medium_positions)

        fine_positions = torch.arange(self.fine_size**2, dtype=torch.long).view(
            self.fine_size, self.fine_size
        )
        if self.position_order == "region-first":
            fine_positions = rearrange(
                fine_positions,
                "(h1 h2) (w1 w2) -> h1 w1 (h2 w2)",
                h1=self.medium_size,
                h2=self.ratio_medium_fine,
                w1=self.medium_size,
                w2=self.ratio_medium_fine,
            )
        self.register_buffer("position_sequence_fine", fine_positions)

    def _process_content_and_position(
        self, indices, grain_indices, original_indices=None
    ):
        batch_size = len(indices)

        # Process coarse grain (grain_indices == 0)
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

        # Process medium grain (grain_indices == 1)
        medium_indices = rearrange(
            indices,
            "B (h1 h2) (w1 w2) c -> B h1 w1 (h2 w2 c)",
            h1=self.medium_size,
            h2=self.ratio_medium_fine,
            w1=self.medium_size,
            w2=self.ratio_medium_fine,
        )
        medium_grain_indices = rearrange(
            grain_indices,
            "B (h1 h2) (w1 w2) -> B h1 w1 (h2 w2)",
            h1=self.medium_size,
            h2=self.ratio_medium_fine,
            w1=self.medium_size,
            w2=self.ratio_medium_fine,
        )

        medium_content_list = [
            torch.cat(
                [
                    medium_indices[i][medium_grain_indices[i] == 1].view(-1),
                    self.content_eos_tensor,
                ]
            )
            for i in range(batch_size)
        ]
        medium_position_list = [
            torch.cat(
                [
                    self.position_sequence_medium[medium_grain_indices[i] == 1].view(
                        -1
                    ),
                    self.medium_position_eos_tensor,
                ]
            )
            for i in range(batch_size)
        ]

        # Process fine grain (grain_indices == 2)
        if self.position_order == "region-first":
            fine_grain_mask = grain_indices == 2
            fine_content_list = [
                torch.cat(
                    [indices[i][fine_grain_mask[i]].view(-1), self.content_eos_tensor]
                )
                for i in range(batch_size)
            ]
            fine_position_list = [
                torch.cat(
                    [
                        self.position_sequence_fine[fine_grain_mask[i]].view(-1),
                        self.fine_position_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]
        else:  # row-first
            fine_grain_indices = grain_indices.repeat_interleave(
                2, dim=-1
            ).repeat_interleave(2, dim=-2)
            fine_grain_mask = fine_grain_indices == 2
            fine_content_list = [
                torch.cat(
                    [
                        original_indices[i][fine_grain_mask[i]].view(-1),
                        self.content_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]
            fine_position_list = [
                torch.cat(
                    [
                        self.position_sequence_fine[fine_grain_mask[i]],
                        self.fine_position_eos_tensor,
                    ]
                )
                for i in range(batch_size)
            ]

        return self._pad_sequences(
            coarse_content_list,
            coarse_position_list,
            medium_content_list,
            medium_position_list,
            fine_content_list,
            fine_position_list,
        )

    def _pad_sequences(
        self,
        coarse_content_list,
        coarse_position_list,
        medium_content_list,
        medium_position_list,
        fine_content_list,
        fine_position_list,
    ):
        coarse_content_tensor = pad_sequence(
            coarse_content_list, batch_first=True, padding_value=self.content_pad_code
        )
        coarse_position_tensor = pad_sequence(
            coarse_position_list,
            batch_first=True,
            padding_value=self.coarse_position_pad_code,
        )
        medium_content_tensor = pad_sequence(
            medium_content_list, batch_first=True, padding_value=self.content_pad_code
        )
        medium_position_tensor = pad_sequence(
            medium_position_list,
            batch_first=True,
            padding_value=self.medium_position_pad_code,
        )
        fine_content_tensor = pad_sequence(
            fine_content_list, batch_first=True, padding_value=self.content_pad_code
        )
        fine_position_tensor = pad_sequence(
            fine_position_list,
            batch_first=True,
            padding_value=self.fine_position_pad_code,
        )

        coarse_segment_tensor = torch.zeros_like(coarse_content_tensor)
        medium_segment_tensor = torch.ones_like(medium_content_tensor)
        fine_segment_tensor = torch.full_like(fine_content_tensor, 2)

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

    def forward(self, indices, grain_indices):
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
        self,
        coarse_content,
        medium_content,
        fine_content,
        coarse_position,
        medium_position,
        fine_position,
    ):
        batch_size = len(coarse_content)
        device = coarse_content.device

        target_coarse_idx = torch.zeros(
            batch_size, self.coarse_size**2, device=device, dtype=torch.long
        )
        target_medium_idx = torch.zeros(
            batch_size, self.medium_size**2, device=device, dtype=torch.long
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

                # Upscale coarse to medium
                target_medium_idx[i] = target_coarse_idx[i].repeat_interleave(
                    self.ratio_coarse_medium**2
                )
                target_medium_idx[i] = rearrange(
                    target_medium_idx[i],
                    "(h1 w1 h2 w2) -> (h1 h2 w1 w2)",
                    h1=self.coarse_size,
                    h2=self.ratio_coarse_medium,
                    w1=self.coarse_size,
                    w2=self.ratio_coarse_medium,
                )

            # Process medium positions
            medium_end = (medium_position[i] == self.medium_position_eos_code).nonzero(
                as_tuple=True
            )[0]
            if len(medium_end) > 0:
                medium_end = medium_end[0]
                for pos in range(medium_end):
                    target_medium_idx[i, medium_position[i, pos]] = medium_content[
                        i, pos
                    ]

                # Upscale medium to fine
                target_idx[i] = target_medium_idx[i].repeat_interleave(
                    self.ratio_medium_fine**2
                )
                target_idx[i] = rearrange(
                    target_idx[i],
                    "(h1 w1 h2 w2) -> (h1 h2 w1 w2)",
                    h1=self.medium_size,
                    h2=self.ratio_medium_fine,
                    w1=self.medium_size,
                    w2=self.ratio_medium_fine,
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
            "B (h1 h2) (w1 w2) -> B h1 w2",
            h1=self.fine_size,
            w1=self.fine_size,
        )
