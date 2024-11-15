import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


class DualGrainSeparatePermuter(nn.Module):
    """
    A module to represent all-grain positions using fine positions,
    separating coarse and fine positions.

    Args:
        coarse_size (int): Size of the coarse grid. Default is 16.
        fine_size (int): Size of the fine grid. Default is 32.
        content_pad_code (int): Padding code for content. Default is 1024.
        content_eos_code (int): End-of-sequence code for content. Default is 1025.
        coarse_position_pad_code (int): Padding code for coarse positions. Default is 256.
        coarse_position_eos_code (int): End-of-sequence code for coarse positions. Default is 257.
        fine_position_pad_code (int): Padding code for fine positions. Default is 1024.
        fine_position_eos_code (int): End-of-sequence code for fine positions. Default is 1025.
        position_order (str): Order of positions, either "row-first" or "region-first". Default is "region-first".
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
    ) -> None:
        super(DualGrainSeparatePermuter, self).__init__()

        self.hw1 = coarse_size
        self.hw2 = fine_size // coarse_size
        self.fine_size = fine_size
        self.hw2_square = int(self.hw2 * self.hw2)

        self.content_pad_code = content_pad_code
        self.content_eos_code = content_eos_code
        self.coarse_position_pad_code = coarse_position_pad_code
        self.coarse_position_eos_code = coarse_position_eos_code
        self.fine_position_pad_code = fine_position_pad_code
        self.fine_position_eos_code = fine_position_eos_code

        self.content_eos_tensor = torch.tensor(
            [self.content_eos_code], dtype=torch.long
        )
        self.coarse_position_eos_tensor = torch.tensor(
            [self.coarse_position_eos_code], dtype=torch.long
        )
        self.fine_position_eos_tensor = torch.tensor(
            [self.fine_position_eos_code], dtype=torch.long
        )

        self.position_order = position_order
        assert self.position_order in ["row-first", "region-first"]
        self.position_sequence_coarse = torch.arange(
            int(coarse_size**2), dtype=torch.long
        )
        self.position_sequence_fine = torch.arange(
            int(fine_size**2), dtype=torch.long
        ).view(fine_size, fine_size)
        if self.position_order == "region-first":
            self.position_sequence_fine = rearrange(
                self.position_sequence_fine,
                "(h1 h2) (w1 w2) -> h1 w1 (h2 w2)",
                h1=self.hw1,
                h2=self.hw2,
                w1=self.hw1,
                w2=self.hw2,
            )

    def forward(self, indices, grain_indices):
        """
        Forward pass for the DualGrainSeparatePermuter.

        Args:
            indices (torch.Tensor): Input indices tensor.
            grain_indices (torch.Tensor): Grain indices tensor.

        Returns:
            dict: A dictionary containing coarse and fine content, position, and segment tensors.
        """
        batch_size = indices.size(0)
        device = indices.device

        original_indices = indices.clone()

        indices = rearrange(
            indices,
            "B (h1 h2) (w1 w2) -> B h1 w1 (h2 w2)",
            h1=self.hw1,
            h2=self.hw2,
            w1=self.hw1,
            w2=self.hw2,
        )

        # Coarse-grain sequence
        coarse_content = indices[:, :, :, 0]
        coarse_content_list = [
            torch.cat(
                [
                    coarse_content[i][(grain_indices[i] == 0)].to(device),
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
                    self.position_sequence_coarse[
                        grain_indices[i].view(-1).cpu() == 0
                    ].to(device),
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

        coarse_segment_tensor = (
            torch.zeros_like(coarse_content_tensor).to(device).long()
        )

        # Fine-grain sequence
        if self.position_order == "region-first":
            fine_content_list = [
                torch.cat(
                    [
                        indices[i][grain_indices[i] == 1].to(device).view(-1),
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
                        self.position_sequence_fine[grain_indices[i] == 1]
                        .view(-1)
                        .to(device),
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
        elif self.position_order == "row-first":
            fine_grain_indices = grain_indices.repeat_interleave(
                2, dim=-1
            ).repeat_interleave(2, dim=-2)
            fine_content_list = [
                torch.cat(
                    [
                        original_indices[i][fine_grain_indices[i] == 1]
                        .view(-1)
                        .to(device),
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
                        self.position_sequence_fine[
                            fine_grain_indices[i].cpu() == 1
                        ].to(device),
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
        else:
            raise NotImplementedError(
                "{} is not supported yet!".format(self.position_order)
            )

        fine_segment_tensor = torch.ones_like(fine_content_tensor).to(device).long()

        return_dict = {
            "coarse_content": coarse_content_tensor,
            "fine_content": fine_content_tensor,
            "coarse_position": coarse_position_tensor,
            "fine_position": fine_position_tensor,
            "coarse_segment": coarse_segment_tensor,
            "fine_segment": fine_segment_tensor,
        }
        return return_dict

    def reverse(self, coarse_content, fine_content, coarse_position, fine_position):
        """
        Reverse operation for the DualGrainSeparatePermuter.

        Args:
            coarse_content (torch.Tensor): Coarse content tensor.
            fine_content (torch.Tensor): Fine content tensor.
            coarse_position (torch.Tensor): Coarse position tensor.
            fine_position (torch.Tensor): Fine position tensor.

        Returns:
            torch.Tensor: The reconstructed target indices tensor.
        """
        batch_size, coarse_length = coarse_content.size()
        device = coarse_content.device
        fine_length = fine_content.size(1)

        target_coarse_idx = torch.zeros(
            batch_size, int(self.hw1) ** 2, dtype=torch.long
        ).to(device)
        target_idx = torch.zeros(
            batch_size, int(self.fine_size) ** 2, dtype=torch.long
        ).to(device)

        for i in range(batch_size):
            for current_position in range(coarse_length):
                if (
                    coarse_position[i, current_position]
                    == self.coarse_position_eos_code
                ):
                    target_idx[i] = target_coarse_idx[i].repeat_interleave(4, dim=-1)
                    target_idx[i] = rearrange(
                        target_idx[i],
                        "(h1 w1 h2 w2) -> (h1 h2 w1 w2)",
                        h1=self.hw1,
                        h2=self.hw2,
                        w1=self.hw1,
                        w2=self.hw2,
                    )
                    break
                else:
                    target_coarse_idx[i, coarse_position[i, current_position]] = (
                        coarse_content[i, current_position]
                    )

            for current_position in range(fine_length):
                if fine_position[i, current_position] == self.fine_position_eos_code:
                    break
                else:
                    target_idx[i, fine_position[i, current_position]] = fine_content[
                        i, current_position
                    ]

        target_idx = rearrange(
            target_idx,
            "B (h1 h2 w1 w2) -> B (h1 h2) (w1 w2)",
            h1=self.hw1,
            h2=self.hw2,
            w1=self.hw1,
            w2=self.hw2,
        )
        return target_idx
