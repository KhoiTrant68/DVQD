import json

import numpy as np
import torch
import torch.nn as nn


class TripleGrainFeatureRouter(nn.Module):
    """
    Triple Grain Feature Router for dynamic feature selection.
    """

    def __init__(self, num_channels, num_groups=None):
        """
        Initializes the TripleGrainFeatureRouter.

        Args:
            num_channels (int): Number of input channels.
            num_groups (int, optional): Number of groups for group normalization. Defaults to None.
        """
        super().__init__()
        self.gate_median_pool = nn.AvgPool2d(2, 2)
        self.gate_fine_pool = nn.AvgPool2d(4, 4)
        self.num_splits = 3

        self.gate = nn.Sequential(
            nn.Linear(num_channels * self.num_splits, num_channels * self.num_splits),
            nn.SiLU(inplace=True),
            nn.Linear(num_channels * self.num_splits, self.num_splits),
        )

        self.feature_norm = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if num_groups is None
                    else nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=True)
                )
                for _ in range(3)
            ]
        )

    def forward(self, h_coarse, h_median, h_fine, entropy=None):
        """
        Forward pass to compute gate values for feature selection.

        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_median (torch.Tensor): Median grain features.
            h_fine (torch.Tensor): Fine grain features.
            entropy (torch.Tensor, optional): Entropy map for dynamic routing. Defaults to None.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        h_coarse, h_median, h_fine = [
            norm(h) for norm, h in zip(self.feature_norm, [h_coarse, h_median, h_fine])
        ]
        avg_h_fine = self.gate_fine_pool(h_fine)
        avg_h_median = self.gate_median_pool(h_median)
        h_logistic = torch.cat([h_coarse, avg_h_median, avg_h_fine], dim=1).permute(
            0, 2, 3, 1
        )
        gate = self.gate(h_logistic)
        return gate


class TripleGrainEntropyRouter(nn.Module):
    """
    Triple Grain Entropy Router for threshold-based feature selection.
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the TripleGrainEntropyRouter.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def _get_gate_from_threshold(self, entropy, threshold_median, threshold_fine):
        """
        Computes gate values based on entropy thresholds.
        Args:
            entropy (torch.Tensor): Entropy map for dynamic routing.
            threshold_median (float): Threshold value for median grain feature selection.
            threshold_fine (float): Threshold value for fine grain feature selection.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        gate_fine = (entropy > threshold_fine).float().unsqueeze(-1)
        gate_median = (
            ((entropy <= threshold_fine) & (entropy > threshold_median))
            .float()
            .unsqueeze(-1)
        )
        gate_coarse = (entropy <= threshold_median).float().unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_median, gate_fine], dim=-1)
        return gate


class TripleGrainFixedEntropyRouter(TripleGrainEntropyRouter):
    """
    Triple Grain Fixed Entropy Router for static threshold-based feature selection.
    """

    def __init__(self, json_path, median_grain_ratio, fine_grain_ratio):
        """
        Initializes the TripleGrainFixedEntropyRouter.
        Args:
            json_path (str): Path to the JSON file containing threshold values.
            median_grain_ratio (float): Ratio for median grain feature selection.
            fine_grain_ratio (float): Ratio for fine grain feature selection.
        """
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.median_grain_threshold = content[str(int(100 - median_grain_ratio * 100))]
        self.fine_grain_threshold = content[str(int(100 - fine_grain_ratio * 100))]

    def forward(self, h_coarse, h_median, h_fine, entropy):
        """
        Forward pass to compute gate values using fixed entropy thresholds.
        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_median (torch.Tensor): Median grain features.
            h_fine (torch.Tensor): Fine grain features.
            entropy (torch.Tensor): Entropy map for dynamic routing.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        gate = self._get_gate_from_threshold(
            entropy, self.median_grain_threshold, self.fine_grain_threshold
        )
        return gate


class TripleGrainDynamicEntropyRouter(TripleGrainEntropyRouter):
    """
    Triple Grain Dynamic Entropy Router for adaptive threshold-based feature selection.

    Args:
        fine_grain_ratio_min (float, optional): Minimum ratio for fine grain feature selection. Defaults to 0.01.
        fine_grain_ratio_max (float, optional): Maximum ratio for fine grain feature selection. Defaults to 0.99.
    """

    def __init__(self, fine_grain_ratio_min=0.01, fine_grain_ratio_max=0.99):
        """
        Initializes the TripleGrainDynamicEntropyRouter.
        Args:
            fine_grain_ratio_min (float, optional): Minimum ratio for fine grain feature selection. Defaults to 0.01.
            fine_grain_ratio_max (float, optional): Maximum ratio for fine grain feature selection. Defaults to 0.99.
        """
        super().__init__()
        self.fine_grain_ratio_min = fine_grain_ratio_min
        self.fine_grain_ratio_max = fine_grain_ratio_max

    def forward(self, h_coarse, h_median, h_fine, entropy=None):
        """
        Forward pass to compute gate values using dynamic entropy thresholds.

        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_median (torch.Tensor): Median grain features.
            h_fine (torch.Tensor): Fine grain features.
            entropy (torch.Tensor, optional): Entropy map for dynamic routing. Defaults to None.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        median_grain_threshold, fine_grain_threshold = np.random.uniform(
            low=[self.fine_grain_ratio_min, self.fine_grain_ratio_max / 2],
            high=[self.fine_grain_ratio_max / 2, self.fine_grain_ratio_max],
        )
        gate = self._get_gate_from_threshold(
            entropy, median_grain_threshold, fine_grain_threshold
        )
        return gate
