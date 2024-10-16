import json

import numpy as np
import torch
import torch.nn as nn


class DualGrainFeatureRouter(nn.Module):
    """
    Dual Grain Feature Router for dynamic feature selection.
    """

    def __init__(self, num_channels, num_groups=None):
        """
        Initializes the DualGrainFeatureRouter.

        This constructor sets up the pooling, gating, and normalization layers for the router.

        Args:
            num_channels (int): Number of input channels.
            num_groups (int, optional): Number of groups for group normalization. Defaults to None.
        """
        super().__init__()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.num_splits = 2

        self.gate = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(num_channels * 2, 2),
        )
        self.feature_norm = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if num_groups is None
                    else nn.GroupNorm(
                        num_groups=num_groups,
                        num_channels=num_channels,
                        eps=1e-6,
                        affine=True,
                    )
                )
                for _ in range(2)
            ]
        )

    def forward(self, h_coarse, h_fine, entropy=None):
        """
        Forward pass to compute gate values for feature selection.

        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_fine (torch.Tensor): Fine grain features.
            entropy (torch.Tensor, optional): Entropy map for dynamic routing. Defaults to None.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        h_coarse, h_fine = [
            norm(h) for norm, h in zip(self.feature_norm, [h_coarse, h_fine])
        ]
        avg_h_fine = self.gate_pool(h_fine)
        h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0, 2, 3, 1)
        gate = self.gate(h_logistic)
        return gate


class DualGrainEntropyRouter(nn.Module):
    """
    Dual Grain Entropy Router for threshold-based feature selection.
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DualGrainEntropyRouter.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def _get_gate_from_threshold(self, entropy, threshold):
        """
        Computes gate values based on entropy threshold.
        Args:
            entropy (torch.Tensor): Entropy map for dynamic routing.
            threshold (float): Threshold value for feature selection.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        gate_fine = (entropy > threshold).float().unsqueeze(-1)
        gate_coarse = (entropy <= threshold).float().unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_fine], dim=-1)
        return gate


class DualGrainFixedEntropyRouter(DualGrainEntropyRouter):
    """
    Dual Grain Fixed Entropy Router for static threshold-based feature selection.

    Args:
        json_path (str): Path to the JSON file containing threshold values.
        fine_grain_ratito (float): Ratio for fine grain feature selection.
    """

    def __init__(self, json_path, fine_grain_ratito):
        """
        Initializes the DualGrainFixedEntropyRouter.
        Args:
            json_path (str): Path to the JSON file containing threshold values.
            fine_grain_ratito (float): Ratio for fine grain feature selection.
        """
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.fine_grain_threshold = content[str(int(100 - fine_grain_ratito * 100))]

    def forward(self, h_coarse, h_fine, entropy):
        """
        Forward pass to compute gate values using a fixed entropy threshold.
        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_fine (torch.Tensor): Fine grain features.
            entropy (torch.Tensor): Entropy map for dynamic routing.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        gate = self._get_gate_from_threshold(entropy, self.fine_grain_threshold)
        return gate


class DualGrainDynamicEntropyRouter(DualGrainEntropyRouter):
    """
    Dual Grain Dynamic Entropy Router for adaptive threshold-based feature selection.
    """

    def __init__(self, fine_grain_ratio_min=0.01, fine_grain_ratio_max=0.99):
        """
        Initializes the DualGrainDynamicEntropyRouter.
        Args:
            fine_grain_ratio_min (float, optional): Minimum ratio for fine grain feature selection. Defaults to 0.01.
            fine_grain_ratio_max (float, optional): Maximum ratio for fine grain feature selection. Defaults to 0.99.
        """
        super().__init__()
        self.fine_grain_ratio_min = fine_grain_ratio_min
        self.fine_grain_ratio_max = fine_grain_ratio_max

    def forward(self, h_coarse, h_fine, entropy=None):
        """
        Forward pass to compute gate values using a dynamic entropy threshold.

        Args:
            h_coarse (torch.Tensor): Coarse grain features.
            h_fine (torch.Tensor): Fine grain features.
            entropy (torch.Tensor, optional): Entropy map for dynamic routing. Defaults to None.

        Returns:
            torch.Tensor: Gate values for feature selection.
        """
        fine_grain_threshold = np.random.uniform(
            low=self.fine_grain_ratio_min, high=self.fine_grain_ratio_max
        )
        gate = self._get_gate_from_threshold(entropy, fine_grain_threshold)
        return gate
