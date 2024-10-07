import json

import numpy as np
import torch
import torch.nn as nn


class TripleGrainFeatureRouter(nn.Module):
    def __init__(
        self,
        num_channels,
        num_groups=None,
    ):
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
        h_coarse, h_median, h_fine = [
            norm(h) for norm, h in zip(self.feature_norm, [h_coarse, h_median, h_fine])
        ]
        avg_h_fine = self.gate_fine_pool(h_fine)
        avg_h_median = self.gate_median_pool(h_median)
        print(h_coarse.shape, avg_h_median.shape, avg_h_fine.shape)
        h_logistic = torch.cat([h_coarse, avg_h_median, avg_h_fine], dim=1).permute(
            0, 2, 3, 1
        )
        gate = self.gate(h_logistic)
        return gate


class TripleGrainEntropyRouter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_gate_from_threshold(self, entropy, threshold_median, threshold_fine):
        gate_fine = (entropy > threshold_fine).unsqueeze(-1)
        gate_median = (
            (entropy <= threshold_fine) & (entropy > threshold_median)
        ).unsqueeze(-1)
        gate_coarse = (entropy <= threshold_median).unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_median, gate_fine], dim=-1)
        return gate


class TripleGrainFixedEntropyRouter(TripleGrainEntropyRouter):
    def __init__(self, json_path, median_grain_ratio, fine_grain_ratio):
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.median_grain_threshold = content[str(int(100 - median_grain_ratio * 100))]
        self.fine_grain_threshold = content[str(int(100 - fine_grain_ratio * 100))]

    def forward(self, h_coarse, h_median, h_fine, entropy):
        gate = self._get_gate_from_threshold(
            entropy, self.median_grain_threshold, self.fine_grain_threshold
        )
        return gate


class TripleGrainDynamicEntropyRouter(TripleGrainEntropyRouter):
    def __init__(self, fine_grain_ratio_min=0.01, fine_grain_ratio_max=0.99):
        super().__init__()
        self.fine_grain_ratio_min = fine_grain_ratio_min
        self.fine_grain_ratio_max = fine_grain_ratio_max

    def forward(self, h_coarse, h_median, h_fine, entropy=None):
        median_grain_threshold, fine_grain_threshold = np.random.uniform(
            low=[self.fine_grain_ratio_min, self.fine_grain_ratio_max / 2],
            high=[self.fine_grain_ratio_max / 2, self.fine_grain_ratio_max],
        )
        gate = self._get_gate_from_threshold(
            entropy, median_grain_threshold, fine_grain_threshold
        )
        return gate
