import json

import numpy as np
import torch
import torch.nn as nn


class DualGrainFeatureRouter(nn.Module):
    def __init__(self, num_channels, num_groups=None):
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
                    else nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=True)
                )
                for _ in range(2)
            ]
        )

    def forward(self, h_coarse, h_fine, entropy=None):
        h_coarse, h_fine = [
            norm(h) for norm, h in zip(self.feature_norm, [h_coarse, h_fine])
        ]

        print(h_coarse.shape, h_fine.shape)
        avg_h_fine = self.gate_pool(h_fine)
        print(avg_h_fine.shape)
        h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0, 2, 3, 1)
        print(h_logistic.shape)
        gate = self.gate(h_logistic)
        return gate


class DualGrainEntropyRouter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_gate_from_threshold(self, entropy, threshold):
        gate_fine = (entropy > threshold).unsqueeze(-1)
        gate_coarse = (entropy <= threshold).unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_fine], dim=-1)
        return gate


class DualGrainFixedEntropyRouter(DualGrainEntropyRouter):
    def __init__(self, json_path, fine_grain_ratito):
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.fine_grain_threshold = content[str(int(100 - fine_grain_ratito * 100))]

    def forward(self, h_coarse, h_fine, entropy):
        gate = self._get_gate_from_threshold(entropy, self.fine_grain_threshold)
        return gate


class DualGrainDynamicEntropyRouter(DualGrainEntropyRouter):
    def __init__(self, fine_grain_ratito_min=0.01, fine_grain_ratito_max=0.99):
        super().__init__()
        self.fine_grain_ratito_min = fine_grain_ratito_min
        self.fine_grain_ratito_max = fine_grain_ratito_max

    def forward(self, h_coarse, h_fine, entropy=None):
        fine_grain_threshold = np.random.uniform(
            low=self.fine_grain_ratito_min, high=self.fine_grain_ratito_max
        )
        gate = self._get_gate_from_threshold(entropy, fine_grain_threshold)
        return gate
