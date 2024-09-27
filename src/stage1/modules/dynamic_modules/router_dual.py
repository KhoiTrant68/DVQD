import json

import numpy as np
import torch
import torch.nn as nn


class DualGrainFeatureRouter(nn.Module):
    def __init__(self, num_channels, num_groups=None):
        super().__init__()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.gate = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(num_channels * 2, 2),
        )
        self.num_splits = 2
        if num_groups is None:
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        else:
            self.feature_norm_fine = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )
            self.feature_norm_coarse = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )

    def forward(self, h_coarse, h_fine, entropy=None):
        h_coarse = self.feature_norm_coarse(h_coarse)
        h_fine = self.feature_norm_fine(h_fine)
        avg_h_fine = self.gate_pool(h_fine)
        h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0, 2, 3, 1)
        gate = self.gate(h_logistic)
        return gate


class DualGrainEntropyRouter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_gate_from_threshold(self, entropy, threshold):
        gate_fine = (entropy > threshold).bool().long().unsqueeze(-1)
        gate_coarse = (entropy <= threshold).bool().long().unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_fine], dim=-1)
        return gate


class DualGrainFixedEntropyRouter(DualGrainEntropyRouter):
    def __init__(self, json_path, fine_grain_ratito):
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.fine_grain_threshold = content[
            "{}".format(str(int(100 - fine_grain_ratito * 100)))
        ]

    def forward(self, entropy):
        gate = self._get_gate_from_threshold(entropy, self.fine_grain_threshold)
        return gate


class DualGrainDynamicEntropyRouter(DualGrainEntropyRouter):
    def __init__(self, fine_grain_ratito_min=0.01, fine_grain_ratito_max=0.99):
        super().__init__()
        self.fine_grain_ratito_min = fine_grain_ratito_min
        self.fine_grain_ratito_max = fine_grain_ratito_max

    def forward(self, entropy=None):
        fine_grain_threshold = np.random.uniform(
            low=self.fine_grain_ratito_min, high=self.fine_grain_ratito_max
        )
        gate = self._get_gate_from_threshold(entropy, fine_grain_threshold)
        return gate
