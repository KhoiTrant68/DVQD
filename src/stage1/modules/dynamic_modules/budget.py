import torch
from torch import nn


class BudgetConstraint_RatioMSE_DualGrain(nn.Module):
    def __init__(
        self,
        target_ratio=0.0,
        gamma=1.0,
        min_grain_size=8,
        max_grain_size=16,
        calculate_all=True,
    ):
        """
        Initializes the BudgetConstraint_RatioMSE_DualGrain.

        Args:
            target_ratio (float): Desired target ratio for the budget.
            gamma (float): Weighting factor for the loss.
            min_grain_size (int): Minimum grain size.
            max_grain_size (int): Maximum grain size.
            calculate_all (bool): Whether to calculate additional loss terms.
        """
        super().__init__()
        self.target_ratio = target_ratio
        self.gamma = gamma
        self.calculate_all = calculate_all
        self.loss = nn.MSELoss()

        self.const = min_grain_size**2
        self.max_const = max_grain_size**2 - self.const

    def forward(self, gate):
        """
        Forward pass to calculate the budget constraint loss.

        Args:
            gate (torch.Tensor): Input tensor representing the gate values.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        beta = (gate[:, 0, :, :] + 4.0 * gate[:, 1, :, :]).sum() / gate.size(
            0
        ) - self.const
        budget_ratio = beta / self.max_const
        target_ratio = torch.full_like(budget_ratio, self.target_ratio)
        loss_budget = self.gamma * self.loss(budget_ratio, target_ratio)

        if self.calculate_all:
            loss_budget_last = self.gamma * self.loss(
                1 - budget_ratio, 1 - target_ratio
            )
            return loss_budget + loss_budget_last

        return loss_budget


class BudgetConstraint_NormedSeperateRatioMSE_TripleGrain(nn.Module):
    """
    Budget Constraint with Normed Separate Ratio MSE for Triple Grain.
    """

    def __init__(
        self,
        target_fine_ratio=0.0,
        target_median_ratio=0.0,
        gamma=1.0,
        min_grain_size=8,
        median_grain_size=16,
        max_grain_size=32,
    ):
        """
        Initializes the BudgetConstraint_NormedSeperateRatioMSE_TripleGrain.
        Args:
            target_fine_ratio (float): Desired target fine ratio for the budget.
            target_median_ratio (float): Desired target median ratio for the budget.
            gamma (float): Weighting factor for the loss.
            min_grain_size (int): Minimum grain size.
            median_grain_size (int): Median grain size.
            max_grain_size (int): Maximum grain size.
        """
        super().__init__()
        assert target_fine_ratio + target_median_ratio <= 1.0
        self.target_fine_ratio = target_fine_ratio
        self.target_median_ratio = target_median_ratio
        self.gamma = gamma
        self.loss = nn.MSELoss()

        self.min_const = min_grain_size**2
        self.median_const = median_grain_size**2 - self.min_const
        self.max_const = max_grain_size**2 - self.min_const

    def forward(self, gate):
        """
        Forward pass to calculate the budget constraint loss.

        Args:
            gate (torch.Tensor): Input tensor representing the gate values.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        batch_size = gate.size(0)
        gate_sum = gate.sum(dim=(0, 2, 3)) / batch_size

        beta_median = gate_sum[0] + 4.0 * gate_sum[1] + gate_sum[2] - self.min_const
        budget_ratio_median = beta_median / self.median_const

        beta_fine = gate_sum[0] + 16.0 * gate_sum[2] + gate_sum[1] - self.min_const
        budget_ratio_fine = beta_fine / self.max_const

        target_ratio_median = torch.full_like(
            budget_ratio_median, self.target_median_ratio
        )
        target_ratio_fine = torch.full_like(budget_ratio_fine, self.target_fine_ratio)

        loss_budget_median = self.loss(budget_ratio_median, target_ratio_median)
        loss_budget_fine = self.gamma * self.loss(budget_ratio_fine, target_ratio_fine)

        return loss_budget_fine + loss_budget_median
