import torch
import torch.nn as nn


class AbstractEncoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BaseSOSProvider(AbstractEncoder):
    def __init__(
        self, coarse_pos_sos, fine_pos_sos=None, coarse_seg_sos=None, fine_seg_sos=None
    ):
        super().__init__()
        self.coarse_pos_sos = coarse_pos_sos
        self.fine_pos_sos = fine_pos_sos
        self.coarse_seg_sos = coarse_seg_sos
        self.fine_seg_sos = fine_seg_sos

    def _get_sos_tensors(self, value, batch_size, device):
        if value is not None:
            return torch.ones(batch_size, 1, dtype=torch.long, device=device) * value
        return None

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        c_pos_coarse = self._get_sos_tensors(self.coarse_pos_sos, batch_size, device)
        c_pos_fine = self._get_sos_tensors(self.fine_pos_sos, batch_size, device)
        c_seg_coarse = self._get_sos_tensors(self.coarse_seg_sos, batch_size, device)
        c_seg_fine = self._get_sos_tensors(self.fine_seg_sos, batch_size, device)

        return None, None, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine


class PositionAwareSOSProvider(BaseSOSProvider):
    def __init__(
        self,
        coarse_sos,
        coarse_pos_sos,
        fine_sos=None,
        fine_pos_sos=None,
        coarse_seg_sos=None,
        fine_seg_sos=None,
    ):
        super().__init__(coarse_pos_sos, fine_pos_sos, coarse_seg_sos, fine_seg_sos)
        self.coarse_sos = coarse_sos
        self.fine_sos = fine_sos

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        c_coarse = self._get_sos_tensors(self.coarse_sos, batch_size, device)
        c_fine = self._get_sos_tensors(self.fine_sos, batch_size, device)

        return c_coarse, c_fine, *super().forward(x)[2:]


class ClassBasedSOSProvider(BaseSOSProvider):
    def __init__(
        self,
        threshold_content,
        threshold_coarse_position,
        threshold_fine_position=None,
        coarse_seg_sos=None,
        fine_seg_sos=None,
    ):
        super().__init__(
            threshold_coarse_position,
            threshold_fine_position,
            coarse_seg_sos,
            fine_seg_sos,
        )
        self.threshold_content = threshold_content

    def forward(self, x):
        device = x.device
        c_coarse = (x[:, None] + self.threshold_content).long().to(device)
        c_fine = c_coarse if self.fine_seg_sos is not None else None

        if isinstance(self.coarse_pos_sos, int):
            c_pos_coarse = (x[:, None] + self.coarse_pos_sos).long().to(device)
            c_pos_fine = (
                (x[:, None] + self.fine_pos_sos).long().to(device)
                if self.fine_pos_sos is not None
                else None
            )
        else:
            c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine = super().encode(x)[
                2:
            ]  # Reuse logic if using SOS tokens

        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine
