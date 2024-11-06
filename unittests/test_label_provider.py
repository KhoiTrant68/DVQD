import sys

import torch

sys.path.append("..")

from src.stage2.modules.label_provider import (ClassBasedSOSProvider,
                                               PositionAwareSOSProvider)

base_sos_provider = PositionAwareSOSProvider(
    coarse_sos=1026,
    fine_sos=1026,
    coarse_pos_sos=258,
    fine_pos_sos=1026,
    coarse_seg_sos=0,
    fine_seg_sos=1,
)

dummy_input = torch.rand(1, 3, 256, 256)

out = base_sos_provider(dummy_input)

print(out)
