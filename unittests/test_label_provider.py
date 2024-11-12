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


class_sos_provider = ClassBasedSOSProvider(
    threshold_content=1026,
    threshold_coarse_position=258,
    threshold_fine_position=1026,
    coarse_seg_sos=0,
    fine_seg_sos=1,
)
dummy_input_1 = torch.rand(4, 3, 256, 256)
dummy_input_2 = torch.Tensor([443, 443, 443, 443])

out_1 = base_sos_provider(dummy_input_1)
out_2 = class_sos_provider(dummy_input_2)
print(out_1)
print("================================================")

print(out_2)
