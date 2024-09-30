import unittest

import torch

from stage1.modules.dynamic_modules.router_dual import *
from stage1.modules.dynamic_modules.router_triple import *

PATH = "D:\\AwesomeCV\\DynamicVectorQuantization\\scripts\\tools\\thresholds\\entropy_thresholds_imagenet_train_patch-16.json"


class TestDualGrainFeatureRouter(unittest.TestCase):
    def setUp(self):
        self.router = DualGrainFeatureRouter(num_groups=None, num_channels=4)

    def test_forward(self):
        h_coarse = torch.randn(1, 4, 8, 8)
        h_fine = torch.randn(1, 4, 4, 4)
        output = self.router.forward(h_coarse, h_fine)
        print("TestDualGrainFeatureRouter", output.shape)


class TestDualGrainEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = DualGrainEntropyRouter()

    def test_get_gate_from_threshold(self):
        entropy = torch.tensor([0.1, 0.5, 0.9])
        threshold = 0.4
        gate = self.router._get_gate_from_threshold(entropy, threshold)
        print("TestDualGrainEntropyRouter", gate.shape)


class TestDualGrainFixedEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = DualGrainFixedEntropyRouter(json_path=PATH, fine_grain_ratito=0.5)

    def test_forward(self):
        h_coarse = torch.randn(1, 4, 8, 8)
        h_fine = torch.randn(1, 4, 8, 8)
        entropy = torch.tensor([0.1, 0.5, 0.9])
        gate = self.router.forward(h_coarse, h_fine, entropy)
        print("TestDualGrainFixedEntropyRouter", gate.shape)


class TestDualGrainDynamicEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = DualGrainDynamicEntropyRouter(
            fine_grain_ratito_min=0.01, fine_grain_ratito_max=0.99
        )

    def test_forward(self):
        h_coarse = torch.randn(1, 4, 8, 8)
        h_fine = torch.randn(1, 4, 8, 8)
        entropy = torch.tensor([0.1, 0.5, 0.9])
        gate = self.router.forward(h_coarse, h_fine, entropy)

        print("TestDualGrainDynamicEntropyRouter", gate.shape)


class TestTripleGrainFeatureRouter(unittest.TestCase):
    def setUp(self):
        self.router = TripleGrainFeatureRouter(num_channels=16)

    def test_forward(self):
        h_fine = torch.randn(1, 16, 32, 32)
        h_median = torch.randn(1, 16, 16, 16)
        h_coarse = torch.randn(1, 16, 8, 8)
        output = self.router(h_fine, h_median, h_coarse)
        print("TestTripleGrainFeatureRouter", output.shape)


class TestTripleGrainEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = TripleGrainEntropyRouter()

    def test_get_gate_from_threshold(self):
        entropy = torch.tensor([0.1, 0.5, 0.9])
        threshold_median = 0.4
        threshold_fine = 0.8
        gate = self.router._get_gate_from_threshold(
            entropy, threshold_median, threshold_fine
        )
        print("TestTripleGrainEntropyRouter", gate.shape)


class TestTripleGrainFixedEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = TripleGrainFixedEntropyRouter(
            json_path=PATH, median_grain_ratio=0.5, fine_grain_ratio=0.5
        )

    def test_forward(self):
        h_coarse = torch.randn(1, 16, 8, 8)
        h_median = torch.randn(1, 16, 16, 16)
        h_fine = torch.randn(1, 16, 32, 32)
        entropy = torch.tensor([0.1, 0.5, 0.9])
        gate = self.router.forward(h_coarse, h_median, h_fine, entropy)
        print("TestTripleGrainFixedEntropyRouter", gate.shape)


class TestTripleGrainDynamicEntropyRouter(unittest.TestCase):
    def setUp(self):
        self.router = TripleGrainDynamicEntropyRouter()

    def test_forward(self):
        h_coarse = torch.randn(1, 16, 8, 8)
        h_median = torch.randn(1, 16, 16, 16)
        h_fine = torch.randn(1, 16, 32, 32)
        entropy = torch.tensor([0.1, 0.5, 0.9])
        gate = self.router.forward(h_coarse, h_median, h_fine, entropy)
        print("TestTripleGrainDynamicEntropyRouter", gate.shape)


if __name__ == "__main__":
    unittest.main()
