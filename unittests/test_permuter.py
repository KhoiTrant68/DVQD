import sys
import unittest

sys.path.append("..")

import torch

from src.stage2.modules.permuter import DualGrainSeperatePermuter


class TestDualGrainSeperatePermuter(unittest.TestCase):
    def setUp(self):
        self.permuter = DualGrainSeperatePermuter()
        self.batch_size = 2
        self.coarse_size = 16
        self.fine_size = 32
        self.indices = torch.randint(
            0, 100, (self.batch_size, self.fine_size, self.fine_size)
        )
        self.grain_indices = torch.zeros(
            (self.batch_size, self.coarse_size, self.coarse_size)
        )

    def test_forward(self):
        output = self.permuter(self.indices, self.grain_indices)
        self.assertEqual(output["coarse_content"].shape[0], self.batch_size)
        self.assertEqual(output["fine_content"].shape[0], self.batch_size)

    def test_reverse(self):
        forward_output = self.permuter(self.indices, self.grain_indices)
        reverse_output = self.permuter.forward_back(
            forward_output["coarse_content"],
            forward_output["fine_content"],
            forward_output["coarse_position"],
            forward_output["fine_position"],
        )
        self.assertEqual(reverse_output.shape, self.indices.shape)


# class TestTripleGrainSeperatePermuter(unittest.TestCase):
#     def setUp(self):
#         self.permuter = TripleGrainSeperatePermuter()
#         self.batch_size = 2
#         self.coarse_size = 16
#         self.medium_size = 32
#         self.fine_size = 64
#         self.indices = torch.randint(0, 100, (self.batch_size, self.fine_size, self.fine_size))
#         self.grain_indices = torch.zeros((self.batch_size, self.coarse_size, self.coarse_size))

#     def test_forward(self):
#         output = self.permuter(self.indices, self.grain_indices)
#         self.assertEqual(output["coarse_content"].shape[0], self.batch_size)
#         self.assertEqual(output["medium_content"].shape[0], self.batch_size)
#         self.assertEqual(output["fine_content"].shape[0], self.batch_size)

#     def test_reverse(self):
#         forward_output = self.permuter(self.indices, self.grain_indices)
#         reverse_output = self.permuter.reverse(
#             forward_output["coarse_content"],
#             forward_output["medium_content"],
#             forward_output["fine_content"],
#             forward_output["coarse_position"],
#             forward_output["medium_position"],
#             forward_output["fine_position"]
#         )
#         self.assertEqual(reverse_output.shape, self.indices.shape)

if __name__ == "__main__":
    unittest.main()
