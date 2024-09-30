import unittest

import torch

from stage1.modules.vector_quantization.masked_vq import (MaskVectorQuantize,
                                                          VQEmbedding)


class TestVQEmbedding(unittest.TestCase):
    def setUp(self):
        self.n_embed = 10
        self.embed_dim = 5
        self.vq_embedding = VQEmbedding(self.n_embed, self.embed_dim)

    def test_forward(self):
        inputs = torch.randn(2, 3, self.embed_dim)  # Batch of 2, 3 vectors
        embeds, embed_idxs = self.vq_embedding(inputs)
        print("embeds", embeds.shape)
        print("embed_idxs", embed_idxs.shape)

        self.assertEqual(embeds.shape, (2, 3, self.embed_dim))
        self.assertEqual(embed_idxs.shape, (2, 3))


class TestMaskVectorQuantize(unittest.TestCase):
    def setUp(self):
        self.codebook_size = 10
        self.codebook_dim = 5
        self.mask_vq = MaskVectorQuantize(self.codebook_size, self.codebook_dim)

    def test_forward(self):
        x = torch.randn(2, self.codebook_dim, 4, 4)  # Batch of 2, 4x4 feature maps
        x_q, loss, _ = self.mask_vq(x)
        print("x_q", x_q.shape)
        print("loss", loss)

        self.assertEqual(x_q.shape, (2, self.codebook_dim, 4, 4))
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
