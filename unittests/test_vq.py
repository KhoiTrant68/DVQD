import sys

sys.path.append("..")
import torch

from src.stage1.modules.vector_quantization.masked_vq import (
    MaskVectorQuantize, VQEmbedding)


def run_test_vq_embedding():
    n_embed = 1024
    embed_dim = 256
    vq_embedding = VQEmbedding(n_embed, embed_dim)

    inputs = torch.randn(4, n_embed, embed_dim)
    embeds, embed_idxs = vq_embedding(inputs)
    print("VQEmbedding Output Shapes:", embeds.shape, embed_idxs.shape)


def run_test_mask_vector_quantize():
    codebook_size = 1024
    codebook_dim = 256
    mask_vq = MaskVectorQuantize(codebook_size, codebook_dim)

    x = torch.randn(4, codebook_dim, 32, 32)  # Batch of 2, 4x4 feature maps
    x_q, loss, _ = mask_vq(x)
    print("MaskVectorQuantize Output Shapes:", x_q.shape, loss.shape)


if __name__ == "__main__":
    run_test_vq_embedding()
    run_test_mask_vector_quantize()
