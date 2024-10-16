import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Embedding):
    """Vector Quantization (VQ) embedding module with Exponential Moving Average (EMA) update."""

    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
    ):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        """Compute the distances between input vectors and the codebook.

        Args:
            inputs (torch.Tensor): Input tensor of shape [..., embed_dim].

        Returns:
            torch.Tensor: Distances of shape [..., n_embed].
        """
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(
            *inputs_shape[:-1], -1
        )  # [B, h, w, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        """Find the nearest embedding indices for the input vectors.

        Args:
            inputs (torch.Tensor): Input tensor of shape [..., embed_dim].

        Returns:
            torch.Tensor: Indices of the nearest embeddings.
        """
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        """Tile the input tensor with noise to reach a target number of vectors.

        Args:
            x (torch.Tensor): Input tensor of shape [B, embed_dim].
            target_n (int): Target number of vectors.

        Returns:
            torch.Tensor: Tiled tensor with added noise.
        """
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        """Update the EMA buffers for cluster sizes and embeddings.

        Args:
            vectors (torch.Tensor): Input vectors of shape [..., embed_dim].
            idxs (torch.Tensor): Indices of the nearest embeddings.
        """
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(0, idxs.unsqueeze(0), 1)

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][
                :n_embed
            ]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(1 - usage.view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        """Update the embedding weights using the EMA values."""
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        """Forward pass to quantize the input vectors.

        Args:
            inputs (torch.Tensor): Input tensor of shape [..., embed_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Quantized embeddings and their indices.
        """
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema:
            self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        """Retrieve embeddings for the given indices.

        Args:
            idxs (torch.Tensor): Indices of the embeddings.

        Returns:
            torch.Tensor: Embeddings corresponding to the indices.
        """
        embeds = super().forward(idxs)
        return embeds


class MaskVectorQuantize(nn.Module):
    """Masked Vector Quantization module."""

    def __init__(
        self,
        codebook_size=1024,
        codebook_dim=256,
        accept_image_fmap=True,
        commitment_beta=0.25,
        decay=0.99,
        restart_unused_codes=True,
        channel_last=False,
    ):
        super().__init__()
        self.accept_image_fmap = accept_image_fmap
        self.beta = commitment_beta
        self.channel_last = channel_last
        self.restart_unused_codes = restart_unused_codes
        self.codebook = VQEmbedding(
            codebook_size,
            codebook_dim,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
        )
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x, codebook_mask=None, *ignorewargs, **ignorekwargs):
        """Forward pass for vector quantization with optional masking.

        Args:
            x (torch.Tensor): Input tensor.
            codebook_mask (Optional[torch.Tensor]): Mask for the codebook.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[None, None, torch.Tensor]]:
            Quantized tensor, loss, and additional outputs.
        """
        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        if need_transpose:
            x = rearrange(x, "b d n -> b n d").contiguous()
        flatten = rearrange(x, "h ... d -> h (...) d").contiguous()

        x_q, x_code = self.codebook(flatten)

        if codebook_mask is not None:
            if codebook_mask.dim() == 4:
                codebook_mask = rearrange(
                    codebook_mask, "b c h w -> b (h w) c"
                ).contiguous()
                loss = self.beta * torch.mean(
                    (x_q.detach() - x) ** 2 * codebook_mask
                ) + torch.mean((x_q - x.detach()) ** 2 * codebook_mask)
            else:
                loss = self.beta * torch.mean(
                    (x_q.detach() - x) ** 2 * codebook_mask.unsqueeze(-1)
                ) + torch.mean((x_q - x.detach()) ** 2 * codebook_mask.unsqueeze(-1))
        else:
            loss = self.beta * torch.mean((x_q.detach() - x) ** 2) + torch.mean(
                (x_q - x.detach()) ** 2
            )

        loss *= torch.mean(codebook_mask) if codebook_mask is not None else 1

        x_q = x + (x_q - x).detach()

        if need_transpose:
            x_q = rearrange(x_q, "b n d -> b d n").contiguous()

        if self.accept_image_fmap:
            x_q = rearrange(x_q, "b (h w) c -> b c h w", h=height, w=width).contiguous()
            x_code = rearrange(
                x_code, "b (h w) ... -> b h w ...", h=height, w=width
            ).contiguous()

        return x_q, loss, (None, None, x_code)

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        """Get soft codes for the input vectors.

        Args:
            x (torch.Tensor): Input tensor.
            temp (float): Temperature for softmax.
            stochastic (bool): Whether to sample stochastically.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Soft codes and their indices.
        """
        distances = self.codebook.compute_distances(x)
        soft_code = F.softmax(-distances / temp, dim=-1)

        if stochastic:
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
            code = torch.multinomial(soft_code_flat, 1)
            code = code.reshape(*soft_code.shape[:-1])
        else:
            code = distances.argmin(dim=-1)

        return soft_code, code

    def get_codebook_entry(self, indices, *kwargs):
        """Retrieve codebook entries for the given indices.

        Args:
            indices (torch.Tensor): Indices of the codebook entries.

        Returns:
            torch.Tensor: Codebook entries corresponding to the indices.
        """
        z_q = self.codebook.embed(indices)  # (batch, height, width, channel)
        return z_q
