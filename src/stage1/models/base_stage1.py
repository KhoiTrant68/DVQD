import abc
from typing import Dict, Tuple

import torch
from torch import nn

from src.utils.draw import draw_dual_grain_256res_color


class Stage1Model(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for a stage 1 model.

    Attributes:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        quantize (nn.Module): Quantization module.
    """

    def __init__(self):
        """Initialize the Stage1Model with encoder, decoder, and quantization modules."""
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.quantize = None

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate the encoded codebook from the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded codebook tensor.
        """
        pass

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate the decoded image from the given code.

        Args:
            z (torch.Tensor): Code tensor.

        Returns:
            torch.Tensor: Decoded image tensor.
        """
        pass

    def init_from_ckpt(self, path: str, ignore_keys: list = None) -> None:
        """
        Initialize model from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
            ignore_keys (list, optional): List of keys to ignore when loading the state dict.
        """
        if ignore_keys is None:
            ignore_keys = []

        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch: Dict, k: str) -> torch.Tensor:
        """
        Get input tensor from a batch dictionary.

        Args:
            batch (Dict): Batch dictionary containing input data.
            k (str): Key for the input tensor in the batch.

        Returns:
            torch.Tensor: Input tensor.
        """
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def calculate_loss(
        self,
        x: torch.Tensor,
        xrec: torch.Tensor,
        qloss: torch.Tensor,
        step: int,
        optimizer_idx: int,
        gate: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the loss for either autoencoder or discriminator.

        Args:
            x (torch.Tensor): Input image tensor.
            xrec (torch.Tensor): Reconstructed image tensor.
            qloss (torch.Tensor): Quantization loss.
            step (int): Current training step (epoch or global step).
            optimizer_idx (int): Index of the optimizer being used (0 for AE, 1 for Disc).
            gate (torch.Tensor, optional): Gate values.

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing the calculated loss and a log dictionary.
        """
        kwargs = {
            "codebook_loss": qloss,
            "inputs": x,
            "reconstructions": xrec,
            "optimizer_idx": optimizer_idx,
            "global_step": step,
            "last_layer": self.get_last_layer(),
            "split": "train",
        }
        if gate is not None:
            kwargs["gate"] = gate

        loss, log_dict = self.loss(**kwargs)

        return loss, log_dict

    def get_last_layer(self) -> torch.Tensor:
        """
        Return the weights of the last layer of the decoder.

        Returns:
            torch.Tensor: Weights of the last decoder layer.
        """
        try:
            return self.decoder.conv_out.weight
        except AttributeError:
            return self.decoder.last_layer

    def log_images(
        self, device, mode: str, batch: dict, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Log input images, reconstructions, grain maps, and entropy maps.

        Args:
            device: Device to which tensors are moved.
            mode (str): Mode of operation, either 'feat' or other.
            batch (dict): Batch dictionary containing input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing logged image tensors.
        """
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(device)
        x_entropy = None
        if mode == "feat":
            xrec, _, grain_indices, _ = self(x)
        else:
            xrec, _, grain_indices, _, x_entropy = self(x)

        log["inputs"] = x
        log["reconstructions"] = xrec
        log["grain_map"] = draw_dual_grain_256res_color(
            images=x.clone(), indices=grain_indices, scaler=0.7
        )
        if x_entropy is not None:
            x_entropy = x_entropy.sub(x_entropy.min()).div(
                max(x_entropy.max() - x_entropy.min(), 1e-5)
            )
            log["entropy_map"] = draw_dual_grain_256res_color(
                images=x.clone(), indices=x_entropy, scaler=0.7
            )
        return log

    def get_code_emb_with_depth(
        self, code: torch.Tensor, mode="entropy"
    ) -> torch.Tensor:
        """
        Retrieve codebook embeddings for given codes.

        Args:
            code (torch.Tensor): Code tensor.
            mode (str, optional): Mode of model. Choices are ['entropy', 'feat']. Defaults to "entropy".

        Returns:
            torch.Tensor: Codebook embeddings for the given codes.
        """
        if mode == "entropy":
            embed = self.quantize.get_codebook_entry(code)
        else:
            embed = self.quantize.embed_code_with_depth(code)
        return embed
