from typing import List, Tuple

import torch

from src.stage1.models.base_stage1 import Stage1Model
from src.stage1.models.entropy import Entropy
from src.utils.util_modules import disabled_train, instantiate_from_config


class DualGrainVQModel(Stage1Model):
    """
    Dual Grain VQ Model for image generation.
    """

    def __init__(
        self,
        encoderconfig,
        decoderconfig,
        lossconfig,
        vqconfig,
        quant_before_dim: int,
        quant_after_dim: int,
        quant_sample_temperature: float = 0.0,
        ckpt_path: str = None,
        ignore_keys: List[str] = [],
        image_key: str = "image",
        entropy_patch_size: int = 16,
        image_size: int = 256,
        **ignore_kwargs,
    ):
        """
        Initialize the DualGrainVQModel.

        This method sets up the encoder, decoder, quantization, and entropy calculation
        components of the model, and optionally loads weights from a checkpoint.

        Args:
            encoderconfig: Configuration for the encoder.
            decoderconfig: Configuration for the decoder.
            lossconfig: Configuration for the loss function.
            vqconfig: Configuration for the vector quantization layer.
            quant_before_dim (int): Number of channels before quantization.
            quant_after_dim (int): Number of channels after quantization.
            quant_sample_temperature (float, optional): Temperature for sampling from the quantized embeddings.
            ckpt_path (str, optional): Path to a checkpoint file to load weights from.
            ignore_keys (List[str], optional): List of keys to ignore when loading weights from a checkpoint.
            image_key (str, optional): Key for the input image in the input batch dictionary.
            entropy_patch_size (int, optional): Patch size for entropy calculation.
            image_size (int, optional): Size of the input image.
        """
        super().__init__()

        self.image_key = image_key
        self.encoder = instantiate_from_config(encoderconfig)
        self.decoder = instantiate_from_config(decoderconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(vqconfig)

        self.quant_conv = torch.nn.Conv2d(quant_before_dim, quant_after_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(quant_after_dim, quant_before_dim, 1)
        self.quant_sample_temperature = quant_sample_temperature

        self.entropy_patch_size = entropy_patch_size
        self.image_size = image_size
        self.entropy_calculation = Entropy(entropy_patch_size, image_size, image_size)
        self.entropy_calculation = self.entropy_calculation.eval()
        self.entropy_calculation.train = disabled_train

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Encodes an input image into quantized embeddings.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            Tuple: A tuple containing:
                - Quantized embeddings.
                - Embedding loss.
                - Information dictionary from the quantization layer.
                - Grain indices.
                - Gate values.
                - Entropy map.
        """
        x_entropy = self.entropy_calculation(x)
        h_dict = self.encoder(x, x_entropy)
        h = h_dict["h_dual"]
        grain_indices = h_dict["indices"]
        codebook_mask = h_dict["codebook_mask"]
        gate = h_dict["gate"]

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(
            x=h, temp=self.quant_sample_temperature, codebook_mask=codebook_mask
        )
        return quant, emb_loss, info, grain_indices, gate, x_entropy

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decodes quantized embeddings into an image.

        Args:
            quant (torch.Tensor): Quantized embeddings.

        Returns:
            torch.Tensor: Decoded image tensor.
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input (torch.Tensor): Input image tensor.

        Returns:
            Tuple: A tuple containing:
                - Decoded image tensor.
                - Quantization loss.
                - Grain indices.
                - Gate values.
                - Entropy map.
        """
        quant, diff, _, grain_indices, gate, x_entropy = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, grain_indices, gate, x_entropy
