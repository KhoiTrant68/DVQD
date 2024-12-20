from typing import List, Tuple

import torch

from src.stage1.models.base_stage1 import Stage1Model
from src.utils.util_modules import instantiate_from_config


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
    ):
        """
        Initializes the DualGrainVQModel with the given configurations.

        Args:
            encoderconfig: Configuration for the encoder.
            decoderconfig: Configuration for the decoder.
            lossconfig: Configuration for the loss function.
            vqconfig: Configuration for the vector quantization layer.
            quant_before_dim: Number of channels before quantization.
            quant_after_dim: Number of channels after quantization.
            quant_sample_temperature: Temperature for sampling from the quantized embeddings.
            ckpt_path: Path to a checkpoint file to load weights from.
            ignore_keys: List of keys to ignore when loading weights from a checkpoint.
            image_key: Key for the input image in the input batch dictionary.
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

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """
        Encodes an input image into quantized embeddings.

        Args:
            x: Input image tensor.

        Returns:
            A tuple containing:
                - Quantized embeddings (torch.Tensor).
                - Embedding loss (torch.Tensor).
                - Information dictionary from the quantization layer (dict).
                - Grain indices (torch.Tensor).
                - Gate values (torch.Tensor).
        """
        h_dict = self.encoder(x, None)
        h = h_dict["h_dual"]
        grain_indices = h_dict["indices"]
        codebook_mask = h_dict["codebook_mask"]
        gate = h_dict["gate"]

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(
            x=h, temp=self.quant_sample_temperature, codebook_mask=codebook_mask
        )
        return quant, emb_loss, info, grain_indices, gate

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decodes quantized embeddings into an image.

        Args:
            quant: Quantized embeddings.

        Returns:
            Decoded image tensor.
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input: Input image tensor.

        Returns:
            A tuple containing:
                - Decoded image tensor (torch.Tensor).
                - Quantization loss (torch.Tensor).
                - Grain indices (torch.Tensor).
                - Gate values (torch.Tensor).
        """
        quant, diff, _, grain_indices, gate = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, grain_indices, gate
