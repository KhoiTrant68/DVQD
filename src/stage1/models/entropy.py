import torch
from einops import rearrange
from torch import Tensor, nn


class Entropy(nn.Module):
    def __init__(self, patch_size, image_width, image_height):
        """
        Initializes the Entropy module with the given patch size and image dimensions.

        Args:
            patch_size (int): The size of each patch.
            image_width (int): The width of the input image.
            image_height (int): The height of the input image.
        """
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.patch_size = patch_size
        self.patch_num = int(self.width * self.height / self.patch_size**2)
        self.hw = int(self.width // self.patch_size)
        self.unfold = nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )

        self.register_buffer("bins", torch.linspace(-1, 1, 32))
        self.register_buffer("sigma", torch.tensor(0.01))

    def entropy(self, values: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Computes the entropy of the given values using kernel density estimation.

        Args:
            values (torch.Tensor): The input tensor containing image patches.
            batch_size (int): The batch size of the input images.

        Returns:
            torch.Tensor: The computed entropy for each patch, reshaped to match the
            batch size and patch grid dimensions.
        """
        epsilon = 1e-40

        residuals = values.unsqueeze(1) - self.bins.unsqueeze(0).unsqueeze(-1)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=2)
        pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + epsilon) + epsilon
        entropy = -torch.sum(pdf * torch.log(pdf), dim=1)

        return entropy.view(batch_size, self.hw, self.hw)

    def forward(self, inputs: Tensor) -> torch.Tensor:
        """
        Forward pass to compute the entropy of image patches.

        Args:
            inputs (Tensor): The input image tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The entropy of each patch in the input images.
        """
        batch_size = inputs.shape[0]

        gray_images = (
            0.2989 * inputs[:, 0, :, :]
            + 0.5870 * inputs[:, 1, :, :]
            + 0.1140 * inputs[:, 2, :, :]
        )
        gray_images = gray_images.unsqueeze(1)

        unfolded_images = self.unfold(gray_images)
        unfolded_images = unfolded_images.view(batch_size * self.patch_num, -1)

        return self.entropy(unfolded_images, batch_size)
