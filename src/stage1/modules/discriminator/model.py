import torch.nn as nn

from src.utils.nn_modules import ActNorm, weights_init


class NLayerDiscriminator(nn.Module):
    """
    N-Layer Discriminator for adversarial networks.
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """
        Initializes the NLayerDiscriminator.
        Args:
            input_nc (int): Number of input channels.
            ndf (int): Number of filters in the last convolutional layer.
            n_layers (int): Number of convolutional layers.
            use_actnorm (bool): Whether to use activation normalization.
        """
        super().__init__()
        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d
        use_bias = use_actnorm or norm_layer != nn.BatchNorm2d

        layers = []
        in_channels = input_nc
        for i in range(n_layers + 2):
            out_channels = min(ndf * (2**i), ndf * 8)
            stride = 1 if i == n_layers + 1 else 2
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=stride,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(out_channels) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, True) if i < n_layers + 1 else nn.Identity(),
                ]
            )
            in_channels = out_channels

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        """
        Forward pass of the discriminator.

        Args:
            input (torch.Tensor): Input tensor to the discriminator.

        Returns:
            torch.Tensor: Output tensor representing the authenticity score.
        """
        return self.main(input)
