import torch.nn as nn
from utils.utils import ActNorm


def weights_init(m):
    """Initialize weights for Conv2d and BatchNorm2d layers."""
    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- use ActNorm instead of BatchNorm
        """
        super().__init__()
        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d
        use_bias = use_actnorm or norm_layer != nn.BatchNorm2d

        layers = []
        in_channels = input_nc
        for i in range(n_layers + 2):
            out_channels = min(ndf * (2 ** i), ndf * 8)
            stride = 1 if i == n_layers + 1 else 2
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=use_bias),
                norm_layer(out_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, True) if i < n_layers + 1 else nn.Identity()
            ])
            in_channels = out_channels

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
