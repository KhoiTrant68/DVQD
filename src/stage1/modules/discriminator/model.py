import torch
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        super().__init__()
        assert affine
        self.logdet = logdet
        self.allow_reverse_init = allow_reverse_init
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).view(1, -1, 1, 1)
            std = flatten.std(1).view(1, -1, 1, 1)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)

        if input.dim() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if input.dim() == 2:
            h = h.view(h.size(0), h.size(1))

        if self.logdet:
            logdet = (
                torch.sum(torch.log(torch.abs(self.scale)))
                * input.size(2)
                * input.size(3)
            )
            return h, logdet.expand(input.size(0))

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if output.dim() == 2:
            output = output.view(output.size(0), output.size(1), 1, 1)

        h = output / self.scale - self.loc

        if output.dim() == 2:
            h = h.view(h.size(0), h.size(1))
        return h


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super().__init__()
        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d
        use_bias = use_actnorm or norm_layer != nn.BatchNorm2d

        layers = []
        in_channels = input_nc
        for i in range(n_layers + 2):
            out_channels = min(ndf * (2**i), ndf * 8)
            stride = 1 if i == n_layers + 1 else 2
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=use_bias,
                )
            )
            if i > 0:
                layers.append(norm_layer(out_channels))
            if i < n_layers + 1:
                layers.append(nn.LeakyReLU(0.2, True))
            in_channels = out_channels

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
