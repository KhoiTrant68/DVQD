import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(num_channels):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.2,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3 if conv_shortcut else 1,
                stride=1,
                padding=1 if conv_shortcut else 0,
            )
        else:
            self.shortcut = None

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        h = self.dropout(F.silu(self.norm2(h)))
        h = self.conv2(h)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = group_norm(in_channels)
        self.qkv = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = self.norm(x)
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)

        # compute attention
        b, c, h, w = q.shape
        q = q.view(b, c, -1).transpose(1, 2)  # b, hw, c
        k = k.view(b, c, -1)  # b, c, hw
        v = v.view(b, c, -1)  # b, c, hw

        attn = torch.bmm(q, k) * (c**-0.5)
        attn = F.softmax(attn, dim=2)

        # attend to values
        h_ = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            mean = input.mean(dim=[0, 2, 3], keepdim=True)
            std = input.std(dim=[0, 2, 3], keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if input.dim() == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            logdet = (
                torch.sum(torch.log1p(torch.abs(self.scale) - 1))
                * input.shape[2]
                * input.shape[3]
            )
            return h, logdet.expand(input.shape[0])

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            self.initialize(output)
            self.initialized.fill_(1)

        if output.dim() == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h
