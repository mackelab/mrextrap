"""Convolutional VAE and PatchGAN discriminator used for the MRExtrap autoencoder.

Ported verbatim from the research code: module and attribute names (and hence state-dict keys)
must not change, otherwise released weights no longer load.
"""

import torch
import torch.nn.functional as F
from torch import nn


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        # NOTE(paper): out_channels=None is not usable (conv1 is built with out_channels); always pass it.
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        z_channels: int,
        ch_mult: tuple[int],
        num_res_blocks: int,
        resolution: tuple[int],
        **ignorekwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convolution
        blocks.append(nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = n_channels * in_ch_mult[i]
            block_out_ch = n_channels * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = tuple(ti // 2 for ti in curr_res)

        # normalise and convert to latent size
        blocks.append(Normalize(block_in_ch))

        self.mu_conv = nn.Conv3d(block_in_ch, z_channels, kernel_size=3, stride=1, padding=1)
        self.logsigma_conv = nn.Conv3d(block_in_ch, z_channels, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        mu = self.mu_conv(x)
        # NOTE(paper): log-variance is clamped from above only, to avoid numerical issues.
        logsigma = torch.clamp(self.logsigma_conv(x), max=1.0)
        return mu, logsigma


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        z_channels: int,
        out_channels: int,
        ch_mult: tuple[int],
        num_res_blocks: int,
        resolution: tuple[int],
        **ignorekwargs,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        block_in_ch = n_channels * self.ch_mult[-1]
        curr_res = tuple(ti // 2 ** (self.num_resolutions - 1) for ti in resolution)

        blocks = []
        # initial conv
        blocks.append(nn.Conv3d(z_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = n_channels * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = tuple(ti * 2 for ti in curr_res)

        blocks.append(Normalize(block_in_ch))
        blocks.append(nn.Conv3d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        mu, logsigma = self.encoder(x)
        return mu, logsigma

    def _sample(self, mu, logsigma, sample_precision=1, stochastic_sampling=False):
        if stochastic_sampling:
            std = torch.exp(0.5 * logsigma) / sample_precision
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, x, stochastic_sampling=True):
        mu, logsigma = self.encode(x)
        z = self._sample(mu, logsigma, stochastic_sampling=stochastic_sampling)
        x_hat = self.decode(z)
        return x_hat, (mu, logsigma)

    def kl_loss(self, mu, logsigma, prior_precision=1):
        kl = -0.5 * (1 + logsigma - prior_precision * mu.pow(2) - prior_precision * logsigma.exp())
        return torch.sum(torch.sum(kl.reshape(-1, 1), dim=-1))


# Discriminator network from SynthAnatomy (AmigoLab), itself based on @warvito's baseline:
# https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/discriminator/baseline.py


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class BaselineDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        norm_layer = nn.BatchNorm3d
        use_bias = False  # BatchNorm3d has affine parameters, so convs before it carry no bias

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map

        self.main = nn.Sequential(*sequence)

        self.apply(weights_init)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
