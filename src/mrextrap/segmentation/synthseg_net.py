# ruff: noqa: SIM102, B006  (vendored upstream style, kept for fidelity)
"""SynthSeg network (torch port).

Provenance: torch port of the SynthSeg (robust, v2.0) segmentation network and its
pre/post-processing, as used in the MRExtrap paper. Code in this file is adapted from
snapshot synthseg_torch/predict_synthseg.py, which in turn is derived from the official SynthSeg repository
(https://github.com/BBillot/SynthSeg, B. Billot et al., Apache License 2.0; see
``LICENSE-SynthSeg`` in this directory and THIRD_PARTY_NOTICES.md).
Modifications: split into modules, callable API instead of ``sys.argv``, device-agnostic
(no hard-coded CUDA), unused code removed. Numerics are unchanged.

Weights
-------
The weights (``synthseg_torch.pt``, a plain ``state_dict`` of ``SynthSeg``) are NOT part of this
repository. They are a conversion of the official SynthSeg 2.0 robust TensorFlow/Keras model
(``synthseg_2.0.h5`` from https://github.com/BBillot/SynthSeg) into this torch architecture:
Keras ``Conv3D`` kernels ``(kd, kh, kw, in, out)`` are transposed to torch ``(out, in, kd, kh, kw)``,
``BatchNormalization`` gamma/beta/moving mean/moving variance map to
``weight``/``bias``/``running_mean``/``running_var`` (eps 1e-3), and layers are matched in
network order (unet1, unet2, unet3; encoder blocks, then decoder blocks, then the 1x1 output conv).
Same-padding is emulated by ``Conv3dSamePadding``/``MaxPool3dSamePadding``. The torch output was
compared against the TF output on ADNI scans by the author (the comparison script is not included).
# The conversion script is not included.

Where to put the file: ``$MREXTRAP_SYNTHSEG_WEIGHTS`` (full path) or ``<run_dir>/weights/synthseg_torch.pt``;
see ``docs/weights.md`` (``mrextrap_weights.zip``) for the download. The upstream
licence (Apache-2.0) applies to the weights as derived work; check the terms before redistributing.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class MaxPool3dSamePadding(nn.MaxPool3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        in_depth, in_height, in_width = input.shape[2:]  # Added depth dimension

        if type(self.stride) is not tuple:
            self.stride = (self.stride, self.stride, self.stride)  # Three dimensions stride
        if type(self.kernel_size) is not tuple:
            self.kernel_size = (
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )  # Three dimensions kernel

        # Padding for depth
        if in_depth % self.stride[0] == 0:
            pad_along_depth = max(self.kernel_size[0] - self.stride[0], 0)
        else:
            pad_along_depth = max(self.kernel_size[0] - (in_depth % self.stride[0]), 0)

        # Padding for height
        if in_height % self.stride[1] == 0:
            pad_along_height = max(self.kernel_size[1] - self.stride[1], 0)
        else:
            pad_along_height = max(self.kernel_size[1] - (in_height % self.stride[1]), 0)

        # Padding for width
        if in_width % self.stride[2] == 0:
            pad_along_width = max(self.kernel_size[2] - self.stride[2], 0)
        else:
            pad_along_width = max(self.kernel_size[2] - (in_width % self.stride[2]), 0)

        pad_front = pad_along_depth // 2  # padding for depth dimension
        pad_back = pad_along_depth - pad_front
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # Adjusted to three dimensions, order corrected to (pad_left, pad_right, pad_bottom, pad_top, pad_front, pad_back)
        input = F.pad(
            input,
            (pad_left, pad_right, pad_bottom, pad_top, pad_front, pad_back),
            value=float("-inf"),
        )

        self.padding = (0, 0, 0)  # Adjusted to three dimensions

        # Using max_pool3d instead of max_pool2d
        return F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class Conv3dSamePadding(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        in_depth, in_height, in_width = input.shape[2:]

        if type(self.stride) is not tuple:
            self.stride = (self.stride, self.stride, self.stride)
        if type(self.kernel_size) is not tuple:
            self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)

        # Padding for depth
        if in_depth % self.stride[0] == 0:
            pad_along_depth = max(self.kernel_size[0] - self.stride[0], 0)
        else:
            pad_along_depth = max(self.kernel_size[0] - (in_depth % self.stride[0]), 0)

        # Padding for height
        if in_height % self.stride[1] == 0:
            pad_along_height = max(self.kernel_size[1] - self.stride[1], 0)
        else:
            pad_along_height = max(self.kernel_size[1] - (in_height % self.stride[1]), 0)

        # Padding for width
        if in_width % self.stride[2] == 0:
            pad_along_width = max(self.kernel_size[2] - self.stride[2], 0)
        else:
            pad_along_width = max(self.kernel_size[2] - (in_width % self.stride[2]), 0)

        pad_front = pad_along_depth // 2
        pad_back = pad_along_depth - pad_front
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        input = F.pad(
            input, (pad_left, pad_right, pad_bottom, pad_top, pad_front, pad_back), value=0.0
        )
        self.padding = (0, 0, 0)

        return self._conv_forward(input, self.weight, self.bias)


class DownBlock(nn.Module):
    def __init__(self, num_ch, num_ch_in=None, last_block=False, first_block=False, conv_size=3):
        super().__init__()
        if not first_block:
            num_ch_in = num_ch // 2
        self.conv1 = Conv3dSamePadding(num_ch_in, num_ch, conv_size)
        self.act = nn.ELU()
        self.conv2 = Conv3dSamePadding(num_ch, num_ch, conv_size)
        self.bn = nn.BatchNorm3d(num_ch, eps=1e-03, momentum=0.99)
        self.maxpool = MaxPool3dSamePadding(2)
        self.last_block = last_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        skip_ = x
        x = self.bn(x)
        if not self.last_block:
            x = self.maxpool(x)
        return x, skip_


class Encoder(nn.Module):
    def __init__(self, initial_ch=32, num_blocks=5, num_ch_in=1, conv_size=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                DownBlock(
                    initial_ch * (2**i),
                    last_block=(i == num_blocks - 1),
                    first_block=(i == 0),
                    num_ch_in=num_ch_in,
                    conv_size=conv_size,
                )
            )

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            x, skip_ = block(x)
            if i < len(self.blocks) - 1:
                skips.append(skip_)
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, num_ch, num_ch_out=None, skip_connection=True, conv_size=3):
        super().__init__()
        if num_ch_out is None:
            num_ch_out = num_ch // 2
        if skip_connection:
            num_ch = int(num_ch * 1.5)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = Conv3dSamePadding(num_ch, num_ch_out, conv_size)
        self.act = nn.ELU()
        self.conv2 = Conv3dSamePadding(num_ch_out, num_ch_out, conv_size)
        self.bn = nn.BatchNorm3d(num_ch_out, eps=1e-03, momentum=0.99)
        self.skip_connection = skip_connection

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.skip_connection:
            x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, initial_ch, num_blocks=5, num_ch_out=5, conv_size=3, mid_unet=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks - 1):  # i goes from 0 to num_blocks-1, innermost block is 0
            skip_connection = True
            if mid_unet:
                if i == num_blocks - 2 or i == num_blocks - 3:
                    skip_connection = False
            self.blocks.append(
                UpBlock(
                    initial_ch * (2 ** (num_blocks - i - 1)),
                    skip_connection=skip_connection,
                    num_ch_out=None,
                    conv_size=conv_size,
                )
            )
        self.blocks.append(Conv3dSamePadding(initial_ch, num_ch_out, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, skips=[], softmax=True):
        # summary printing
        if not len(skips):
            # print("Decoder input shape: ", x.shape)
            for i in range(len(self.blocks) - 1):
                skips.append(
                    torch.randn(
                        x.shape[0],
                        x.shape[1] // (2 ** (i + 1)),
                        x.shape[2] * (2 ** (i + 1)),
                        x.shape[3] * (2 ** (i + 1)),
                        x.shape[4] * (2 ** (i + 1)),
                    )
                )
                # print("Decoder skip shape: ", skips[-1].shape)
        for i, block in enumerate(self.blocks[:-1]):
            skip = skips[i]
            # print("input shape: ", x.shape)
            # print("skip shape: ", skip.shape)
            x = block(x, skip)
        x = self.blocks[-1](x)
        if softmax:
            x = self.softmax(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_ch_in, num_ch_out, num_ch, num_blocks=5, conv_size=3, mid_unet=False):
        super().__init__()
        self.encoder = Encoder(
            initial_ch=num_ch, num_blocks=num_blocks, num_ch_in=num_ch_in, conv_size=conv_size
        )
        self.decoder = Decoder(
            initial_ch=num_ch,
            num_blocks=num_blocks,
            num_ch_out=num_ch_out,
            conv_size=conv_size,
            mid_unet=mid_unet,
        )

    def forward(self, x, softmax=True):
        x, skips = self.encoder(x)
        # print(x.shape)
        # for skip in skips:
        #     print(skip.shape)
        # return x
        return self.decoder(x, skips[::-1], softmax=softmax)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class OneHotDiscretize(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

    def forward(self, x):
        x_type = x.dtype
        x = torch.argmax(x, dim=1)
        x = (
            F.one_hot(x.to(torch.int64), num_classes=self.n_class)
            .type(x_type)
            .permute(0, 4, 1, 2, 3)
        )
        return x


class SynthSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet1 = UNet(1, 5, 24, num_blocks=5)
        self.discrete1 = OneHotDiscretize(5)
        self.unet2 = UNet(5, 5, 16, num_blocks=5, conv_size=5, mid_unet=True)
        self.discrete2 = OneHotDiscretize(5)
        self.unet3 = UNet(6, 32, 24, num_blocks=5)

    def forward(self, x, softmax=True):
        x_input = x
        x = self.unet1(x)
        x = self.discrete1(x)
        x = self.unet2(x)
        x = self.discrete2(x)
        x = torch.cat([x_input, x], dim=1)
        x = self.unet3(x, softmax=softmax)
        return x


WEIGHTS_ENV_VAR = "MREXTRAP_SYNTHSEG_WEIGHTS"
WEIGHTS_FILENAME = "synthseg_torch.pt"


def resolve_weights_path(weights=None, run_dir=None) -> Path:
    """Locate the SynthSeg torch weights.

    Order: explicit ``weights`` argument, ``$MREXTRAP_SYNTHSEG_WEIGHTS``,
    ``<run_dir>/weights/synthseg_torch.pt`` (also ``<run_dir>/synthseg_torch.pt``).
    Raises ``FileNotFoundError`` with instructions if nothing is found.
    """
    candidates = []
    if weights is not None:
        candidates.append(Path(weights))
    env = os.environ.get(WEIGHTS_ENV_VAR)
    if env:
        candidates.append(Path(env))
    if run_dir is not None:
        candidates.append(Path(run_dir) / "weights" / WEIGHTS_FILENAME)
        candidates.append(Path(run_dir) / WEIGHTS_FILENAME)
    for c in candidates:
        if c.is_file():
            return c
    tried = ", ".join(str(c) for c in candidates) or "(no candidate locations given)"
    raise FileNotFoundError(
        f"SynthSeg weights '{WEIGHTS_FILENAME}' not found (tried: {tried}). They are not part of "
        f"the repository: download mrextrap_weights.zip (docs/weights.md) and unzip it into <run_dir>, "
        f"or set {WEIGHTS_ENV_VAR}=/path/to/{WEIGHTS_FILENAME}."
    )


def load_synthseg(weights=None, run_dir=None, device=None) -> SynthSeg:
    """Build ``SynthSeg``, load weights, set eval mode and move to ``device``.

    ``device`` defaults to CUDA when available, else CPU.
    """
    path = resolve_weights_path(weights, run_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = SynthSeg()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net.to(device)
