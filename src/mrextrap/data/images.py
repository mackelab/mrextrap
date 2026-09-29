"""Image loading, intensity normalisation and cropping for preprocessed T1 volumes."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import nibabel as nib
import numpy as np
from monai.transforms import EnsureChannelFirst, SpatialCrop

DEFAULT_CROPPING_OFFSET = (11, 16, 0)
DEFAULT_CROP_SIZE = (160, 192, 160)


class ReadImage:
    """Load a NIfTI file and return its voxel data (raw dataobj, no scaling to float64)."""

    def __call__(self, pth):
        return nib.load(pth).dataobj[:]


def clip_and_normalize_image(img, method="minmax0mean", normalize_to_1=True):
    """Clip to [0, 99.7th percentile], then normalise intensities.

    ``minmax0mean`` maps to [-1, 1] (``normalize_to_1=True``) or [-1.5, 1.5].
    """
    # NOTE(paper): clip at the 0.997 quantile of the whole volume (background included).
    img = np.clip(
        img,
        0,
        np.quantile(img.reshape(-1), q=0.997),
    )
    if method == "minmax0mean":
        # NOTE(paper): the +0.001 in the denominator means the maximum maps to slightly below +1.
        if normalize_to_1:
            return 2 * ((img - np.min(img)) / (np.max(img) - np.min(img) + 0.001) - 0.5)
        else:
            return 3 * ((img - np.min(img)) / (np.max(img) - np.min(img) + 0.001) - 0.5)
    elif method == "minmax":
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 0.001)
    else:
        return (img - np.mean(img)) / np.std(img)


class ClipAndNormalizeImage:
    def __init__(self, normalize, normalize_to_1):
        self.normalize = normalize
        self.normalize_to_1 = normalize_to_1

    def __call__(self, img):
        return clip_and_normalize_image(
            img, method=self.normalize, normalize_to_1=self.normalize_to_1
        )


class ToTensor:
    def __call__(self, np_data):
        return np_data.as_tensor()


class ToFloat:
    def __call__(self, t):
        return t.float()


class Compose:
    """Minimal sequential composition (avoids a torchvision dependency)."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_t1_transform(
    cropping_offset: Sequence[int] = DEFAULT_CROPPING_OFFSET,
    size: Sequence[int] = DEFAULT_CROP_SIZE,
    normalize: str = "minmax0mean",
    normalize_to_1: bool = True,
) -> Compose:
    """path -> normalised float tensor ``[1, *size]`` (read, clip+normalise, add channel, crop)."""
    start = tuple(int(o) for o in cropping_offset)
    end = tuple(o + int(s) for o, s in zip(start, size))
    return Compose(
        [
            ReadImage(),
            ClipAndNormalizeImage(normalize, normalize_to_1=normalize_to_1),
            EnsureChannelFirst(channel_dim="no_channel"),
            SpatialCrop(roi_start=start, roi_end=end),
            ToTensor(),
            ToFloat(),
        ]
    )
