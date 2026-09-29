# ruff: noqa: UP031, TRY002  (vendored upstream style, kept for fidelity)
"""SynthSeg pre-processing, file IO and the prediction pipeline.

Provenance: torch port of the SynthSeg (robust, v2.0) segmentation network and its
pre/post-processing, as used in the MRExtrap paper. Code in this file is adapted from
snapshot synthseg_torch/predict_synthseg.py, which in turn is derived from the official SynthSeg repository
(https://github.com/BBillot/SynthSeg, B. Billot et al., Apache License 2.0; see
``LICENSE-SynthSeg`` in this directory and THIRD_PARTY_NOTICES.md).
Modifications: split into modules, callable API instead of ``sys.argv``, device-agnostic
(no hard-coded CUDA), unused code removed. Numerics are unchanged.
"""

import csv
import fnmatch
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from mrextrap.segmentation.synthseg_net import load_synthseg
from mrextrap.segmentation.synthseg_postprocess import (
    align_volume_to_ref,
    build_binary_structure,  # noqa: F401  (re-exported for parity with upstream utils)
    find_closest_number_divisible_by_m,
    flip_volume,
    get_dims,
    get_ras_axes,
    postprocess,
    reformat_to_list,
)

LABELS_DIR = Path(__file__).parent / "labels"
N_LEVELS = 5


def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with a identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith((".nii", ".nii.gz", ".mgz", ".npz")), (
        "Unknown data file: %s" % path_volume
    )

    if path_volume.endswith((".nii", ".nii.gz", ".mgz")):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)["vol_data"]
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if "int" in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(
            volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims
        )

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """

    mkdir(os.path.dirname(path))
    if ".npz" in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == "FS":
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            # print("sdfkshf")
            if "int" in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=np.int16)
            nifty.set_data_dtype(np.int16)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :return: volume (if return_volume is true), and corresponding info. If aff_ref is not None, the returned aff is
    the original one, i.e. the affine of the image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if ".nii" in path_volume:
        data_res = np.array(header["pixdim"][1 : n_dims + 1])
    elif ".mgz" in path_volume:
        data_res = np.array(header["delta"])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        ras_axes = get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
        im = align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz whithin a folder."""
    basename = os.path.basename(path_dir)
    # print(path_dir, basename)
    # if include_single_image & (
    #     ('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)
    # ):
    if include_single_image and (
        ("final.nii.gz" in basename)
        | ("final.nii" in basename)
        | ("final.mgz" in basename)
        | ("final.npz" in basename)
    ):
        assert os.path.isfile(path_dir), "file %s does not exist" % path_dir
        list_images = [path_dir]
    else:
        # print(path_dir)
        if os.path.isdir(path_dir):
            # list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
            #                      glob.glob(os.path.join(path_dir, '*nii')) +
            #                      glob.glob(os.path.join(path_dir, '*.mgz')) +
            #                      glob.glob(os.path.join(path_dir, '*.npz')))

            list_images = sorted(
                # recursive_glob(path_dir, "*reg_stripped_final.nii.gz")
                recursive_glob(path_dir, "*final.nii.gz")
                # + recursive_glob(path_dir, "*final.nii")
                # + recursive_glob(path_dir, "*final.mgz")
                # + recursive_glob(path_dir, "*final.npz")
            )
            # print(list_images)
        else:
            raise Exception(
                "extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz" % path_dir
            )
        if check_if_empty:
            assert len(list_images) > 0, (
                "no .nii, .nii.gz, .mgz or .npz image could be found in %s" % path_dir
            )
    return list_images


def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == "/":
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)


# ---------------------------------------------- shape-related functions -----------------------------------------------


def add_axis(x, axis=0):
    """Add axis to a numpy array.
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time.
    """
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


def rescale_volume(
    volume,
    new_min=0,
    new_max=255,
    min_percentile=2,
    max_percentile=98,
    use_positive_only=False,
):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = (
        np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    )
    robust_max = (
        np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)
    )

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


def crop_volume(
    volume,
    cropping_margin=None,
    cropping_shape=None,
    aff=None,
    return_crop_idx=False,
    mode="center",
):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. The cropping margin is applied on both sides.
    Can be an int, sequence or 1d numpy array of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None, and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), (
        "cropping_margin or cropping_shape should be provided"
    )
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), (
        "only one of cropping_margin or cropping_shape should be provided"
    )

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [
            vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i]
            for i in range(n_dims)
        ]
    else:
        cropping_shape = reformat_to_list(cropping_shape, length=n_dims)
        if mode == "center":
            min_crop_idx = np.maximum(
                [int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0
            )
            max_crop_idx = np.minimum(
                [min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                np.array(vol_shape)[:n_dims],
            )
        elif mode == "random":
            crop_max_val = np.maximum(
                np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0
            )
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(
                min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims]
            )
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0] : crop_idx[2], crop_idx[1] : crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[
            crop_idx[0] : crop_idx[3],
            crop_idx[1] : crop_idx[4],
            crop_idx[2] : crop_idx[5],
            ...,
        ]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)


def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims, dtype="int")

    # check if need to pad
    if np.any(
        np.array(padding_shape, dtype=np.int32) > np.array(vol_shape[:n_dims], dtype=np.int32)
    ):
        # get padding margins
        min_margins = np.maximum(
            np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)),
            0,
        )
        max_margins = np.maximum(
            np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)),
            0,
        )
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode="constant", constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)


def resample_volume(volume, aff, new_vox_size, interpolation="linear"):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = RegularGridInterpolator(
        (x, y, z), volume_filt, method=interpolation
    )

    start = -(factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing="ij", sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def preprocess(path_image, n_levels=5, crop=None, min_pad=None, path_resample=None):
    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = get_volume_info(path_image, True)
    if n_dims < 3:
        raise ValueError("input should have 3 dimensions, had %s" % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        raise ValueError("input should have 3 dimensions, had %s" % n_dims)
    elif n_channels > 1:
        print("WARNING: detected more than 1 channel, only keeping the first channel.")
        im = im[..., 0]

    # resample image if necessary
    if np.any((im_res > 1.05) | (im_res < 0.95)):
        im_res = np.array([1.0] * 3)
        im, aff = resample_volume(im, aff, im_res)
        if path_resample is not None:
            save_volume(im, aff, h, path_resample)

    # align image
    im = align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = reformat_to_list(crop, length=n_dims, dtype="int")
        crop_shape = [find_closest_number_divisible_by_m(s, 2**n_levels, "higher") for s in crop]
        im, crop_idx = crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    im = rescale_volume(im, new_min=0, new_max=1, min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [find_closest_number_divisible_by_m(s, 2**n_levels, "higher") for s in input_shape]
    if min_pad is not None:
        min_pad = reformat_to_list(min_pad, length=n_dims, dtype="int")
        min_pad = [find_closest_number_divisible_by_m(s, 2**n_levels, "higher") for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # test-time augmentation by RL flipping
    im_flipped = flip_volume(im, direction="rl", aff=np.eye(4))

    # add batch and channel axes
    im = add_axis(im, axis=[0, -1])
    im_flipped = add_axis(im_flipped, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx, im_flipped


# ================================================================================================
#                                     Callable prediction API
# ================================================================================================


def load_label_tables() -> dict[str, np.ndarray]:
    """Load the SynthSeg label tables shipped in ``labels/`` (verbatim copies of the upstream files).

    Returns the raw arrays (``labels``, ``names``, ``topology``, ``groups``) plus the derived
    ``unique_labels``/``unique_indices`` (``np.unique(labels, return_index=True)``): the network
    output channel ``k`` corresponds to ``unique_labels[k]`` and to ``names[unique_indices[k]]``.
    """
    labels = np.load(LABELS_DIR / "synthseg_segmentation_labels.npy", allow_pickle=False)
    names = np.load(LABELS_DIR / "synthseg_segmentation_names.npy", allow_pickle=False)
    topology = np.load(LABELS_DIR / "synthseg_topological_classes.npy", allow_pickle=False)
    groups = np.load(LABELS_DIR / "synthseg_group_labels.npy", allow_pickle=False)
    unique_labels, indices = np.unique(np.int32(labels), return_index=True)
    return {
        "labels": labels,
        "names": names,
        "topology": topology,
        "groups": groups,
        "unique_labels": unique_labels,
        "unique_indices": indices,
    }


def seg_output_path(image_path, in_root, out_root) -> str:
    """Output segmentation path used by the legacy pipeline: ``<name>_synthseg_robust.nii.gz``."""
    image_path, in_root, out_root = (
        os.path.abspath(str(p)) for p in (image_path, in_root, out_root)
    )
    if os.path.isfile(in_root):
        rel = os.path.basename(in_root)
    else:
        rel = os.path.relpath(image_path, in_root)
    head, tail = os.path.split(os.path.join(out_root, rel))
    return os.path.join(head, tail.replace(".nii", "_synthseg_robust.nii"))


class SynthSegPredictor:
    """SynthSeg robust-mode predictor (no test-time flipping, whole volume, CPU or GPU).

    Reproduces the paper pipeline: percentile rescaling to [0, 1], padding to at least 160^3,
    forward pass of the torch port, largest-connected-component clean-up, argmax, volumes as the
    sum of posteriors times voxel volume (mm^3, rounded to 3 decimals).
    """

    def __init__(self, weights=None, run_dir=None, device=None, net=None):
        self.net = net if net is not None else load_synthseg(weights, run_dir, device)
        self.device = next(self.net.parameters()).device
        self.tables = load_label_tables()

    def segment_image(self, path_image, path_seg=None, return_posteriors=False):
        """Segment one image. Returns ``(seg, volumes)``; ``volumes`` has one entry per label
        (background excluded) in ``tables["unique_labels"][1:]`` order."""
        path_image = str(path_image)
        image, aff, h, im_res, shape, pad_idx, crop_idx, _ = preprocess(
            path_image, crop=None, min_pad=160, path_resample=None
        )
        with torch.no_grad():
            pred = self.net(torch.from_numpy(image).to(self.device).permute(0, 4, 1, 2, 3).float())
        pred = pred.permute(0, 2, 3, 4, 1).cpu().numpy()
        # NOTE(paper): robust mode => no flipped second pass, no topology-class clean-up.
        seg, posteriors = postprocess(
            pred, shape, pad_idx, crop_idx, self.tables["unique_labels"], None, aff, None, None
        )
        if path_seg is not None:
            save_volume(seg, aff, h, str(path_seg), dtype="int")
        volumes = np.sum(posteriors[..., 1:], axis=tuple(range(len(posteriors.shape) - 1)))
        volumes = np.around(volumes * np.prod(im_res), 3)
        return (seg, volumes, posteriors) if return_posteriors else (seg, volumes)

    def segment_folder(self, image_dir, out_dir, vol_csv=None) -> list[str]:
        """Segment every ``*final.nii.gz`` below ``image_dir`` (sorted; or a single such file).

        Writes ``<name>_synthseg_robust.nii.gz`` into ``out_dir`` and, if ``vol_csv`` is given,
        a CSV with two header rows (label names, label ids) and one row per image, whose first
        column (empty header) holds the image name. Returns the segmentation paths.
        """
        image_dir = str(image_dir)
        paths = list_images_in_folder(image_dir)
        mkdir(os.path.abspath(str(out_dir)))
        out_paths = [seg_output_path(p, image_dir, out_dir) for p in paths]
        unique_labels, idx = self.tables["unique_labels"], self.tables["unique_indices"]
        if vol_csv is not None:
            mkdir(os.path.dirname(os.path.abspath(str(vol_csv))))
            names = self.tables["names"][idx]
            with open(vol_csv, "w") as f:
                csv.writer(f).writerows(
                    [[""] + names[1:].tolist(), [""] + [str(lab) for lab in unique_labels[1:]]]
                )
        for path_image, path_seg in zip(paths, out_paths, strict=True):
            mkdir(os.path.dirname(path_seg))
            _, volumes = self.segment_image(path_image, path_seg)
            if vol_csv is not None:
                row = [os.path.basename(path_image).replace(".nii.gz", "")] + [
                    str(v) for v in volumes
                ]
                with open(vol_csv, "a") as f:
                    csv.writer(f).writerow(row)
        return out_paths
