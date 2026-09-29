# ruff: noqa: UP031, TRY002, SIM101, C409  (vendored upstream style, kept for fidelity)
"""SynthSeg post-processing and array utilities.

Provenance: torch port of the SynthSeg (robust, v2.0) segmentation network and its
pre/post-processing, as used in the MRExtrap paper. Code in this file is adapted from
snapshot synthseg_torch/postprocess.py, which in turn is derived from the official SynthSeg repository
(https://github.com/BBillot/SynthSeg, B. Billot et al., Apache License 2.0; see
``LICENSE-SynthSeg`` in this directory and THIRD_PARTY_NOTICES.md).
Modifications: split into modules, callable API instead of ``sys.argv``, device-agnostic
(no hard-coded CUDA), unused code removed. Numerics are unchanged.
"""

import os

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
)
from scipy.ndimage import label as scipy_label


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), "No such path: %s" % var
        var = np.load(var)
    return var


def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: croppping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0] : crop_idx[2], crop_idx[1] : crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[
            crop_idx[0] : crop_idx[3],
            crop_idx[1] : crop_idx[4],
            crop_idx[2] : crop_idx[5],
            ...,
        ]
    else:
        raise Exception("cannot crop volumes with more than 3 dimensions")

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def find_closest_number_divisible_by_m(n, m, answer_type="lower"):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == "lower":
            return lower
        elif answer_type == "higher":
            return higher
        elif answer_type == "closer":
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception(
                "answer_type should be lower, higher, or closer, had : %s" % answer_type
            )


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int32, np.int64, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str) or isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError(
                    f"if var is a list/tuple/numpy array, it should be of length 1 or {length}, "
                    f"had {var}"
                )
    else:
        raise TypeError(
            "var should be an int, float, tuple, list, numpy array, or path to numpy array"
        )

    # convert items type
    if dtype is not None:
        if dtype == "int":
            var = [int(v) for v in var]
        elif dtype == "float":
            var = [float(v) for v in var]
        elif dtype == "bool":
            var = [bool(v) for v in var]
        elif dtype == "str":
            var = [str(v) for v in var]
        else:
            raise ValueError(f"dtype should be 'str', 'float', 'int', or 'bool'; had {dtype}")
    return var


def build_binary_structure(connectivity, n_dims, shape=None):
    """Return a dilation/erosion element with provided connectivity"""
    if shape is None:
        shape = [connectivity * 2 + 1] * n_dims
    else:
        shape = reformat_to_list(shape, length=n_dims)
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct


def mask_volume(
    volume,
    mask=None,
    threshold=0.1,
    dilate=0,
    erode=0,
    fill_holes=False,
    masking_value=0,
    return_mask=False,
    return_copy=True,
):
    """Mask a volume, either with a given mask, or by keeping only the values above a threshold.
    :param volume: a numpy array, possibly with several channels
    :param mask: (optional) a numpy array to mask volume with.
    Mask doesn't have to be a 0/1 array, all strictly positive values of mask are considered for masking volume.
    Mask should have the same size as volume. If volume has several channels, mask can either be uni- or multi-channel.
     In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed mask.
    :param erode: (optional) number of voxels by which to erode the provided or computed mask.
    :param fill_holes: (optional) whether to fill the holes in the provided or computed mask.
    :param masking_value: (optional) masking value
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    vol_shape = list(new_volume.shape)
    n_dims, n_channels = get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = new_volume >= threshold
    else:
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], (
            f"mask should have shape {vol_shape[:n_dims]}, or {vol_shape[:n_dims] + [n_channels]}, had {list(mask.shape)}"
        )
        mask = mask > 0
    if dilate > 0:
        dilate_struct = build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == new_volume.shape:
        new_volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        new_volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return new_volume, mask_to_apply
    else:
        return new_volume


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def flip_volume(volume, axis=None, direction=None, aff=None, return_copy=True):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :return: flipped volume
    """

    new_volume = volume.copy() if return_copy else volume
    assert (axis is not None) | ((aff is not None) & (direction is not None)), (
        "please provide either axis, or an affine matrix with a direction"
    )

    # get flipping axis from aff if axis not provided
    if (axis is None) & (aff is not None):
        volume_axes = get_ras_axes(aff)
        if direction == "rl":
            axis = volume_axes[0]
        elif direction == "ap":
            axis = volume_axes[1]
        elif direction == "si":
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)

    # flip volume
    return np.flip(new_volume, axis=axis)


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return (
        components == np.argmax(np.bincount(components.flat)[1:]) + 1
        if n_components > 0
        else mask.copy()
    )


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = (
                ras_axes_flo[i],
                ras_axes_flo[swapped_axis_idx],
            )

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = -aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


def postprocess(
    post_patch,
    shape,
    pad_idx,
    crop_idx,
    segmentation_labels,
    lr_indices,
    aff,
    topology_classes,
    post_patch_flip,
):
    # get posteriors
    post_patch = np.squeeze(post_patch)
    if post_patch_flip is not None:
        post_patch_flip = np.squeeze(post_patch_flip)
        post_patch_flip = flip_volume(
            post_patch_flip, direction="rl", aff=np.eye(4), return_copy=False
        )
        post_patch_flip[..., lr_indices.flatten()] = post_patch_flip[
            ..., lr_indices[::-1].flatten()
        ]
        post_patch = 0.5 * (post_patch + post_patch_flip)

    # keep biggest connected component

    tmp_post_patch = post_patch[..., 1:]
    post_patch_mask = np.sum(tmp_post_patch, axis=-1) > 0.25
    post_patch_mask = get_largest_connected_component(post_patch_mask)
    post_patch_mask = np.stack([post_patch_mask] * tmp_post_patch.shape[-1], axis=-1)

    tmp_post_patch = mask_volume(tmp_post_patch, mask=post_patch_mask, return_copy=False)
    post_patch[..., 1:] = tmp_post_patch

    # reset posteriors to zero outside the largest connected component of each topological class
    if post_patch_flip is not None:
        post_patch_mask = post_patch > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch[..., idx] *= tmp_mask
    else:
        post_patch_mask = post_patch > 0.2
        post_patch[..., 1:] *= post_patch_mask[..., 1:]

    # renormalise posteriors and get hard segmentation
    post_patch /= np.sum(post_patch, axis=-1)[..., np.newaxis]
    seg_patch = post_patch.argmax(-1)

    # paste patches back to matrix of original image size
    seg_patch = crop_volume_with_idx(seg_patch, pad_idx, n_dims=3, return_copy=False)
    post_patch = crop_volume_with_idx(post_patch, pad_idx, n_dims=3, return_copy=False)
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype=np.int32)
        posteriors = np.zeros(shape=[*shape, segmentation_labels.shape[0]])
        posteriors[..., 0] = np.ones(shape)  # place background around patch
        seg[
            crop_idx[0] : crop_idx[3],
            crop_idx[1] : crop_idx[4],
            crop_idx[2] : crop_idx[5],
        ] = seg_patch
        posteriors[
            crop_idx[0] : crop_idx[3],
            crop_idx[1] : crop_idx[4],
            crop_idx[2] : crop_idx[5],
            :,
        ] = post_patch
    else:
        seg = seg_patch
        posteriors = post_patch
    seg = segmentation_labels[seg.astype(np.int32)].astype(np.int32)

    # align prediction back to first orientation
    seg = align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    posteriors = align_volume_to_ref(
        posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False
    )

    return seg, posteriors
