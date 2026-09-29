"""Losses used by MRExtrap: SSIM, adversarial and LPIPS (autoencoder) and Gaussian NLL (UNet prior).

Ported from the research code (``losses.py`` and ``train_unet_heteroscedastic.py``).
"""

import warnings
from enum import Enum

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all(ws == 1 for ws in win.shape[1:-1]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    r"""Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(
            f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}."
        )

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(
            f"Input images should have the same dtype, but got {X.type()} and {Y.type()}."
        )

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, _cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r"""class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super().__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        X = (X + 1.0) / 2.0
        Y = (Y + 1.0) / 2.0
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


## additional losses for training the autoencoder
## (https://github.com/AmigoLab/SynthAnatomy/blob/main/src/losses/adversarial/adversarial.py)


class AdversarialLosses(Enum):
    VANILLA = "vanilla"
    HINGE = "hinge"
    LEAST_SQUARE = "least_square"


class AdversarialLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        criterion: str = AdversarialLosses.LEAST_SQUARE.value,
        is_discriminator: bool = True,
        weight=None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ):
        super().__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.criterion = criterion
        self.is_discriminator = is_discriminator
        self.criterion_function = AdversarialLoss.get_criterion_function(self.criterion)

        self._weight = weight

    def forward(self, logits_fake: torch.Tensor, logits_real: torch.Tensor = None) -> torch.Tensor:
        logits_fake = logits_fake.float()

        loss_fake = torch.mean(self.criterion_function(logits_fake, not self.is_discriminator))

        loss = loss_fake

        if self.is_discriminator:
            logits_real = logits_real.float()
            loss_real = torch.mean(self.criterion_function(logits_real, True))

            loss = 0.5 * (loss + loss_real)

        loss = self._weight * loss

        return loss

    def get_weight(self) -> float:
        return self._weight

    def set_weight(self, weight: float) -> float:
        self._weight = weight

        return self.get_weight()

    @staticmethod
    def get_criterion_function(criterion):
        if criterion == AdversarialLosses.VANILLA.value:

            def criterion(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
                # An equivalent explicit implementation would be
                #     loss_real = torch.mean(F.relu(1.0 - logits_real))
                #     loss_fake = torch.mean(F.relu(1.0 + logits_fake))
                return F.relu(1.0 + (-1 if is_real else 1) * logits)

        elif criterion == AdversarialLosses.HINGE.value:

            def criterion(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
                # An equivalent explicit implementation would be
                #     loss_real = torch.mean(torch.nn.functional.softplus(-logits_real))
                #     loss_fake = torch.mean(torch.nn.functional.softplus(logits_fake))
                return torch.nn.functional.softplus((-1 if is_real else 1) * logits)

        elif criterion == AdversarialLosses.LEAST_SQUARE.value:

            def criterion(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
                # An equivalent explicit implementation would be
                #     loss_real = torch.mean((logits_real - 1) ** 2)
                #     loss_fake = torch.mean(logits_fake ** 2)
                return (logits - (1 if is_real else 0)) ** 2

        return criterion


## LPIPS wrapper from medfusion (https://github.com/mueller-franzes/medfusion/blob/main/medical_diffusion/loss/perceivers.py)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""

    def __init__(self, linear_calibration=False, normalize=False, slice_size=5, pretrained=True):
        super().__init__()
        # `pretrained=False` builds the VGG backbone with random weights and skips the linear
        # calibration weights, so nothing is downloaded (offline tests only; not a valid loss).
        self.loss_fn = lpips.LPIPS(
            net="vgg",
            lpips=linear_calibration,
            pretrained=pretrained,
            pnet_rand=not pretrained,
        )  # Note: only 'vgg' valid as loss
        self.normalize = normalize  # If true, normalize [0, 1] to [-1, 1]
        self.slice_size = slice_size

    def forward(self, pred, target):
        if pred.ndim == 5:  # 3D Image: Just use 2D model and compute average over slices
            # NOTE(paper): the slice axis and slices are random per call (global numpy/torch RNG);
            # needs at least `slice_size` entries along the chosen axis.
            # randomly select dimension to average over
            dim = np.random.randint(2, 5)

            # get a bunch of random slices, no way this is gonna scale for backprop otherwise
            losses = 0
            perm = torch.randperm(target.size(dim))
            idx = perm[: self.slice_size]
            # print(dim, idx)

            if dim == 2:
                pred = pred[:, :, idx]
                target = target[:, :, idx]
                losses = torch.stack(
                    [
                        self.loss_fn(pred[:, :, d], target[:, :, d], normalize=self.normalize)
                        for d in range(self.slice_size)
                    ],
                    dim=2,
                )

            elif dim == 3:
                pred = pred[:, :, :, idx]
                target = target[:, :, :, idx]
                losses = torch.stack(
                    [
                        self.loss_fn(pred[:, :, :, d], target[:, :, :, d], normalize=self.normalize)
                        for d in range(self.slice_size)
                    ],
                    dim=3,
                )

            elif dim == 4:
                pred = pred[:, :, :, :, idx]
                target = target[:, :, :, :, idx]
                losses = torch.stack(
                    [
                        self.loss_fn(
                            pred[:, :, :, :, d], target[:, :, :, :, d], normalize=self.normalize
                        )
                        for d in range(self.slice_size)
                    ],
                    dim=4,
                )

            return torch.mean(losses, dim=dim, keepdim=True)
        else:
            return self.loss_fn(pred, target, normalize=self.normalize)


class GaussLoss(nn.Module):
    """Heteroscedastic Gaussian negative log-likelihood (up to constants) on (mean, log-variance)."""

    def __init__(self, reduction="mean", sigma_eps=1e-4, resi_min=1e-4, resi_max=1e3) -> None:
        super().__init__()
        self.reduction = reduction
        self.sigma_eps = sigma_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def forward(self, mean: torch.Tensor, logsigma2: torch.Tensor, target: torch.Tensor):
        sigma2 = torch.exp(logsigma2)
        one_over_sigma2 = 1 / (self.sigma_eps + sigma2)

        # l2 residual
        resi = torch.abs(mean - target) ** 2

        # NOTE(paper): the weighted residual is clamped to [resi_min, resi_max] before summing with
        # the log-variance term, so gradients vanish outside that range.
        resi = (resi * one_over_sigma2).clamp(min=self.resi_min, max=self.resi_max)

        # NOTE(paper): NaNs are reported and None is returned (the training loop must handle it).
        if torch.isnan(resi).any():
            print("resi has nans!!")
            return None

        loss = resi + logsigma2  # adding the log(sigma^2) term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            print("Reduction not supported")
            return None


class TempCombGaussLoss(nn.Module):
    """``T1 * L1(mean, target) + T2 * GaussLoss(mean, logsigma2, target)``."""

    def __init__(self, reduction="mean", sigma_eps=1e-4, resi_min=1e-4, resi_max=1e3) -> None:
        super().__init__()
        self.reduction = reduction
        self.sigma_eps = sigma_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

        self.L_Gauss = GaussLoss(
            reduction=self.reduction,
            sigma_eps=self.sigma_eps,
            resi_min=self.resi_min,
            resi_max=self.resi_max,
        )
        self.L_l1 = nn.L1Loss(reduction=self.reduction)

    def forward(
        self,
        mean: torch.Tensor,
        logsigma2: torch.Tensor,
        target: torch.Tensor,
        T1: float,
        T2: float,
    ):
        l1 = self.L_l1(mean, target)
        l2 = self.L_Gauss(mean, logsigma2, target)
        return T1 * l1 + T2 * l2


def make_prior_loss() -> TempCombGaussLoss:
    """Loss instance used to train the paper UNet prior (``L_TCG`` in the research code)."""
    return TempCombGaussLoss(reduction="mean", sigma_eps=1e-6, resi_min=1e-6, resi_max=1e4)
