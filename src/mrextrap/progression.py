"""Per-subject latent progression rates (betas), Global prior parameters and normalisation.

Ported from ``create_latent_betas.py`` (``TorchLinearRegression``,
``train_linear_regression_per_time``) and ``compute_global_prior_and_var.py``. Numerics are kept
verbatim (operation order, dtypes, defaults, seeds); only the hard-coded CUDA device became an
argument.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch import nn

from mrextrap.io import load_latent_dict, save_latent_dict, validate_latent_dict


class TorchLinearRegression:
    """Gradient-fitted linear regression ``y = X W^T`` (no intercept), one weight per output.

    NOTE(paper): (Q10) the fit is an Adam gradient descent (lr 3e-3, 1000 iterations, L1 loss),
    not a closed-form solution. ``tol`` stays at 1e-6 because the original assigns ``tol * 10``
    to a local variable only. ``weight_decay`` only parametrises an AdamW optimiser that
    ``fit`` replaces by Adam before use. The ``nn.Linear`` initialisation is random and comes
    from the global torch RNG (seeded once in ``compute_betas``); ``fit_intercept=None`` means
    ``bias=None`` i.e. no intercept.
    """

    def __init__(
        self,
        in_features,
        out_features,
        lr=0.003,
        weight_decay=0.1,
        n_iter=1000,
        tol=1e-6,
        use_l1=False,
        fit_intercept=None,
        device="cpu",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.device = device
        self.model = nn.Linear(in_features, out_features, bias=fit_intercept).to(device)
        if use_l1:
            self.criterion = nn.L1Loss()
            tol = tol * 10  # NOTE(paper): (Q10) local only, self.tol is unchanged
        else:
            self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )

    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            X = X.numpy()
            y = y.numpy()
        if X.shape[1] != self.in_features:
            raise ValueError(f"Expected X to have {self.in_features} features, got {X.shape[1]}")
        if y.shape[1] != self.out_features:
            raise ValueError(f"Expected y to have {self.out_features} features, got {y.shape[1]}")

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        # re-initialize the optimizer to reset any previous state
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        converged_in = self.n_iter
        previous_loss = float("inf")
        for i in range(self.n_iter):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            if abs(previous_loss - current_loss) < self.tol:
                converged_in = i
                break
            previous_loss = current_loss
        return converged_in

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.numpy()
        if X.shape[1] != self.in_features:
            raise ValueError(f"Expected X to have {self.in_features} features, got {X.shape[1]}")
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X).cpu()
        return predictions.numpy()

    def get_regression_params(self):
        return {"beta": self.model.weight.detach().cpu()}


def fit_betas_per_reference(z, age, use_l1=True, device="cpu", pbar=None):
    """Betas of one subject for every reference scan (``train_linear_regression_per_time``).

    Args:
        z: (N, C, D, H, W) latents of one subject (tensor).
        age: (N, 1) ages (years since the subject's first scan in the paper pipeline).
    Returns:
        (N, C, D, H, W); row ``i`` is the slope of ``z - z[i]`` on ``age - age[i]`` (through the
        origin, i.e. anchored at scan ``i``).
    """
    z_shape = z.shape
    seqlen = z_shape[0]
    assert seqlen == age.shape[0], f"Sequence length mismatch, z: {seqlen}, age: {age.shape[0]}"

    betas = torch.empty(z_shape)
    for ref_i in np.arange(0, seqlen, 1):
        z_0 = z[ref_i : ref_i + 1]
        age_0 = age[ref_i : ref_i + 1]
        z_res = z - z_0
        age_res = age - age_0

        z_res_flat = z_res.reshape(seqlen, -1)

        linreg_model = TorchLinearRegression(
            in_features=1, out_features=z_res_flat.shape[-1], use_l1=use_l1, device=device
        )
        converged_in = linreg_model.fit(age_res, z_res_flat)
        if pbar is not None:
            pbar.set_description(f"converged in {converged_in} iters")
        beta = linreg_model.get_regression_params()["beta"]
        betas[ref_i] = rearrange(
            beta,
            "(c d h w) 1 -> 1 c d h w",
            c=z_shape[1],
            d=z_shape[2],
            h=z_shape[3],
            w=z_shape[4],
        )
    assert betas.shape == z_shape
    return betas


def compute_betas(latent_dict, use_l1=True, seed=42, device="cpu") -> dict:
    """Fit betas for every subject of a ``{split}_zs_ages.pt`` dict; returns the betas dict.

    The output has the legacy ``latent_betas_l1.pt`` schema (zs, ages, ages_abs, subject_ids,
    dataset_type, betas). ``ages_abs`` is taken unchanged from the input (semantics of the
    original patch that copied ``ages_abs`` from ``zs_ages`` into ``latent_betas_l1.pt``).
    Regression uses ``ages`` (years since first scan), as in the original.

    NOTE(paper): (Q10) the global torch RNG is seeded once here (seed 42) and every regression
    draws its ``nn.Linear`` initialisation from it in subject/reference order. Subjects with
    fewer than 2 scans are skipped (dropped from every key so the lists stay aligned; the
    original silently misaligned them, which cannot occur after ``encode_split``).
    """
    validate_latent_dict(latent_dict)
    torch.manual_seed(seed)
    keep, betas = [], []
    for sub, z in enumerate(latent_dict["zs"]):
        if z.shape[0] < 2:
            warnings.warn(
                f"Skipping subject {latent_dict['subject_ids'][sub][0]} with seq len {z.shape[0]}",
                stacklevel=2,
            )
            continue
        age = latent_dict["ages"][sub].permute(1, 0)
        betas.append(fit_betas_per_reference(z, age, use_l1=use_l1, device=device))
        keep.append(sub)
    out = {k: [latent_dict[k][i] for i in keep] for k in ("zs", "ages", "ages_abs", "subject_ids")}
    out["dataset_type"] = latent_dict.get("dataset_type")
    out["betas"] = [b.data.cpu() for b in betas]
    return out


def compute_residuals(betas, zs, ages_abs):
    """Per-subject residual of the linear model, averaged over reference scans (Q2, verbatim).

    NOTE(paper): (Q2) only subjects with >= 4 scans are used and each residual is scaled by
    N/(N-1). The broadcasting of ``zs_si - intercept`` (1, N, ...) against
    ``age_diff * beta`` (N, 1, ...) yields an (N, N, ...) tensor before the mean over the
    reference points; this is kept exactly as in the original.
    """
    residuals = []
    for si in range(len(betas)):
        if len(betas[si]) < 4:
            continue
        betas_si = betas[si]
        zs_si = zs[si]
        ages_abs_si = ages_abs[si][0]
        residuals_si = []
        for ni in range(len(betas_si)):
            betas_si_ni = betas_si[ni : ni + 1].unsqueeze(0)
            ages_abs_si_ = (
                ages_abs_si.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            age_ni = ages_abs_si_[ni : ni + 1]
            intercept = zs_si[ni : ni + 1].unsqueeze(0)

            residual = (zs_si - intercept) - (ages_abs_si_ - age_ni) * betas_si_ni
            residuals_si.append(residual * len(betas_si) / (len(betas_si) - 1))
        residuals.append(torch.cat(residuals_si, dim=0).mean(dim=0))
    residuals = torch.cat(residuals, dim=0)
    return residuals


def compute_obs_var(betas, zs, ages_abs) -> torch.Tensor:
    """Global observation variance ``global_obs_var``, shape (C, D, H, W).

    NOTE(paper): (Q2) ``var`` (unbiased) over the subject-averaged residuals, assuming they have
    mean zero (the mean is not subtracted separately). The paper text calls this the "diagonal
    covariance of the residuals".
    """
    residuals = compute_residuals(betas, zs, ages_abs)
    return residuals.var(dim=0)


def compute_global_prior(latent_dict) -> dict:
    """Global prior parameters from a *train* betas dict (legacy ``global_params.pt`` keys).

    ``global_beta_mean`` / ``global_beta_std`` are the mean and unbiased ``torch.std`` over all
    scans of all subjects (``cat(betas)``), each (C, D, H, W); ``global_obs_var`` see
    ``compute_obs_var``.
    """
    validate_latent_dict(latent_dict, require_betas=True)
    betas = latent_dict["betas"]
    betas_tensor = torch.cat(betas)
    return {
        "global_beta_mean": betas_tensor.mean(dim=0),
        "global_beta_std": betas_tensor.std(dim=0),
        "global_obs_var": compute_obs_var(betas, latent_dict["zs"], latent_dict["ages_abs"]),
    }


def compute_normalization(latent_dict_train) -> dict:
    """Standardisation statistics from the *train* latents/betas (``priors/normalization.pt``).

    NOTE(paper): (Q13) channel-wise mean and unbiased std over all (scan, x, y, z) positions,
    sequences containing NaNs removed, same as ``LatentBetaDataset``. ``age_mu``/``age_sigma``
    are mean/std of the absolute ages (all scans).
    """
    d = latent_dict_train
    validate_latent_dict(d, require_betas=True)
    nan = [
        bool(
            torch.isnan(z).any()
            | torch.isnan(a).any()
            | torch.isnan(aa).any()
            | torch.isnan(b).any()
        )
        for z, a, aa, b in zip(d["zs"], d["ages"], d["ages_abs"], d["betas"])
    ]
    zs = torch.cat([z for z, n in zip(d["zs"], nan) if not n])
    betas = torch.cat([b for b, n in zip(d["betas"], nan) if not n])
    ages_abs = torch.cat([a[0] for a, n in zip(d["ages_abs"], nan) if not n])
    return {
        "latent_means": zs.mean(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1),
        "latent_stds": zs.std(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1),
        "beta_means": betas.mean(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1),
        "beta_stds": betas.std(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1),
        "age_mu": ages_abs.mean(),
        "age_sigma": ages_abs.std(),
    }


# ---------------------------------------------------------------- CLI stage functions
def _latents_dir(cfg) -> Path:
    return Path(cfg.paths.latents_dir)


def run_fit_betas(cfg) -> list[Path]:
    """``mrextrap fit-betas``: ``latents/{split}_zs_ages.pt`` -> ``latents/{split}_latent_betas_l1.pt``.

    Optional config keys: ``progression.splits`` (default train/val/test), ``progression.seed``
    (42), ``progression.use_l1`` (true), ``progression.device`` (cpu).
    """
    p = cfg.get("progression", {})
    device = p.get("device", "cpu")
    out = []
    for split in p.get("splits", ["train", "val", "test"]):
        d = load_latent_dict(_latents_dir(cfg) / f"{split}_zs_ages.pt")
        res = compute_betas(d, use_l1=p.get("use_l1", True), seed=p.get("seed", 42), device=device)
        path = _latents_dir(cfg) / f"{split}_latent_betas_l1.pt"
        save_latent_dict(res, path)
        out.append(path)
    return out


def run_global_prior(cfg) -> tuple[Path, Path]:
    """``mrextrap global-prior``: train betas -> ``priors/global_params.pt`` + ``normalization.pt``."""
    d = load_latent_dict(_latents_dir(cfg) / "train_latent_betas_l1.pt")
    priors = Path(cfg.paths.priors_dir)
    priors.mkdir(parents=True, exist_ok=True)
    torch.save(compute_global_prior(d), priors / "global_params.pt")
    torch.save(compute_normalization(d), priors / "normalization.pt")
    return priors / "global_params.pt", priors / "normalization.pt"
