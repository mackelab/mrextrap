"""Tests for beta fitting, Global prior parameters, obs_var and normalisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from einops import rearrange
from torch import nn

from mrextrap import progression as pg
from mrextrap.data.datasets import LatentBetaDataset
from mrextrap.data.synthetic import make_synthetic_latent_dict
from mrextrap.io import save_latent_dict


# ---- verbatim copies of the original logic (only ``.cuda()`` removed / tqdm dropped) ----------
def _orig_prior(betas):
    betas_tensor = torch.cat(betas)
    return betas_tensor.mean(dim=0), betas_tensor.std(dim=0)


def _orig_compute_residuals(betas, zs, ages_abs):
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
    return torch.cat(residuals, dim=0)


class _OrigLinReg:
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
    ):
        self.in_features, self.out_features = in_features, out_features
        self.lr, self.n_iter, self.tol = lr, n_iter, tol
        self.model = nn.Linear(in_features, out_features, bias=fit_intercept)
        if use_l1:
            self.criterion = nn.L1Loss()
            tol = tol * 10
        else:
            self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )

    def fit(self, X, y):
        X = X.numpy()
        y = y.numpy()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        converged_in = self.n_iter
        previous_loss = float("inf")
        for _ in range(self.n_iter):
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), y)
            loss.backward()
            self.optimizer.step()
            current_loss = loss.item()
            if abs(previous_loss - current_loss) < self.tol:
                converged_in = _
                break
            previous_loss = current_loss
        return converged_in


def _orig_betas_per_time(z, age, use_l1=False):
    z_shape = z.shape
    seqlen = z_shape[0]
    betas = torch.empty(z_shape)
    for ref_i in np.arange(0, seqlen, 1):
        z_res = z - z[ref_i : ref_i + 1]
        age_res = age - age[ref_i : ref_i + 1]
        z_res_flat = z_res.reshape(seqlen, -1)
        m = _OrigLinReg(1, z_res_flat.shape[-1], use_l1=use_l1)
        m.fit(age_res, z_res_flat)
        beta = m.model.weight
        betas[ref_i] = rearrange(
            beta, "(c d h w) 1 -> 1 c d h w", c=z_shape[1], d=z_shape[2], h=z_shape[3], w=z_shape[4]
        )
    return betas


# ---- fixtures -----------------------------------------------------------------------------------
@pytest.fixture
def exact_betas_dict():
    """Noise-free linear latents with the true betas stored (so fitting is not needed)."""
    return make_synthetic_latent_dict(
        5, seq_len_range=(4, 7), latent_shape=(2, 3, 3, 3), seed=3, include_betas=True, noise=0.02
    )


# ---- beta fitting -------------------------------------------------------------------------------
def test_fit_recovers_true_betas_l1():
    d = make_synthetic_latent_dict(
        1, seq_len_range=(5, 5), latent_shape=(1, 2, 2, 2), seed=0, noise=0.0
    )
    z, age, truth = d["zs"][0], d["ages"][0].permute(1, 0), d["true_betas"][0]
    torch.manual_seed(42)
    betas = pg.fit_betas_per_reference(z, age, use_l1=True)
    assert betas.shape == z.shape  # one beta per reference point
    assert torch.allclose(betas, truth[None].expand_as(betas), atol=1e-2)


def test_compute_betas_schema_and_ages_abs(exact_betas_dict):
    d = {k: v for k, v in exact_betas_dict.items() if k not in ("betas", "true_betas")}
    d = {
        **d,
        "zs": [z[:, :1, :2, :2, :2] for z in d["zs"][:2]],
        "ages": d["ages"][:2],
        "ages_abs": d["ages_abs"][:2],
        "subject_ids": d["subject_ids"][:2],
    }
    out = pg.compute_betas(d, use_l1=True, seed=42)
    assert set(out) == {"zs", "ages", "ages_abs", "subject_ids", "dataset_type", "betas"}
    for z, b in zip(out["zs"], out["betas"]):
        assert b.shape == z.shape and not b.requires_grad
    # the original patch semantics: ages_abs is the one from zs_ages, unchanged
    assert all(torch.equal(a, b) for a, b in zip(out["ages_abs"], d["ages_abs"]))


def test_compute_betas_matches_original_logic():
    d = make_synthetic_latent_dict(2, seq_len_range=(3, 4), latent_shape=(1, 2, 2, 2), seed=1)
    out = pg.compute_betas(d, use_l1=True, seed=42)
    torch.manual_seed(42)
    expected = [
        _orig_betas_per_time(z, a.permute(1, 0), use_l1=True) for z, a in zip(d["zs"], d["ages"])
    ]
    for got, exp in zip(out["betas"], expected):
        assert torch.equal(got, exp)


def test_compute_betas_skips_short_subjects():
    d = make_synthetic_latent_dict(2, seq_len_range=(3, 3), latent_shape=(1, 2, 2, 2), seed=1)
    for k in ("zs", "ages", "ages_abs", "subject_ids"):
        d[k][0] = d[k][0][:1] if k != "ages" and k != "ages_abs" else d[k][0][:, :1]
    with pytest.warns(UserWarning, match="Skipping"):
        out = pg.compute_betas(d)
    assert len(out["betas"]) == len(out["zs"]) == len(out["subject_ids"]) == 1


# ---- global prior: bit-identical to the original functions ---------------------------------------
def test_global_prior_bit_identical():
    d = make_synthetic_latent_dict(
        10, seq_len_range=(2, 7), latent_shape=(4, 4, 4, 4), seed=5, include_betas=True, noise=0.1
    )
    assert any(len(z) < 4 for z in d["zs"]), "need subjects that must be ignored by obs_var"
    got = pg.compute_global_prior(d)
    mean, std = _orig_prior(d["betas"])
    obs_var = _orig_compute_residuals(d["betas"], d["zs"], d["ages_abs"]).var(dim=0)
    assert set(got) == {"global_beta_mean", "global_beta_std", "global_obs_var"}
    assert torch.equal(got["global_beta_mean"], mean)
    assert torch.equal(got["global_beta_std"], std)
    assert torch.equal(got["global_obs_var"], obs_var)
    assert got["global_obs_var"].shape == (4, 4, 4, 4)


def test_global_prior_on_fixture_latents(synthetic_latent_dict):
    d = dict(synthetic_latent_dict["train"])
    d["betas"] = [torch.randn_like(z) for z in d["zs"]]
    got = pg.compute_global_prior(d)
    assert torch.equal(got["global_beta_std"], torch.cat(d["betas"]).std(dim=0))
    assert torch.equal(
        got["global_obs_var"],
        _orig_compute_residuals(d["betas"], d["zs"], d["ages_abs"]).var(dim=0),
    )


def test_obs_var_ignores_short_subjects_and_matches_hand_reference():
    torch.manual_seed(1)
    shape = (1, 1, 1, 1)
    # subject A: 4 scans, subject B: 3 scans (ignored), subject C: 4 scans
    ns = [4, 3, 4]
    zs = [torch.randn(n, *shape) for n in ns]
    betas = [torch.randn(n, *shape) for n in ns]
    ages_abs = [(60 + torch.arange(n, dtype=torch.float32) * 1.5)[None] for n in ns]

    def hand(si):
        # mean over reference points ni and over first axis of the (N, N) broadcast grid:
        # residual[j, k] = (z_k - z_ni) - (a_j - a_ni) * b_ni   (j: age index, k: latent index)
        n = ns[si]
        z, b, a = zs[si].flatten(), betas[si].flatten(), ages_abs[si][0]
        acc = torch.zeros(n)  # per k, after the mean over (ni, j)
        for ni in range(n):
            grid = (z[None, :] - z[ni]) - (a[:, None] - a[ni]) * b[ni]  # [j, k]
            acc += grid.mean(dim=0) * n / (n - 1)
        return acc / n  # mean over ni (cat then mean in the original)

    expected = torch.cat([hand(0), hand(2)]).var(dim=0, unbiased=True)
    got = pg.compute_obs_var(betas, zs, ages_abs)
    assert got.shape == shape
    assert torch.allclose(got.flatten(), expected.reshape(1), rtol=1e-5, atol=1e-7)
    # dropping the short subject changes nothing
    keep = [0, 2]
    got2 = pg.compute_obs_var(
        [betas[i] for i in keep], [zs[i] for i in keep], [ages_abs[i] for i in keep]
    )
    assert torch.equal(got, got2)


# ---- normalisation ------------------------------------------------------------------------------
def test_normalization_matches_dataset(tmp_path, exact_betas_dict):
    d = {k: v for k, v in exact_betas_dict.items() if k != "true_betas"}
    path = tmp_path / "train_latent_betas_l1.pt"
    save_latent_dict(d, path)
    ds = LatentBetaDataset(path)
    ref = ds.normalization_stats()
    got = pg.compute_normalization(d)
    assert (
        set(got)
        == set(ref)
        == {"latent_means", "latent_stds", "beta_means", "beta_stds", "age_mu", "age_sigma"}
    )
    for k in ref:
        assert torch.equal(got[k], ref[k]), k
    assert got["latent_means"].shape == (2, 1, 1, 1)


# ---- CLI stage functions -------------------------------------------------------------------------
def test_stage_functions(tmp_path):
    from omegaconf import OmegaConf

    run = tmp_path / "run"
    lat = run / "latents"
    d = make_synthetic_latent_dict(3, seq_len_range=(4, 5), latent_shape=(1, 2, 2, 2), seed=2)
    d = {k: v for k, v in d.items() if k != "true_betas"}
    save_latent_dict(d, lat / "train_zs_ages.pt")
    cfg = OmegaConf.create(
        {
            "paths": {"latents_dir": str(lat), "priors_dir": str(run / "priors")},
            "progression": {"splits": ["train"]},
        }
    )
    (path,) = pg.run_fit_betas(cfg)
    assert path.name == "train_latent_betas_l1.pt"
    gp, norm = pg.run_global_prior(cfg)
    assert set(torch.load(gp)) == {"global_beta_mean", "global_beta_std", "global_obs_var"}
    assert "age_sigma" in torch.load(norm)
