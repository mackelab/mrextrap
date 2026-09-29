"""Tests for mrextrap.prediction, including equivalence against verbatim copies of the legacy code."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler
from einops import rearrange, repeat
from torch import nn

from mrextrap.models.unet3d import Normalization
from mrextrap.prediction import (
    DiffusionPrior,
    GlobalPrior,
    UNetPrior,
    delta_t_for_targets,
    extrapolate,
    update_beta_posterior_closed_form,
)

C, S = 4, 4  # channels, spatial size


# ---- verbatim legacy copies (eval_unet_additional.py::predict_beta, notebook cell 8,
# ---- eval_global_prior_varcond.py::update_beta_posterior_closed_form) -----------------------


def legacy_predict_beta(
    latent_dict,
    model,
    train_dataloader,
    age_normalization,
    subject_index=0,
    seq_position=0,
    device="cpu",
):
    z = latent_dict["zs"][subject_index]
    age_abs = latent_dict["ages_abs"][subject_index]
    if age_abs.dim() == 1:
        age_abs = age_abs.unsqueeze(1).float()
    z_norm = train_dataloader.dataset.standardize_latents(z)
    age_abs_norm = age_normalization(age_abs)
    z_norm = z_norm[seq_position : seq_position + 1].to(device)
    age_abs_norm = age_abs_norm[seq_position : seq_position + 1].to(device)
    with torch.no_grad():
        out = model(z_norm, y=age_abs_norm)
        beta_norm = out.sample_mean
    return train_dataloader.dataset.destandardize_betas(beta_norm.cpu())


def legacy_sample_beta_from_diffusion(
    latent_dict,
    denoiser,
    NT,
    train_dataloader,
    age_normalization,
    subject_index=0,
    seq_position=0,
    num_samples=1,
    device="cpu",
    seed=0,
):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NT, clip_sample=False, beta_schedule="linear"
    )
    z = latent_dict["zs"][subject_index]
    age_abs = latent_dict["ages_abs"][subject_index].permute(1, 0).float()
    z_norm = train_dataloader.dataset.standardize_latents(z)
    age_abs_norm = age_normalization(age_abs)
    z_norm = z_norm[seq_position : seq_position + 1].to(device)
    age_abs_norm = age_abs_norm[seq_position : seq_position + 1].to(device)
    cond = repeat(z_norm, "b c d h w -> (b s) c d h w", s=num_samples)
    age_abs_norm = repeat(age_abs_norm, "b c -> (b s) c", s=num_samples)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    beta_t = torch.empty_like(cond).normal_(generator=generator)
    noise_scheduler.set_timesteps(NT)
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            unet_input = torch.cat([beta_t, cond], dim=1)
            model_output = denoiser(unet_input, t, y=age_abs_norm).sample
            beta_t = noise_scheduler.step(model_output, t, beta_t, generator=generator).prev_sample
    beta_norm = rearrange(beta_t, "(b s) c d h w -> b s c d h w", s=num_samples)
    return train_dataloader.dataset.destandardize_betas(beta_norm.cpu())


def legacy_update(beta_prior_mean, beta_prior_var, z0, deltaages, z_preds, obs_var=0.1):
    delta_z = z_preds - z0
    dt = deltaages.reshape(-1, 1, 1, 1, 1)
    sum_dt2 = (dt**2).sum(dim=0)
    sum_dtz = (dt * delta_z).sum(dim=0)
    posterior_var = 1.0 / (1.0 / beta_prior_var + sum_dt2 / obs_var)
    posterior_mean = posterior_var * (beta_prior_mean / beta_prior_var + sum_dtz / obs_var)
    return posterior_mean, posterior_var


# ---- fixtures -------------------------------------------------------------------------------


class TinyMeanVar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(C, 2 * C, 3, padding=1)
        self.label = nn.Linear(1, C)

    def forward(self, x, y=None):
        h = self.conv(x) + self.label(y).sum() * 0.1
        return SimpleNamespace(sample_mean=h[:, :C], sample_logvar=h[:, C:])


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2 * C, C, 3, padding=1)
        self.label = nn.Linear(1, C)

    def forward(self, x, t, y=None):
        out = self.conv(x) * 0.3 + self.label(y)[:, :, None, None, None] + float(t) * 1e-3
        return SimpleNamespace(sample=out)


@pytest.fixture
def norm():
    g = torch.Generator().manual_seed(1)
    return {
        "latent_means": torch.randn(C, 1, 1, 1, generator=g),
        "latent_stds": torch.rand(C, 1, 1, 1, generator=g) + 0.5,
        "beta_means": torch.randn(C, 1, 1, 1, generator=g) * 0.1,
        "beta_stds": torch.rand(C, 1, 1, 1, generator=g) * 0.1 + 0.05,
        "age_mu": torch.tensor(72.0),
        "age_sigma": torch.tensor(7.0),
    }


@pytest.fixture
def legacy_env(norm):
    dataloader = SimpleNamespace(
        dataset=SimpleNamespace(
            standardize_latents=lambda z: (z - norm["latent_means"]) / norm["latent_stds"],
            destandardize_betas=lambda b: b * norm["beta_stds"] + norm["beta_means"],
        )
    )
    age_norm = Normalization(norm["age_mu"], norm["age_sigma"])
    g = torch.Generator().manual_seed(2)
    latent_dict = {
        "zs": [torch.randn(5, C, S, S, S, generator=g)],
        "ages_abs": [torch.tensor([[70.0, 71.2, 72.5, 74.0, 75.1]])],
    }
    return dataloader, age_norm, latent_dict


# ---- equivalence ----------------------------------------------------------------------------


def test_unet_prior_matches_legacy_predict_beta(norm, legacy_env):
    dataloader, age_norm, latent_dict = legacy_env
    torch.manual_seed(0)
    model = TinyMeanVar().eval()
    for pos in (0, 2):
        # the legacy call site passes a single-scan dict: z [N,...], ages_abs [N,1]
        ld = {"zs": [latent_dict["zs"][0]], "ages_abs": [latent_dict["ages_abs"][0].T]}
        ref = legacy_predict_beta(ld, model, dataloader, age_norm, seq_position=pos)
        mean, logvar = UNetPrior(model, norm).predict(
            latent_dict["zs"][0][pos : pos + 1], latent_dict["ages_abs"][0][0, pos]
        )
        assert torch.equal(mean, ref)
        assert logvar.shape == mean.shape


def test_unet_prior_identity_model_destandardises(norm):
    class Identity(nn.Module):
        def forward(self, x, y=None):
            return SimpleNamespace(sample_mean=x, sample_logvar=torch.zeros_like(x))

    z = torch.randn(1, C, S, S, S)
    mean, _ = UNetPrior(Identity(), norm).predict(z, torch.tensor(72.0))
    expected = ((z - norm["latent_means"]) / norm["latent_stds"]) * norm["beta_stds"] + norm[
        "beta_means"
    ]
    assert torch.allclose(mean, expected, atol=1e-6)


@pytest.mark.parametrize("K", [1, 3])
def test_diffusion_prior_matches_legacy_sampler(norm, legacy_env, K):
    dataloader, age_norm, latent_dict = legacy_env
    torch.manual_seed(0)
    denoiser = TinyDenoiser().eval()
    NT = 12
    pos = 1
    ref = legacy_sample_beta_from_diffusion(
        latent_dict, denoiser, NT, dataloader, age_norm, seq_position=pos, num_samples=K, seed=0
    )
    prior = DiffusionPrior(denoiser, norm, NT=NT)
    out = prior.sample(
        latent_dict["zs"][0][pos : pos + 1],
        latent_dict["ages_abs"][0][0, pos],
        num_samples=K,
        seed=0,
        device="cpu",
        autocast_bf16=False,
    )
    assert out.shape == (1, K, C, S, S, S)
    assert torch.equal(out, ref)
    again = prior.sample(
        latent_dict["zs"][0][pos : pos + 1],
        latent_dict["ages_abs"][0][0, pos],
        num_samples=K,
        seed=0,
        autocast_bf16=False,
    )
    assert torch.equal(out, again)
    other = prior.sample(
        latent_dict["zs"][0][pos : pos + 1],
        latent_dict["ages_abs"][0][0, pos],
        num_samples=K,
        seed=1,
        autocast_bf16=False,
    )
    assert not torch.equal(out, other)


def test_posterior_matches_legacy_bitwise():
    g = torch.Generator().manual_seed(3)
    mean = torch.randn(C, S, S, S, generator=g)
    var = torch.rand(C, S, S, S, generator=g) + 0.1
    z0 = torch.randn(1, C, S, S, S, generator=g)
    dt = torch.tensor([1.1, 2.0, 3.05])
    z_obs = torch.randn(3, C, S, S, S, generator=g)
    obs_var = torch.rand(C, S, S, S, generator=g) + 0.05
    for ov in (0.1, obs_var):
        a = update_beta_posterior_closed_form(mean, var, z0, dt, z_obs, ov)
        b = legacy_update(mean, var, z0, dt, z_obs, ov)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---- posterior properties -------------------------------------------------------------------


def _one_d(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(1, 1, 1, 1, 1, generator=g, dtype=torch.float64)
    dt = torch.rand(n, generator=g, dtype=torch.float64) * 3 + 0.5
    z_obs = (
        z0
        + 0.7 * dt.reshape(-1, 1, 1, 1, 1)
        + 0.05 * torch.randn(n, 1, 1, 1, 1, generator=g, dtype=torch.float64)
    )
    return z0, dt, z_obs


def test_posterior_equals_analytic_bayesian_regression():
    z0, dt, z_obs = _one_d()
    m0, v0, s2 = 0.2, 0.5, 0.09
    y = (z_obs - z0).reshape(-1)
    # precision-form Bayesian linear regression without intercept
    precision = 1 / v0 + (dt @ dt) / s2
    mean = (m0 / v0 + (dt @ y) / s2) / precision
    pm, pv = update_beta_posterior_closed_form(
        torch.tensor(m0, dtype=torch.float64).reshape(1, 1, 1, 1),
        torch.tensor(v0, dtype=torch.float64).reshape(1, 1, 1, 1),
        z0,
        dt,
        z_obs,
        s2,
    )
    assert torch.allclose(pv.flatten(), (1 / precision).reshape(1))
    assert torch.allclose(pm.flatten(), mean.reshape(1))


def test_posterior_limits():
    z0, dt, z_obs = _one_d()
    m0 = torch.full((1, 1, 1, 1), 0.2, dtype=torch.float64)
    v0 = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
    # zero observations -> prior
    empty_obs = torch.empty(0, 1, 1, 1, 1, dtype=torch.float64)
    pm, pv = update_beta_posterior_closed_form(
        m0, v0, z0, torch.empty(0, dtype=torch.float64), empty_obs + z0, 0.1
    )
    assert torch.allclose(pm, m0) and torch.allclose(pv, v0)
    # obs_var -> infinity -> prior
    pm, pv = update_beta_posterior_closed_form(m0, v0, z0, dt, z_obs, 1e18)
    assert torch.allclose(pm, m0) and torch.allclose(pv, v0)
    # obs_var -> 0 -> least-squares slope through the baseline, variance -> 0
    pm, pv = update_beta_posterior_closed_form(m0, v0, z0, dt, z_obs, 1e-12)
    y = (z_obs - z0).reshape(-1)
    assert torch.allclose(pm.flatten(), ((dt @ y) / (dt @ dt)).reshape(1), atol=1e-6)
    assert pv.item() < 1e-9


# ---- helpers --------------------------------------------------------------------------------


def test_extrapolate_and_delta_t_convention():
    z0 = torch.randn(1, C, S, S, S)
    beta = torch.randn(1, C, S, S, S)
    dt = torch.tensor([1.0, 2.5])
    traj = extrapolate(z0, beta, dt)
    assert traj.shape == (2, C, S, S, S)
    assert torch.equal(traj[1], (beta * 2.5 + z0)[0])
    since_first = torch.tensor([3.0, 4.0])
    assert torch.equal(delta_t_for_targets(since_first, 1.5), since_first)
    assert torch.equal(
        delta_t_for_targets(since_first, 1.5, delta_t_from_first_scan=False),
        torch.tensor([1.5, 2.5]),
    )


def test_global_prior_from_file(tmp_path):
    mean, std, ov = torch.randn(C, S, S, S), torch.rand(C, S, S, S) + 0.1, torch.rand(C, S, S, S)
    torch.save(
        {"global_beta_mean": mean, "global_beta_std": std, "global_obs_var": ov}, tmp_path / "gp.pt"
    )
    gp = GlobalPrior.from_file(tmp_path / "gp.pt")
    m, v = gp.mean_var()
    assert torch.equal(m, mean) and torch.equal(v, std**2) and torch.equal(gp.obs_var, ov)
