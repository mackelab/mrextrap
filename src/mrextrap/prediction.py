"""Prediction of future latents from a single scan (Global, UNet and diffusion priors) and the
closed-form Bayesian posterior update for additional scans (paper Eq. ``unet_posterior``).

All functions are ports of the original evaluation code; numerics, operation order and seeds are
preserved. Tensors follow the latent layout ``[N, C, D, H, W]``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from einops import rearrange, repeat
from tqdm import tqdm

__all__ = [
    "AgeNormalization",
    "DiffusionPrior",
    "GlobalPrior",
    "UNetPrior",
    "delta_t_for_targets",
    "extrapolate",
    "update_beta_posterior_closed_form",
]


class AgeNormalization:
    """``(age - mu) / sigma`` (the legacy ``Normalization`` module without parameters)."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma


class _Normalizer:
    """Standardisation with the train-latent statistics stored in ``priors/normalization.pt``
    (keys ``latent_means, latent_stds, beta_means, beta_stds, age_mu, age_sigma``)."""

    def __init__(self, norm: dict):
        self.latent_means = norm["latent_means"]
        self.latent_stds = norm["latent_stds"]
        self.beta_means = norm["beta_means"]
        self.beta_stds = norm["beta_stds"]
        self.age = AgeNormalization(norm["age_mu"], norm["age_sigma"])

    def standardize_latents(self, z):
        return (z - self.latent_means) / self.latent_stds

    def destandardize_betas(self, beta):
        return beta * self.beta_stds + self.beta_means


def _age_column(age_abs: torch.Tensor) -> torch.Tensor:
    """Absolute age of the anchor scan as a ``[B, 1]`` float tensor."""
    return torch.as_tensor(age_abs).reshape(-1, 1).float()


def extrapolate(z_anchor: torch.Tensor, beta: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    """Linear latent trajectory ``z_anchor + beta * delta_t`` -> ``[T, C, D, H, W]``.

    ``delta_t`` has shape ``[T]``; ``beta`` is ``[C,D,H,W]`` or ``[1,C,D,H,W]``.
    """
    return beta * delta_t.reshape(-1, 1, 1, 1, 1).float() + z_anchor


def delta_t_for_targets(
    target_ages_since_first: torch.Tensor,
    baseline_age_since_first: torch.Tensor | float = 0.0,
    delta_t_from_first_scan: bool = True,
) -> torch.Tensor:
    """Time offsets used to extrapolate from the baseline scan.

    NOTE(paper): Q1. The single-volume Global and UNet evaluations extrapolate with
    ``delta_t = age since the subject's first scan`` of the target (the default), which equals
    ``target - baseline`` only if the baseline is the subject's first scan. The diffusion path uses
    ``ages[end] - ages[start]``. Set ``delta_t_from_first_scan=False`` for the latter convention.
    """
    if delta_t_from_first_scan:
        return target_ages_since_first
    return target_ages_since_first - baseline_age_since_first


class GlobalPrior:
    """Global prior ``N(mean, std^2)`` per latent location and the observation-noise variance."""

    def __init__(self, beta_mean, beta_std, obs_var=None):
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.obs_var = obs_var

    @classmethod
    def from_file(cls, path: str | Path) -> GlobalPrior:
        params = torch.load(path, map_location="cpu")
        return cls(
            params["global_beta_mean"], params["global_beta_std"], params.get("global_obs_var")
        )

    def mean_var(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE(paper): the prior variance is ``global_beta_std ** 2`` (unbiased std over all
        # subjects and reference points, see Q2); no standardisation is involved.
        return self.beta_mean, (self.beta_std**2).to(device)


class UNetPrior:
    """Heteroscedastic UNet prior: predicts the mean and log-variance of beta from one scan."""

    def __init__(self, model: torch.nn.Module, norm: dict):
        self.model = model
        self.norm = _Normalizer(norm)

    @torch.no_grad()
    def predict(self, z, age_abs, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """``z`` is ``[1,C,D,H,W]`` (raw latent), ``age_abs`` the absolute age of that scan.

        Returns ``(beta_mean, beta_logvar)``; the mean is destandardised (CPU tensor). The log-variance
        is returned in standardised units as produced by the network.
        """
        z_norm = self.norm.standardize_latents(z).to(device)
        age_norm = self.norm.age(_age_column(age_abs)).to(device)
        out = self.model(z_norm, y=age_norm)
        beta = self.norm.destandardize_betas(out.sample_mean.cpu())
        return beta, out.sample_logvar.cpu()


class DiffusionPrior:
    """Latent-diffusion prior over beta conditioned on (z, age); sampled with a DDPM scheduler."""

    def __init__(
        self, denoiser: torch.nn.Module, norm: dict, NT: int = 500, schedule: str = "linear"
    ):
        self.denoiser = denoiser
        self.norm = _Normalizer(norm)
        self.NT = NT
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        z,
        age_abs,
        num_samples: int = 5,
        seed: int = 0,
        device: str = "cpu",
        autocast_bf16: bool | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Draw ``num_samples`` betas for one anchor scan; returns ``[1, K, C, D, H, W]``.

        NOTE(paper): Q14. The evaluation used K=5 and the *same* generator seed 0 for every subject;
        the generator lives on ``device``. ``autocast_bf16=None`` enables bf16 autocast on CUDA only
        (Q11: the paper denoiser forward ran under bf16 autocast).
        """
        if autocast_bf16 is None:
            autocast_bf16 = torch.device(device).type == "cuda"
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.NT,
            clip_sample=False,
            beta_schedule=self.schedule,
        )
        z_norm = self.norm.standardize_latents(z).to(device)
        age_abs_norm = self.norm.age(_age_column(age_abs)).to(device)

        cond = repeat(z_norm, "b c d h w -> (b s) c d h w", s=num_samples)
        assert len(cond.shape) == 5, f"z should have shape (B, C, D, H, W), got {cond.shape}"
        age_abs_norm = repeat(age_abs_norm, "b c -> (b s) c", s=num_samples)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        beta_t = torch.empty_like(cond).normal_(generator=generator)

        noise_scheduler.set_timesteps(self.NT)
        timesteps = (
            tqdm(noise_scheduler.timesteps, desc="Sampling Diffusion")
            if progress
            else (noise_scheduler.timesteps)
        )
        ctx = (
            torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
            if autocast_bf16
            else contextlib.nullcontext()
        )
        for t in timesteps:
            unet_input = torch.cat([beta_t, cond], dim=1)
            with ctx:
                model_output = self.denoiser(unet_input, t, y=age_abs_norm).sample
            beta_t = noise_scheduler.step(model_output, t, beta_t, generator=generator).prev_sample

        beta_norm = rearrange(beta_t, "(b s) c d h w -> b s c d h w", s=num_samples)
        return self.norm.destandardize_betas(beta_norm.cpu())


def update_beta_posterior_closed_form(
    beta_prior_mean, beta_prior_var, z0, deltaages, z_obs, obs_var=0.1
):
    """Closed-form update for 1D Bayesian linear regression at each latent location.

    Per location ``(c,d,h,w)``: ``dz = beta * dt + eps`` with ``eps ~ N(0, obs_var)`` and prior
    ``beta ~ N(mu0, var0)``. Posterior ``var_n = 1 / (1/var0 + sum(dt^2)/obs_var)`` and
    ``mu_n = var_n * (mu0/var0 + sum(dt * dz)/obs_var)`` (paper Eq. ``unet_posterior``).

    Args:
        beta_prior_mean: prior mean ``[C,D,H,W]``.
        beta_prior_var: prior variance ``[C,D,H,W]``.
        z0: baseline latent ``[1,C,D,H,W]``.
        deltaages: time differences of the observed scans to the baseline ``[N]``.
        z_obs: observed latents ``[N,C,D,H,W]``.
        obs_var: observation noise variance (``global_obs_var``, estimated from training residuals).

    Returns:
        ``(posterior_mean, posterior_var)``, each ``[C,D,H,W]`` (broadcast with the prior).
    """
    delta_z = z_obs - z0
    dt = deltaages.reshape(-1, 1, 1, 1, 1)
    sum_dt2 = (dt**2).sum(dim=0)
    sum_dtz = (dt * delta_z).sum(dim=0)
    posterior_var = 1.0 / (1.0 / beta_prior_var + sum_dt2 / obs_var)
    posterior_mean = posterior_var * (beta_prior_mean / beta_prior_var + sum_dtz / obs_var)
    return posterior_mean, posterior_var
