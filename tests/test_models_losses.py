"""Models and losses: state-dict compatibility with the paper checkpoints, shapes and loss maths."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mrextrap.losses import (
    LPIPS,
    SSIM,
    AdversarialLoss,
    GaussLoss,
    TempCombGaussLoss,
    make_prior_loss,
)
from mrextrap.models.autoencoder import VAE, BaselineDiscriminator, Decoder, Encoder
from mrextrap.models.unet3d import UNetModel, UNetModelMeanVarSingleStep

FIXTURE = Path(__file__).parent / "fixtures" / "paper_state_dict_shapes.json"
SNAPSHOT_MODELS = Path("/Users/jkapoor/Documents/mrextrap_release_work/snapshot/mrextrap/models")

AE_CFG = {"n_channels": 32, "z_channels": 4, "ch_mult": [1, 2, 2, 4], "num_res_blocks": 2}
AE_RES = [160, 192, 160]
LATENT = [4, 20, 24, 20]


def build_vae(resolution=AE_RES, **cfg):
    cfg = {**AE_CFG, **cfg}
    return VAE(
        Encoder(in_channels=1, resolution=resolution, **cfg),
        Decoder(out_channels=1, resolution=resolution, **cfg),
    )


PRIOR_ARGS = {
    "image_size": LATENT,
    "in_channels": 4,
    "model_channels": 64,
    "out_channels": 4,
    "num_res_blocks": 2,
    "attention_resolutions": [],
    "channel_mult": [1, 2],
    "num_classes": 1,
    "emb_bias": True,
}


def build_unet_prior(**kw):
    return UNetModelMeanVarSingleStep(**{**PRIOR_ARGS, **kw})


DENOISER_ARGS = {
    "image_size": LATENT,
    "in_channels": 8,  # 2x latent channels: noisy target and baseline latent
    "out_channels": 4,
    "model_channels": 64,
    "num_res_blocks": 2,
    "attention_resolutions": [],
    "channel_mult": [1, 2],
    "num_classes": 1,
    "concat_emb": False,
}


def build_denoiser(**kw):
    return UNetModel(**{**DENOISER_ARGS, **kw})


def shapes(module):
    return {k: list(v.shape) for k, v in module.state_dict().items()}


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text())


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---- state-dict compatibility with the paper (snapshot) modules -------------------------------


@pytest.mark.parametrize(
    ("name", "build", "n"),
    [
        ("vae", build_vae, 6_375_529),
        ("discriminator", BaselineDiscriminator, 11_048_769),
        ("unet_prior", build_unet_prior, 9_949_832),
        ("diffusion_denoiser", build_denoiser, 9_798_020),
    ],
)
def test_paper_config_state_dict_matches_fixture(expected, name, build, n):
    module = build()
    got = shapes(module)
    # key order matters too for positional loading; compare lists of items
    assert list(got.items()) == list(expected[name].items())
    assert n_params(module) == n


def _load_original(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SNAPSHOT_MODELS.exists(), reason="research-code snapshot not available")
def test_state_dicts_match_original_modules_including_variants():
    ae = _load_original("orig_autoencoder", SNAPSHOT_MODELS / "autoencoder.py")
    df = _load_original("orig_diffusion", SNAPSHOT_MODELS / "diffusion.py")

    # paper configs and a few off-paper variants (attention, resblock up/down, single step, concat)
    orig_vae = VAE(
        ae.Encoder(in_channels=1, resolution=AE_RES, **AE_CFG),
        ae.Decoder(out_channels=1, resolution=AE_RES, **AE_CFG),
    )
    assert shapes(build_vae()) == shapes(orig_vae)
    assert shapes(BaselineDiscriminator()) == shapes(ae.BaselineDiscriminator())

    variants = [
        {},
        {"attention_resolutions": [1, 2], "num_heads": 2},
        {"resblock_updown": True, "use_scale_shift_norm": True},
        {"use_spatial_transformer": True, "context_dim": 16, "attention_resolutions": [2]},
        {"concat_emb": True},
        {"single_step": True},
        {"num_classes": None},
    ]
    for kw in variants:
        assert shapes(build_denoiser(**kw)) == shapes(df.UNetModel(**{**DENOISER_ARGS, **kw}))
    prior_variants = [{}, {"attention_resolutions": [2]}, {"resblock_updown": True}]
    for kw in prior_variants:
        assert shapes(build_unet_prior(**kw)) == shapes(
            df.UNetModelMeanVarSingleStep(**{**PRIOR_ARGS, **kw})
        )


def test_state_dict_round_trip_between_instances():
    a, b = (
        build_unet_prior(model_channels=32, channel_mult=[1]),
        build_unet_prior(model_channels=32, channel_mult=[1]),
    )
    missing = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing.missing_keys
    assert not missing.unexpected_keys


# ---- forward shapes on tiny configs ------------------------------------------------------------


def test_vae_forward_and_kl_shapes():
    vae = build_vae(resolution=[16, 16, 16], n_channels=16, ch_mult=[1, 2], num_res_blocks=1)
    x = torch.randn(2, 1, 16, 16, 16)
    x_hat, (mu, logsigma) = vae(x)
    assert x_hat.shape == x.shape
    assert mu.shape == logsigma.shape == (2, 4, 8, 8, 8)
    assert logsigma.max() <= 1.0  # clamped from above
    z = vae.encode(x)[0]
    assert torch.equal(z, mu)
    assert vae.decode(z).shape == x.shape
    assert vae.kl_loss(mu, logsigma).ndim == 0
    # deterministic path returns the mean
    a, _ = vae(x, stochastic_sampling=False)
    assert torch.allclose(a, vae.decode(mu))


def test_discriminator_patch_output():
    disc = BaselineDiscriminator(ndf=8)
    out = disc(torch.randn(1, 1, 32, 32, 32))
    assert out.shape == (1, 1, 2, 2, 2)


def test_unet_prior_forward_shapes():
    model = build_unet_prior(model_channels=32, channel_mult=[1, 2]).eval()
    x = torch.randn(2, 4, 8, 8, 8)
    out = model(x, y=torch.rand(2, 1))
    assert out.sample_mean.shape == out.sample_logvar.shape == x.shape
    # zero-initialised output convs
    assert torch.count_nonzero(out.sample_mean) == 0


def test_unet_prior_tolerates_rotated_spatial_shape():
    # NOTE(paper) Q12: training augments with rot90 (20x24x20 -> 24x20x20); the UNet is fully conv.
    model = build_unet_prior(model_channels=32).eval()
    out = model(torch.randn(1, 4, 8, 12, 8), y=torch.rand(1, 1))
    assert out.sample_mean.shape == (1, 4, 8, 12, 8)


def test_denoiser_forward_shapes_and_timestep_forms():
    model = build_denoiser(model_channels=32).eval()
    x = torch.randn(2, 8, 8, 8, 8)
    y = torch.rand(2, 1)
    out = model(x, timestep=torch.tensor([3, 7]), y=y).sample
    assert out.shape == (2, 4, 8, 8, 8)
    # scalar timestep tensor is broadcast over the batch
    assert model(x, timestep=torch.tensor(3), y=y).sample.shape == out.shape
    with pytest.raises(AssertionError):
        model(x, y=y)


# ---- losses ------------------------------------------------------------------------------------


def test_ssim_identity_and_ordering():
    x = torch.rand(2, 1, 16, 16, 16) * 2 - 1
    ssim = SSIM(data_range=1.0, size_average=False, channel=1, spatial_dims=3)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5)
    noisy = x + 0.5 * torch.randn_like(x)
    assert (ssim(x, noisy) < 0.99).all()


def test_gauss_loss_hand_computed():
    mean = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 4.0])
    logsigma2 = torch.tensor([0.0, np.log(4.0)]).float()
    loss = GaussLoss(sigma_eps=0.0, resi_min=0.0, resi_max=1e9)
    # resi = [1/1, 4/4] = [1, 1]; + logsigma2 = [1, 1 + ln 4]
    expected = (1.0 + 1.0 + np.log(4.0)) / 2
    assert loss(mean, logsigma2, target).item() == pytest.approx(expected, rel=1e-6)
    assert GaussLoss(reduction="sum", sigma_eps=0.0, resi_min=0.0, resi_max=1e9)(
        mean, logsigma2, target
    ).item() == pytest.approx(2 * expected, rel=1e-6)


def test_gauss_loss_clamps_and_reports_nan():
    mean, target = torch.zeros(3), torch.tensor([100.0, 0.0, 0.0])
    loss = GaussLoss(sigma_eps=0.0, resi_min=0.5, resi_max=10.0)
    # resi = [1e4, 0, 0] -> clamp -> [10, 0.5, 0.5]
    assert loss(mean, torch.zeros(3), target).item() == pytest.approx((10 + 0.5 + 0.5) / 3)
    assert loss(mean, torch.full((3,), float("nan")), target) is None


def test_temp_comb_gauss_and_prior_instance():
    mean = torch.tensor([1.0, -1.0])
    target = torch.zeros(2)
    logsigma2 = torch.zeros(2)
    combo = TempCombGaussLoss(sigma_eps=0.0, resi_min=0.0, resi_max=1e9)
    got = combo(mean, logsigma2, target, T1=2.0, T2=0.5)
    assert got.item() == pytest.approx(2.0 * 1.0 + 0.5 * 1.0)
    prior = make_prior_loss()
    assert (prior.sigma_eps, prior.resi_min, prior.resi_max) == (1e-6, 1e-6, 1e4)
    assert prior.L_Gauss.sigma_eps == 1e-6


def test_adversarial_loss_sign_conventions():
    fake, real = torch.full((4,), 0.3), torch.full((4,), 0.9)
    gen = AdversarialLoss("least_square", is_discriminator=False, weight=1.0)
    disc = AdversarialLoss("least_square", is_discriminator=True, weight=1.0)
    # generator pushes fake logits towards 1, discriminator fake->0 and real->1
    assert gen(fake).item() == pytest.approx((0.3 - 1) ** 2)
    assert disc(fake, real).item() == pytest.approx(0.5 * (0.3**2 + (0.9 - 1) ** 2))
    assert gen(real).item() < gen(fake).item()
    # weight scales the loss
    assert AdversarialLoss(is_discriminator=False, weight=0.005)(fake).item() == pytest.approx(
        0.005 * 0.49
    )
    hinge = AdversarialLoss("vanilla", is_discriminator=True, weight=1.0)
    assert hinge(torch.full((2,), -2.0), torch.full((2,), 2.0)).item() == 0.0
    softplus = AdversarialLoss("hinge", is_discriminator=False, weight=1.0)
    assert softplus(torch.tensor([0.0])).item() == pytest.approx(np.log(2.0))
    with pytest.raises(ValueError):
        AdversarialLoss(reduction="none", weight=1.0)


def test_lpips_runs_offline_on_five_slices():
    loss = LPIPS(pretrained=False, slice_size=5)
    x = torch.randn(1, 3, 32, 32, 32)
    y = torch.randn_like(x)
    out = loss(x, y)  # random axis, 5 random slices, averaged over the slice axis
    assert out.ndim == 5
    assert torch.isfinite(out).all()
    assert loss(x, x).abs().max() < 1e-6
    # 4-d input goes straight to the 2D metric
    assert loss(x[:, :, 0], y[:, :, 0]).shape == (1, 1, 1, 1)
