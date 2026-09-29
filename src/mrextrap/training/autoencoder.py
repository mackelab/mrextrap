"""Autoencoder training (VAE + PatchGAN discriminator, L1 + SSIM + LPIPS + KL + adversarial loss).

Ported from ``train_autoencoder.py`` of the research code. Behaviour is preserved: loss terms and
weights, AdamW (0.95, 0.999) with weight decay 1e-6 for both networks, cosine schedule with 10000
warm-up steps and a x1.3 horizon, delayed discriminator start, KL / adversarial ramps, gradient
clipping, accelerate mixed precision and checkpointing. Quirks are marked ``NOTE(paper)``.

Entry points: ``train_autoencoder(cfg) -> Path`` (CLI ``mrextrap train-ae``), ``build_vae``,
``ae_generator_loss``, ``ae_discriminator_loss``, ``kl_weight``, ``adv_weight``.
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from mrextrap.io import _strip_prefixes, save_safetensors
from mrextrap.losses import LPIPS, SSIM, AdversarialLoss
from mrextrap.models import autoencoder

MODEL_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.yaml"


# --------------------------------------------------------------------------------------------
# construction helpers
# --------------------------------------------------------------------------------------------


def build_vae(model_cfg) -> autoencoder.VAE:
    """Build the VAE from ``cfg.model`` (n_channels, z_channels, ch_mult, num_res_blocks, resolution)."""
    kwargs = {
        "n_channels": model_cfg.n_channels,
        "z_channels": model_cfg.z_channels,
        "ch_mult": model_cfg.ch_mult,
        "num_res_blocks": model_cfg.num_res_blocks,
        "resolution": model_cfg.resolution,
    }
    encoder = autoencoder.Encoder(in_channels=1, **kwargs)
    decoder = autoencoder.Decoder(out_channels=1, **kwargs)
    return autoencoder.VAE(encoder=encoder, decoder=decoder)


def output_dir(cfg) -> Path:
    """``cfg.output_dir`` if set, else ``<paths.run_dir>/autoencoder``."""
    if cfg.get("output_dir", None):
        return Path(str(cfg.output_dir))
    return Path(str(cfg.paths.run_dir)) / "autoencoder"


def legacy_ae_epoch_label(epoch: int) -> int:
    """Checkpoint label of the research code for the 0-based ``epoch``: ``epoch // 2 * 2 + 2``.

    NOTE(paper): the state is saved after every epoch under this label, so labels are overwritten
    in pairs. For an odd ``epoch`` the label equals the number of finished epochs; for an even one
    it is one too large until the next epoch overwrites it. ``epoch_124`` (the paper autoencoder)
    therefore holds the state after 124 finished epochs if training stopped there.
    """
    return epoch // 2 * 2 + 2


def kl_weight(step: int, cfg, steps_per_epoch: int) -> float:
    """KL weight at ``step``: linear ramp over ``min(0.3 * epochs, warmup_epochs)`` epochs.

    Not used (constant ``kl_beta``) when ``training.finetune`` is true.
    """
    t = cfg.training
    if t.get("finetune", False):
        return t.kl_beta
    warmup_epochs = t.get("warmup_epochs", 1)
    return t.kl_beta * min(1.0, step / (steps_per_epoch * min(0.3 * t.num_epochs, warmup_epochs)))


def adv_weight(step: int, epoch: int, cfg, steps_per_epoch: int) -> float:
    """Ramp factor of the adversarial generator loss (1.0 before the discriminator starts).

    NOTE(paper) Q3: ``step`` is the global step since the start of training (not since
    ``disc_epoch_start``), so at ``disc_epoch_start`` (epoch 25 for 125 epochs) the ramp is
    already 1. The paper text says the weight is "linearly increased over one epoch".
    """
    t = cfg.training
    if epoch >= disc_epoch_start(cfg):
        warmup_epochs = t.get("warmup_epochs", 1)
        return min(1.0, step / (steps_per_epoch * min(0.3 * t.num_epochs, warmup_epochs)))
    return 1.0


def disc_epoch_start(cfg) -> int:
    value = cfg.training.get("disc_epoch_start", None)
    return int(cfg.training.num_epochs * 0.2) if value is None else int(value)


# --------------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------------


@dataclass
class AutoencoderLosses:
    """The loss modules used by the training step."""

    discriminator: nn.Module
    l1: nn.Module
    ssim: nn.Module
    adv_generator: nn.Module
    adv_discriminator: nn.Module
    perceptual: nn.Module | None  # None when lpips_beta == 0 (test path, skips building LPIPS)

    @classmethod
    def create(cls, discriminator, disc_beta: float, perceptual=None) -> AutoencoderLosses:
        return cls(
            discriminator=discriminator,
            l1=nn.L1Loss(reduction="none"),
            ssim=SSIM(data_range=1.0, size_average=False, channel=1, spatial_dims=3),
            adv_generator=AdversarialLoss(
                is_discriminator=False, weight=disc_beta
            ),  # manual weighting
            adv_discriminator=AdversarialLoss(is_discriminator=True, weight=disc_beta),
            perceptual=perceptual,
        )


def kl_loss(mu, logsigma, prior_precision=1.0):
    """Per-sample KL divergence to a standard normal (summed over channel and space)."""
    kl = -0.5 * (1 + logsigma - prior_precision * mu.pow(2) - prior_precision * logsigma.exp())
    return kl.sum([1, 2, 3, 4])


def ae_discriminator_loss(losses: AutoencoderLosses, x, xhat):
    d = losses.discriminator
    loss = losses.adv_discriminator(d(xhat), d(x))
    if loss.isnan().any():
        print("discriminator loss is nan")
        print("discriminator loss: ", loss)
    return loss


def _warn_nan(name, value, count):
    if count > 0:
        warnings.warn(f"encountered nan components in {name}: {value}")


def ae_generator_loss(
    losses: AutoencoderLosses,
    x,
    xhat,
    mu,
    logsigma,
    kl_beta,
    ssim_beta,
    lpips_beta,
    use_adv_loss=True,
    adv_loss_warmup_beta=1.0,
) -> dict:
    """Generator loss: (L1 + SSIM) / B + adversarial + LPIPS + KL / numel (see NOTE(paper) below)."""
    rec_loss = losses.l1(xhat, x).mean([1, 2, 3, 4])
    n_nan = torch.isnan(rec_loss).sum()
    rec_loss = rec_loss.sum()
    _warn_nan("rec_loss", rec_loss, n_nan)

    ssim_loss = (1 - losses.ssim(xhat, x)) * ssim_beta
    n_nan = torch.isnan(ssim_loss).sum()
    ssim_loss = ssim_loss.sum()
    _warn_nan("ssim_loss", ssim_loss, n_nan)

    emb_loss = kl_loss(mu, logsigma)
    n_nan = torch.isnan(emb_loss).sum()
    emb_loss = emb_loss.sum()
    _warn_nan("emb_loss", emb_loss, n_nan)

    adv_loss = 0.0
    if use_adv_loss:
        adv_loss = losses.adv_generator(losses.discriminator(xhat))
        n_nan = torch.isnan(adv_loss).sum()
        adv_loss = adv_loss.sum() * adv_loss_warmup_beta
        _warn_nan("adv_loss", adv_loss, n_nan)

    perc_loss = 0.0
    if losses.perceptual is not None:
        perc_loss = losses.perceptual(target=x, pred=xhat)
        # manual weighting according to the MONAI generative repository
        perc_loss = perc_loss.sum() * lpips_beta

    # NOTE(paper): the KL term is summed over the batch and divided by x.numel() (batch included),
    # the L1/SSIM terms are divided by the batch size, and the LPIPS / adversarial terms are not.
    total_loss = (
        (rec_loss + ssim_loss) / x.shape[0]
        + adv_loss
        + perc_loss
        + (emb_loss * kl_beta / x.numel())
    )
    return {
        "total_loss": total_loss,
        "rec_loss": rec_loss / x.shape[0],
        "ssim_loss": ssim_loss / x.shape[0],
        "emb_loss": emb_loss / x.numel(),
        "adv_loss": adv_loss,
        "perc_loss": perc_loss,
    }


# --------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------


def _item(v):
    return v if isinstance(v, float) else v.detach().item()


def _save_checkpoint(accelerator, vae, out_dir: Path, epoch: int, cfg) -> None:
    """accelerate state under ``checkpoints/epoch_<legacy label>/`` plus exported plain weights."""
    ckpt = out_dir / "checkpoints" / f"epoch_{legacy_ae_epoch_label(epoch)}"
    ckpt.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(output_dir=str(ckpt))
    sd = _strip_prefixes(accelerator.unwrap_model(vae).state_dict())
    save_safetensors(sd, ckpt / MODEL_FILENAME)
    save_safetensors(sd, out_dir / MODEL_FILENAME)  # latest weights, plain VAE state dict


def run_training(cfg: DictConfig) -> tuple[Path, list[dict]]:
    """Train the autoencoder; returns ``(output_dir, per-step history of logged scalars)``."""
    from accelerate import Accelerator
    from diffusers.optimization import get_scheduler

    from mrextrap.data.datasets import get_autoencoder_dataloaders

    t = cfg.training
    out_dir = output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / CONFIG_FILENAME)

    ## set seed
    torch.manual_seed(t.seed)
    random.seed(t.seed)
    np.random.seed(t.seed)

    vae = build_vae(cfg.model)

    ## discriminator, adversarial and perceptual loss
    discriminator = autoencoder.BaselineDiscriminator()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    perceptual = None
    if t.lpips_beta != 0:
        perceptual = LPIPS(slice_size=5, pretrained=t.get("lpips_pretrained", True))
    losses = AutoencoderLosses.create(discriminator, t.disc_beta, perceptual)

    disc_start = disc_epoch_start(cfg)

    train_dataloader, val_dataloader, _ = get_autoencoder_dataloaders(
        cfg, num_workers=t.num_workers
    )

    ## optimisers and schedules
    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=t.lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8
    )
    # NOTE(paper): taken from the unsharded dataloader; accelerate's scheduler wrapper steps once
    # per process (see below), so the horizon is "technically reduced by the number of GPUs".
    steps_per_epoch = len(train_dataloader)
    num_warmup_steps = t.get("num_warmup_steps", 10000)  # NOTE(paper): hard-coded 10000 originally
    lr_scheduler = get_scheduler(
        t.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(steps_per_epoch * t.num_epochs * 1.3),  # lr ends at ~20% of initial
    )
    optimizer_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=t.lr,
        betas=(0.95, 0.999),  # patchgan beta1 is 0.5, but we use 0.95
        weight_decay=1e-6,
        eps=1e-8,
    )
    lr_scheduler_disc = get_scheduler(
        t.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=int(steps_per_epoch * (t.num_epochs - disc_start) * 1.3),
    )

    ## accelerate
    accelerator = Accelerator(
        # NOTE(paper): gradient_accumulation_steps is passed on but the loop never uses
        # accelerator.accumulate(), so gradients are NOT accumulated (effective no-op).
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        mixed_precision=t.mixed_precision,
        log_with="wandb" if t.get("use_wandb", False) else None,
        dynamo_backend="INDUCTOR" if t.get("torch_compile", False) else "no",  # torch.compile
        cpu=t.get("cpu", not torch.cuda.is_available()),
    )
    prepared = accelerator.prepare(
        vae, optimizer, train_dataloader, val_dataloader, lr_scheduler, discriminator,
        *([perceptual] if perceptual is not None else []),
        optimizer_disc, lr_scheduler_disc,
    )  # fmt: skip
    vae, optimizer, train_dataloader, val_dataloader, lr_scheduler, discriminator = prepared[:6]
    if perceptual is not None:
        losses.perceptual = prepared[6]
    optimizer_disc, lr_scheduler_disc = prepared[-2:]
    losses.discriminator = discriminator

    if t.get("finetune", False):
        accelerator.load_state(t.accelerator_state_path)

    if t.get("use_wandb", False) and accelerator.is_main_process:
        accelerator.init_trackers(
            "mrextrap",
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.exp_name}},
        )

    ##------training-------
    global_step = 0
    num_skipped_batches = 0
    loss_disc = 0.0
    history: list[dict] = []
    max_steps = t.get("max_steps_per_epoch", None)  # debugging aid (was a commented-out `break`)
    discriminator.eval()

    for epoch in tqdm(
        range(t.num_epochs), desc="Epochs", disable=not accelerator.is_local_main_process
    ):
        accelerator.wait_for_everyone()
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )

        for step, batch in enumerate(train_dataloader):
            if max_steps is not None and step >= max_steps:
                break
            x = batch

            vae.train()
            discriminator.eval()

            ## forward pass
            xhat, (mu, logsigma) = vae(x, stochastic_sampling=True)
            kl_beta = kl_weight(global_step, cfg, steps_per_epoch)
            ssim_beta = 1.0
            lpips_beta = t.lpips_beta
            adv_loss_warmup_beta = adv_weight(global_step, epoch, cfg, steps_per_epoch)

            ## generator loss and backprop
            loss_dict = ae_generator_loss(
                losses,
                x=x,
                xhat=xhat,
                mu=mu,
                logsigma=logsigma,
                kl_beta=kl_beta,
                ssim_beta=ssim_beta,
                lpips_beta=lpips_beta,
                use_adv_loss=(epoch >= disc_start),
                adv_loss_warmup_beta=adv_loss_warmup_beta,
            )
            loss = loss_dict["total_loss"]
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(vae.parameters(), t.get("grad_clip_val", 1.0))
            optimizer.step()
            lr_scheduler.step()

            ## discriminator training
            loss_disc = 0.0
            if epoch >= disc_start:
                discriminator.train()
                loss_disc = ae_discriminator_loss(losses, x=x, xhat=xhat.detach())
                # NOTE(paper): the generator's adversarial backward above already left gradients
                # on the discriminator parameters (zero_grad only runs at the end of the step),
                # so this backward adds to them and the discriminator update includes them.
                accelerator.backward(loss_disc)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        discriminator.parameters(), t.get("grad_clip_val", 1.0)
                    )
                optimizer_disc.step()
                lr_scheduler_disc.step()

            ## reset gradients
            optimizer.zero_grad(set_to_none=True)
            optimizer_disc.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1

            ## log scalars
            logs = {
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                "kl_beta": kl_beta,
                "lpips_beta": lpips_beta,
                "ssim_beta": ssim_beta,
                "num_skipped_batches": num_skipped_batches,
                "loss_disc": _item(loss_disc),
            }
            logs.update({k: _item(v) for k, v in loss_dict.items()})

            if torch.is_tensor(loss) and torch.isnan(loss).any():
                raise ValueError("Loss is NaN. Training failed.")
            if torch.is_tensor(loss_disc) and torch.isnan(loss_disc).any():
                raise ValueError("Discriminator Loss is NaN. Training failed.")

            del loss_dict, loss_disc, loss  # to ensure garbage collection
            progress_bar.set_postfix(**logs)
            history.append(logs)
            if global_step % 5 == 0:
                accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        ## validation SSIM (main process) and checkpoint
        if accelerator.is_main_process:
            if (epoch + 1) % 10 == 0 or epoch == t.num_epochs - 1:
                unwrapped_vae = accelerator.unwrap_model(vae)
                unwrapped_vae.eval()
                ssims = []
                for batch in val_dataloader:
                    x = batch
                    with torch.no_grad():
                        xhat, _ = unwrapped_vae(x, stochastic_sampling=False)
                        ssims.append(losses.ssim(xhat, x).mean().cpu().item())
                val_ssim = float(np.mean(ssims)) if ssims else float("nan")
                accelerator.log({"ssim_val": val_ssim}, step=epoch)
                history.append({"epoch": epoch, "ssim_val": val_ssim})
            _save_checkpoint(accelerator, vae, out_dir, epoch, cfg)
        accelerator.wait_for_everyone()

    accelerator.end_training()
    if accelerator.is_main_process:
        (out_dir / "history.jsonl").write_text("\n".join(json.dumps(h) for h in history) + "\n")
    return out_dir, history


def train_autoencoder(cfg: DictConfig) -> Path:
    """Train the autoencoder described by ``cfg`` and return its output directory.

    Writes ``config.yaml``, ``model.safetensors`` (latest weights), ``history.jsonl`` and
    ``checkpoints/epoch_<label>/`` (accelerate state + ``model.safetensors``) to
    ``cfg.output_dir`` (default ``<paths.run_dir>/autoencoder``).
    """
    return run_training(cfg)[0]
