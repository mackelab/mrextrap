"""Training of the heteroscedastic UNet prior (mean and log-variance of the latent progression rate).

Ported from ``train_unet_heteroscedastic.py`` as pinned by the W&B code snapshot of run
``train_unet_mean_var_with_gauss_loss_lowlr_aug_warmup4000`` (``737ieovl``). ``configs/unet_prior.yaml``
holds that run's hyperparameters (not the later "WIDER" ones).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mrextrap.data.datasets import get_latent_dataloaders
from mrextrap.io import save_safetensors
from mrextrap.losses import make_prior_loss
from mrextrap.models.unet3d import Normalization, UNetModelMeanVarSingleStep


def legacy_epoch_label(epoch: int) -> str:
    """Directory label of the checkpoint written after (zero-based) ``epoch``.

    NOTE(paper): (Q4) the research code named checkpoints ``epoch_{(e+1)//20*20+20}`` and saved
    when ``(e+1) % 10 == 0``. The checkpoint labelled ``epoch_2000`` (used by all evaluations) is
    therefore written after zero-based epoch 1989, i.e. it holds the weights after 1990 epochs;
    the final epoch (1999) is written to ``epoch_2020``. Odd-decade saves overwrite the label of
    the next even decade (epochs 9 and 19 -> ``epoch_20`` / ``epoch_40``, epoch 29 -> ``epoch_40``).
    """
    return f"epoch_{(epoch + 1) // 20 * 20 + 20}"


def is_save_epoch(epoch: int, num_epochs: int, every: int = 10) -> bool:
    """Save/evaluate when ``(epoch+1) % every == 0`` (W&B code); also after the very last epoch
    so runs shorter than ``every`` epochs (smoke tests) still leave a checkpoint. For the paper's
    2000 epochs the extra condition is redundant."""
    return (epoch + 1) % every == 0 or epoch == num_epochs - 1


def output_dir(cfg: DictConfig, stage: str) -> Path:
    """``cfg.output_dir`` if set, else ``<paths.run_dir>/<stage>``, else ``exp/<exp_name>``."""
    if cfg.get("output_dir"):
        return Path(cfg.output_dir)
    if cfg.get("paths") is not None and cfg.paths.get("run_dir"):
        return Path(cfg.paths.run_dir) / stage
    return Path("exp") / cfg.exp_name


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_unet_prior(model_cfg) -> UNetModelMeanVarSingleStep:
    return UNetModelMeanVarSingleStep(
        image_size=list(model_cfg.image_size),
        in_channels=model_cfg.in_channels,
        model_channels=model_cfg.model_channels,
        out_channels=model_cfg.out_channels,
        num_res_blocks=model_cfg.num_res_blocks,
        attention_resolutions=list(model_cfg.attention_resolutions),
        channel_mult=list(model_cfg.channel_mult),
        num_classes=model_cfg.num_classes,
        emb_bias=model_cfg.emb_bias,
    )


def random_flip_random_dim(x, y, p=0.5):
    if torch.rand(1) < p:
        # choose a random spatial dimension from 2, 3, 4
        dim = torch.randint(2, 5, (1,)).item()
        return x.flip(dims=(dim,)), y.flip(dims=(dim,))
    return x, y


def random_rotate_90_dim(x, y, p=0.5):
    if torch.rand(1) < p:
        # choose two distinct dimensions from 2, 3, 4
        dims = tuple(int(d) for d in (torch.randperm(3)[:2] + 2))
        k = torch.randint(1, 3, (1,)).item()  # 1 or 2 quarter turns (never 3)
        # NOTE(paper): (Q12) rotating over two dims of unequal size changes the spatial shape
        # (e.g. 20x24x20 -> 24x20x20); training relies on the fully convolutional UNet.
        return torch.rot90(x, k=k, dims=dims), torch.rot90(y, k=k, dims=dims)
    return x, y


def augment_data(x, y, p_flip=0.5, p_rotate=0.5):
    """Same random flip / 90-degree rotation of latent ``x`` and target ``y`` (25% unchanged)."""
    x, y = random_flip_random_dim(x, y, p_flip)
    x, y = random_rotate_90_dim(x, y, p_rotate)
    return x, y


def save_normalization(stats: dict, cfg: DictConfig, out_dir: Path) -> None:
    """Write the train-latent statistics used (``priors/normalization.pt`` schema)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out_dir / "normalization.pt")
    if cfg.get("paths") is not None and cfg.paths.get("priors_dir"):
        Path(cfg.paths.priors_dir).mkdir(parents=True, exist_ok=True)
        torch.save(stats, Path(cfg.paths.priors_dir) / "normalization.pt")


def train_unet_prior(cfg: DictConfig) -> Path:
    """Train the UNet prior; returns the stage output directory.

    Layout: ``<out>/config.yaml``, ``<out>/normalization.pt``,
    ``<out>/checkpoints/<legacy_epoch_label>/{model.safetensors, state/}``.
    """
    from accelerate import Accelerator
    from diffusers.optimization import get_scheduler

    out_dir = output_dir(cfg, "unet_prior")
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    seed_everything(cfg.training.seed)

    model = build_unet_prior(cfg.model)
    train_dataloader, val_dataloader, _ = get_latent_dataloaders(
        cfg, num_workers=cfg.training.get("num_workers", 4)
    )
    dataset = train_dataloader.dataset
    age_normalization = Normalization(dataset.ages_abs.mean(), dataset.ages_abs.std())
    save_normalization(dataset.normalization_stats(), cfg, out_dir)

    # NOTE(paper): AdamW betas (0.95, 0.999), wd 1e-6 (as in the code; the paper text differs).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8
    )
    # NOTE(paper): cosine horizon is 1.3x the real number of steps, so the lr ends at ~20% of peak.
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.num_warmup_steps,
        num_training_steps=len(train_dataloader) * cfg.training.num_epochs * 1.3,
    )
    loss_fn = make_prior_loss()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb" if cfg.training.use_wandb else None,
        cpu=bool(cfg.training.get("cpu", False)),  # force CPU (tests; avoids MPS on macOS)
        dynamo_backend="INDUCTOR" if cfg.training.torch_compile else None,
    )
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler, age_normalization = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler, age_normalization
        )
    )
    if cfg.training.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            "mrextrap",
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.exp_name}},
        )

    ckpt_root = out_dir / "checkpoints"
    history: list[dict] = []
    global_step = 0
    num_epochs = cfg.training.num_epochs
    for epoch in tqdm(
        range(num_epochs), desc="Epochs", disable=not accelerator.is_local_main_process
    ):
        accelerator.wait_for_everyone()
        model.train()
        for batch in train_dataloader:
            x_cond = batch["zs"]  # latent of the conditioning scan
            age0_norm = age_normalization(batch["ages_abs"].unsqueeze(1).float())
            beta = batch["betas"]  # target beta

            if cfg.training.augment_data:
                x_cond, beta = augment_data(x_cond, beta)

            out = model(x_cond, y=age0_norm)
            pred_mean, pred_logvar = out.sample_mean, out.sample_logvar
            loss = loss_fn(pred_mean, pred_logvar, beta, T1=cfg.training.T1, T2=cfg.training.T2)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_val)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "l1_loss": F.l1_loss(pred_mean, beta).detach().item(),
                "gauss_loss": loss_fn.L_Gauss(pred_mean, pred_logvar, beta).detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
            history.append(logs)
            if global_step % 5 == 0 and cfg.training.use_wandb:
                accelerator.log(logs, step=global_step)

        save_now = is_save_epoch(epoch, num_epochs)
        if save_now and accelerator.is_main_process:
            if torch.isnan(loss).any():
                raise ValueError("Loss is NaN. Training failed.")
            # NOTE(paper): (Q4) the label is not the number of completed epochs.
            save_dir = ckpt_root / legacy_epoch_label(epoch)
            save_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(output_dir=str(save_dir / "state"))
            save_safetensors(
                accelerator.unwrap_model(model, keep_fp32_wrapper=False).state_dict(),
                save_dir / "model.safetensors",
            )
        accelerator.wait_for_everyone()

        if save_now:
            model.eval()
            val = {"val_loss": [], "val_l1_loss": [], "val_gauss_loss": []}
            with torch.no_grad():
                for batch in val_dataloader:
                    age0_norm = age_normalization(batch["ages_abs"].unsqueeze(1).float())
                    out = model(batch["zs"], y=age0_norm)
                    b = batch["betas"]
                    v = loss_fn(
                        out.sample_mean,
                        out.sample_logvar,
                        b,
                        T1=cfg.training.T1,
                        T2=cfg.training.T2,
                    )
                    val["val_loss"].append(accelerator.gather(v.reshape(1)).mean().item())
                    val["val_l1_loss"].append(
                        accelerator.gather(F.l1_loss(out.sample_mean, b).reshape(1)).mean().item()
                    )
                    val["val_gauss_loss"].append(
                        accelerator.gather(
                            loss_fn.L_Gauss(out.sample_mean, out.sample_logvar, b).reshape(1)
                        )
                        .mean()
                        .item()
                    )
            val_logs = {k: float(np.mean(v)) for k, v in val.items()}
            history.append({"epoch": epoch, **val_logs})
            if cfg.training.use_wandb:
                accelerator.log(val_logs, step=global_step)
            if accelerator.is_local_main_process:
                print(
                    f"Epoch {epoch + 1} validation: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in val_logs.items())
                )
        accelerator.wait_for_everyone()

    accelerator.end_training()
    (out_dir / "train_log.json").write_text(json.dumps(history))
    return out_dir


def load_unet_prior(
    cfg: DictConfig, checkpoint_dir: str | os.PathLike
) -> UNetModelMeanVarSingleStep:
    """Build the model from ``cfg.model`` and load ``<checkpoint_dir>/model.safetensors``."""
    from mrextrap.io import load_state_dict_any

    model = build_unet_prior(cfg.model)
    model.load_state_dict(load_state_dict_any(Path(checkpoint_dir) / "model.safetensors"))
    return model.eval()
