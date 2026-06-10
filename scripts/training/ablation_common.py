"""Shared training logic for the LoaDiff ablation studies.

The goal of the ablations is to measure the contribution of each LoaDiff
component **without touching** the production scripts (``train_loadiff.py``,
``inference_loadiff.py``, ...) nor the reference model / dataset code.

To avoid duplicating the (already nicely factored) training loop of
``scripts/training/train_loadiff.py``, this module *imports* its reusable
primitives and only re-implements the thin orchestration layer
(:func:`train_ablation`) so it can inject the few ablation-specific behaviours:

  - a configurable model builder (AdaLN vs concat conditioning, CFG dropout
    probabilities, number of exogenous variables, attention heads, ...);
  - an optional suppression of the exogenous calendar/temperature signal, used
    by the "classic positional encoding" ablations, where the model falls back
    to the fixed 2D sin/cos positional embedding instead of learned calendar
    features.

Every ablation entry-point (``train_loadiff_ablation_*.py``) is a tiny Hydra
wrapper that picks its dedicated config and calls :func:`train_ablation`.
All ablation behaviour is therefore driven purely by the YAML configs under
``configs/ablations/``.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.loadit.models import DiT
from src.loadit.models_ablations import DiTConcatCond
from src.loadit.diffusion import create_diffusion
from src.helpers.optim.lr_scheduler import get_scheduler
from src.helpers.training_utils import update_ema, requires_grad, create_logger
from src.evaluation.features_extractor import ROCKET

# Reuse the *exact* primitives from the production training script so the
# ablations stay faithful to the reference training dynamics.
from scripts.training.train_loadiff import (
    resolve_conditioning,
    build_dataloaders,
    prepare_batch,
    compute_losses,
    backward_step,
    log_train_step,
    save_checkpoint,
    maybe_resume,
    evaluate_valid_loss,
    run_validation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation-aware model builder
# ---------------------------------------------------------------------------
def build_model_ablation(args: DictConfig, num_classes: int, device: torch.device):
    """Build the DiT (or an ablation variant) together with its EMA + diffusion.

    Extra (optional) keys read from ``args.ditmodelargs``:
      - ``conditioning`` (str): ``"adaln"`` (default) or ``"concat"`` -> selects
        :class:`DiTConcatCond` for the concat-conditioning ablation.
      - ``class_dropout_prob`` (float, default 0.1): label CFG dropout. Set to
        ``0.0`` for the "no classifier-free guidance" ablation.
      - ``temp_dropout_prob`` (float, default 0.1): temperature CFG dropout.
      - ``num_heads`` (int, default 8): transformer attention heads.

    The ``conditioning`` / dropout / num_heads keys are *additive*: when absent
    the model is built exactly like ``train_loadiff.build_model``.
    """
    dm = args.ditmodelargs
    conditioning = str(OmegaConf.select(dm, "conditioning", default="adaln")).lower()
    class_dropout_prob = float(OmegaConf.select(dm, "class_dropout_prob", default=0.1))
    temp_dropout_prob = float(OmegaConf.select(dm, "temp_dropout_prob", default=0.1))
    num_heads = int(OmegaConf.select(dm, "num_heads", default=8))

    model_cls = DiTConcatCond if conditioning == "concat" else DiT
    logger.info(
        "Building model: cls=%s, hidden_size=%s, depth=%s, num_heads=%s, "
        "n_exo_var=%s, temperature=%s, num_classes=%s, "
        "class_dropout_prob=%s, temp_dropout_prob=%s",
        model_cls.__name__, dm.hidden_size, dm.depth, num_heads,
        dm.n_exo_var, dm.temperature, num_classes,
        class_dropout_prob, temp_dropout_prob,
    )

    model = model_cls(
        input_size=dm.input_size,
        patch_size=dm.patch_size,
        in_channels=dm.in_channels,
        depth=dm.depth,
        hidden_size=dm.hidden_size,
        num_heads=num_heads,
        n_exo_var=dm.n_exo_var,
        temperature=dm.temperature,
        num_classes=num_classes,
        multilabels=dm.multilabels,
        class_dropout_prob=class_dropout_prob,
        temp_dropout_prob=temp_dropout_prob,
    )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model.to(device)

    # The default "linear" beta schedule is parameterised as scale = 1000 / steps,
    # so beta_end = scale * 0.02 exceeds 1.0 (invalid) for very small step counts
    # (e.g. 10 steps -> beta_end = 2.0). The bounded cosine schedule
    # ("squaredcos_cap_v2") stays valid for any number of steps; expose it via
    # the optional ``training.noise_schedule`` key (default "linear").
    noise_schedule = str(OmegaConf.select(args, "training.noise_schedule", default="linear"))
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=noise_schedule,
        diffusion_steps=args.training.diffusion_steps,
    )

    return model, ema, diffusion


# ---------------------------------------------------------------------------
# Exogenous suppression (classic positional encoding ablations)
# ---------------------------------------------------------------------------
def _disable_exog(diffusion) -> None:
    """Neutralise ``diffusion.set_exog`` so the model never receives exog.

    When the calendar/temperature exogenous tensor is never registered, the
    wrapped model is called as ``model(x, t)`` and the DiT falls back to its
    fixed 2D sin/cos positional embedding (the ``n_exo_var == 0`` branch). This
    is exactly the "classic PE instead of learned calendar features" ablation.
    """
    diffusion.exog = None

    def _noop_set_exog(_exog, _d=diffusion):
        _d.exog = None

    diffusion.set_exog = _noop_set_exog  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Training loop (mirror of train_loadiff.train, with injectable behaviour)
# ---------------------------------------------------------------------------
def train_ablation(args: DictConfig) -> None:
    """Run a LoaDiff training using ablation-specific knobs from the config.

    This is a faithful re-implementation of ``train_loadiff.train`` that swaps
    in :func:`build_model_ablation` and honours the optional
    ``ablation.suppress_exog`` flag. All other behaviour (data, optimisation,
    checkpointing, validation/metrics logging) is delegated to the imported
    production primitives.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting ABLATION training, device: {device}")

    torch.manual_seed(args.training.global_seed)

    bool_col_names, num_classes, conditional, use_temperature = resolve_conditioning(args)
    suppress_exog = bool(OmegaConf.select(args, "ablation.suppress_exog", default=False))
    ablation_name = str(OmegaConf.select(args, "ablation.name", default="unnamed"))

    logging.info(
        f"[ablation={ablation_name}] conditional={conditional} "
        f"(num_classes={num_classes}, labels={bool_col_names}), "
        f"temperature={use_temperature}, suppress_exog={suppress_exog}"
    )

    # Setup tensorboard / logging
    log_dir = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = f"{log_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger_local = create_logger(log_dir)
    logger_local.info(f"[ablation={ablation_name}] Experiment directory created at {log_dir}")

    # Model + diffusion (ablation-aware)
    model, ema, diffusion = build_model_ablation(args, num_classes, device)
    logger_local.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if suppress_exog:
        _disable_exog(diffusion)
        logger_local.info("Exogenous conditioning disabled: using classic sin/cos positional encoding.")

    # Data
    logging.info("Data loading...")
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = build_dataloaders(args)
    logger_local.info(f"Dataset contains {len(train_dataset):,} curves ({args.data.data_path})")

    # Optional horizon override (patch-size ablations): shrink the number of
    # daily windows so a multi-day token (e.g. a 7-day = 336-step patch) divides
    # the horizon cleanly. Acts on the in-memory datasets only.
    nb_days_override = OmegaConf.select(args, "ablation.nb_days", default=None)
    if nb_days_override is not None:
        nb_days_override = int(nb_days_override)
        for ds in (train_dataset, valid_dataset):
            if nb_days_override > ds.available_days:
                raise ValueError(
                    f"ablation.nb_days={nb_days_override} exceeds available_days={ds.available_days}."
                )
            ds.nb_days = nb_days_override
        logger_local.info(f"Horizon overridden: nb_days={nb_days_override} (per dataset window).")

    feature_extractor = ROCKET(
        seq_len=valid_dataset.nb_days * valid_dataset.patch_length,
        n_kernels=1000,
        seed=args.training.global_seed,
    ).to(device)

    # Optimizer + scheduler
    logging.info("Setting optimizer and schedulers...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.lr, weight_decay=args.training.weight_decay)
    scheduler = get_scheduler(
        args.training.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=args.training.n_warmup_steps,
        num_training_steps=args.training.max_steps,
    )

    # Prepare models for training:
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # Optional resume
    passed_steps = maybe_resume(args, model, ema, optimizer, scheduler, device, logger_local)

    min_checkpoint_step = args.training.get("min_checkpoint_step", 0)

    epoch = 0
    train_losses = []
    valid_losses = []

    logger_local.info(f"Training for {args.training.max_steps} steps...")
    while passed_steps < args.training.max_steps:

        total_loss = 0.0
        epoch_passed_steps = 0

        logger_local.info(f"Beginning epoch {epoch}...")
        for batch in train_dataloader:

            x, model_kwargs = prepare_batch(batch, diffusion, device, conditional)
            loss = compute_losses(diffusion, model, x, model_kwargs)

            backward_step(loss, optimizer)
            total_loss += loss.item()

            update_ema(ema, model, decay=args.training.ema_decay)

            passed_steps += 1
            epoch_passed_steps += 1

            log_train_step(writer, loss.item(), optimizer, passed_steps)
            scheduler.step()

            if passed_steps % args.training.log_every == 0:
                logging.info(f"Train steps : {passed_steps}. Will save checkpoint!")

                logging.info("Validation loss ...")
                val_loss = evaluate_valid_loss(model, diffusion, valid_dataloader, device, conditional)
                valid_losses.append(val_loss)
                writer.add_scalar("Valid/loss", val_loss, passed_steps)

                if passed_steps >= min_checkpoint_step:
                    save_checkpoint(checkpoint_dir, passed_steps, model, ema,
                                    optimizer, scheduler, args, logger_local)

                run_validation(args, passed_steps, ema, diffusion, feature_extractor,
                               valid_dataset, train_dataloader, device, writer,
                               conditional, use_temperature)

        train_losses.append(total_loss / max(epoch_passed_steps, 1))
        writer.add_scalar("Train/Epoch_total_loss", train_losses[-1], epoch)
        epoch += 1

    logger_local.info(f"[ablation={ablation_name}] Done!")
