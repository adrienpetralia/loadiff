import torch
import matplotlib.pyplot as plt
import logging
import os
import pickle
import hydra
import datetime
import pandas as pd
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from omegaconf import DictConfig

from src.loadit.models import DiT
from src.loadit.diffusion import create_diffusion
from src.helpers.optim.lr_scheduler import get_scheduler
from src.helpers.dataset import SmachDataset, CERDataset, CERBisDataset
from src.helpers.training_utils import update_ema, requires_grad, create_logger
from src.helpers.plotting import plot_multiple_clients
from src.evaluation.training_metrics import frechet_distance, get_all_metrics
from src.evaluation.features_extractor import ROCKET
from src.evaluation.dispare.timeseriesdata import TimeSeriesData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASETS = {
    "smach": SmachDataset,
    "cer": CERDataset,
    "cer_bis": CERBisDataset,
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def resolve_conditioning(args):
    """
    Derive the conditioning setup from the configuration.

    Returns a tuple ``(bool_col_names, num_classes, conditional, use_temperature)``
    where:
      - ``bool_col_names`` is the (possibly empty) list of label columns;
      - ``num_classes`` is its length (0 => unconditional model);
      - ``conditional`` is True when at least one label is used;
      - ``use_temperature`` is True when a temperature file is provided.
    """
    bool_col_names = list(args.data.bool_col_names) if "bool_col_names" in args.data else []
    num_classes = len(bool_col_names)
    conditional = num_classes > 0
    use_temperature = args.data.get("path_temperature", None) is not None

    return bool_col_names, num_classes, conditional, use_temperature


# ---------------------------------------------------------------------------
# Model / data / optimization builders
# ---------------------------------------------------------------------------
def build_model(args, num_classes, device):
    """
    Build the DiT model together with its (frozen) EMA copy and the diffusion.
    """
    model = DiT(
        input_size=args.ditmodelargs.input_size,
        patch_size=args.ditmodelargs.patch_size,
        in_channels=args.ditmodelargs.in_channels,
        depth=args.ditmodelargs.depth,
        hidden_size=args.ditmodelargs.hidden_size,
        n_exo_var=args.ditmodelargs.n_exo_var,
        temperature=args.ditmodelargs.temperature,
        num_classes=num_classes,
        multilabels=args.ditmodelargs.multilabels,
    )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model.to(device)

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.training.diffusion_steps)

    return model, ema, diffusion


def build_dataset(args, list_pdl):
    """
    Instantiate the dataset for the configured dataset name.

    Optional metadata / temperature / labels are forwarded straight from the
    configuration: when the corresponding keys are absent (unconditional
    setup) the dataset falls back to its neutral defaults.
    """
    if args.data.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{args.data.dataset}'. Available: {list(DATASETS)}")
    CdCDataset = DATASETS[args.data.dataset]

    bool_col_names = list(args.data.bool_col_names) if "bool_col_names" in args.data else None

    return CdCDataset(
        path_load_curves=args.data.data_path,
        list_pdl=list_pdl,
        scale_param2=args.data.value_scale,
        scale_meteo=args.data.get("value_scale_meteo", 1.0),
        random_window=False,
        path_metadata=args.data.get("path_parquet_part_metadata", None),
        bool_col_names=bool_col_names,
        path_temperature=args.data.get("path_temperature", None),
    )


def build_dataloaders(args):
    """
    Load the client split, build train/valid datasets and their dataloaders.
    """
    with open(args.data.path_client_split, "rb") as f:
        splits = pickle.load(f)
    train_clients = splits["train"]
    val_clients = splits["val"]

    logging.info("Loading train dataset...")
    train_dataset = build_dataset(args, train_clients)

    logging.info("Loading valid dataset...")
    valid_dataset = build_dataset(args, val_clients)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid.batch_size,
        shuffle=False,
        num_workers=args.valid.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, valid_dataset, train_dataloader, valid_dataloader


# ---------------------------------------------------------------------------
# Training step primitives
# ---------------------------------------------------------------------------
def prepare_batch(batch, diffusion, device, conditional):
    """
    Move a batch to the device, register the exogenous tensor on the diffusion
    and build the (optional) conditioning kwargs.

    Returns ``(x, model_kwargs)`` with ``x`` shaped ``[B, 1, L, 48]`` and
    ``model_kwargs`` containing the labels ``y`` only in the conditional setup.
    """
    x, exog, y = batch

    x = x.to(device)        # [B, L, 48]
    x = x.unsqueeze(1)      # [B, 1, L, 48]

    exog = exog.to(device)  # [B, L, n_exo (+1 if temp)]
    diffusion.set_exog(exog)

    model_kwargs = {}
    if conditional:
        model_kwargs["y"] = y.long().to(device)  # [B, K]

    return x, model_kwargs


def compute_losses(diffusion, model, x, model_kwargs):
    """
    Forward pass + diffusion loss for a random set of timesteps.
    """
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    return loss_dict["loss"].mean()


def backward_step(loss, optimizer):
    """
    Backward pass and optimizer update.
    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def log_train_step(writer, loss, optimizer, step):
    """
    Log per-step training scalars to TensorBoard.
    """
    writer.add_scalar("Train/loss", loss, step)
    writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], step)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(checkpoint_dir, step, model, ema, optimizer, scheduler, args, logger):
    """
    Persist the current training state so a run can be resumed later.
    """
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "args": args,
    }
    checkpoint_path = f"{checkpoint_dir}/{step:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def maybe_resume(args, model, ema, optimizer, scheduler, device, logger):
    """
    Optionally resume from a checkpoint when ``training.resume_checkpoint`` is
    set in the configuration. Returns the number of steps already done.
    """
    resume_path = args.training.get("resume_checkpoint", None)
    if not resume_path:
        return 0

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint {resume_path} does not exist.")

    logger.info(f"Resuming training from {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["opt"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    passed_steps = int(ckpt.get("step", 0))
    logger.info(f"Resumed at step {passed_steps}")
    return passed_steps


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_valid_loss(model, diffusion, valid_dataloader, device, conditional) -> float:
    """
    Average diffusion loss over the validation dataloader.
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in valid_dataloader:
        x, model_kwargs = prepare_batch(batch, diffusion, device, conditional)
        loss = compute_losses(diffusion, model, x, model_kwargs)
        total_loss += loss.item()
        n_batches += 1

    # NOTE: the original scripts do not restore model.train() here; we keep the
    # exact same behavior to preserve training dynamics (see PR description).
    return total_loss / max(n_batches, 1)


def build_generation_exog(args, train_dataloader, device, use_temperature):
    """
    Build the calendar (and optional temperature) exogenous tensor used to
    drive the sampling during validation.
    """
    start_date = pd.Timestamp(args.valid.gen_sample_start_date)
    end_date = start_date + timedelta(days=args.valid.gen_sample_days)
    extra = pd.date_range(start=start_date, end=end_date - timedelta(days=1), freq="D")
    exogene_array = np.vstack([
        extra.weekday.values     * (2 * np.pi / 6),
        extra.day.values         * (2 * np.pi / 31),
        extra.day_of_year.values * (2 * np.pi / 365),
        extra.month.values       * (2 * np.pi / 12),
    ])

    exog = torch.tensor(exogene_array, dtype=torch.float32, device=device).permute(1, 0)  # [L, 4]

    if use_temperature:
        for _, batch_exog, _ in train_dataloader:
            temp = batch_exog[0, :, -1:].to(device)  # [L, 1]
            break
        exog = torch.cat((exog, temp), -1)  # [L, 5]

    return exog


def run_validation(args, step, ema, diffusion, feature_extractor, valid_dataset,
                   train_dataloader, device, writer, conditional, use_temperature):
    """
    Generate samples with the EMA model, compute the evaluation metrics and log
    every figure / scalar to TensorBoard.
    """
    exog = build_generation_exog(args, train_dataloader, device, use_temperature)
    diffusion.set_exog(exog)  # set exogeneous information

    model_kwargs = {}
    if conditional:
        # Random multi-label conditioning for generation: y in {0,1}^{B x K}
        B = args.valid.gen_sample_number
        K = len(valid_dataset.bool_col_names)
        model_kwargs["y"] = torch.randint(0, 2, (B, K), device=device, dtype=torch.long)

    samples = diffusion.p_sample_loop(
        ema,
        (args.valid.gen_sample_number, 1, args.valid.gen_sample_days, valid_dataset.patch_length),
        clip_denoised=False,
        progress=False,
        model_kwargs=model_kwargs,
        device=device,
    )

    pred = samples.squeeze(1)        # [B, L, 48]
    pred = pred.flatten(start_dim=1)  # [B, 365 * 48 = 17520]

    feature_pred = feature_extractor(pred.unsqueeze(1)).detach().cpu().numpy()  # [B, n_features]
    pred = pred * args.data.value_scale  # [B, 365 * 48 = 17520]
    pred_np = pred.detach().cpu().numpy()  # [B, 365 * 48 = 17520]

    true_data = valid_dataset.data[:args.valid.gen_sample_number, :valid_dataset.nb_days * valid_dataset.patch_length] / args.data.value_scale
    feature_true = feature_extractor(true_data.unsqueeze(1).to(device)).detach().cpu().numpy()  # [B, n_features]
    true_data = true_data.cpu().numpy() * args.data.value_scale

    fid_metric = frechet_distance(feature_true, feature_pred)  # both input are [B, n_features]
    dict_metrics = get_all_metrics(np.expand_dims(true_data, axis=1), np.expand_dims(pred_np, axis=1))

    gen_data_z_norm = (pred_np - pred_np.mean(axis=1, keepdims=True)) / (pred_np.std(axis=1, keepdims=True) + 1e-6)
    gen_data = pred_np.reshape(pred_np.shape[0], -1)
    gen_data_z_norm = gen_data_z_norm.reshape(pred_np.shape[0], -1)
    print("Shape of gen data : ", gen_data.shape)

    print("Shape of True data : ", true_data.shape)

    true_data = true_data.reshape(true_data.shape[0], -1)

    print("Shape of enc original data : ", true_data.shape)

    gen_data_df = TimeSeriesData.array_to_df(
        gen_data, min_time=datetime.datetime.strptime(valid_dataset.user_start_date, "%d/%m/%Y")
    )
    true_data_df = TimeSeriesData.array_to_df(
        true_data, min_time=datetime.datetime.strptime(valid_dataset.user_start_date, "%d/%m/%Y")
    )
    tsd = TimeSeriesData({"real": true_data_df, "fake": gen_data_df})

    writer.add_scalar("valid/fid", fid_metric, global_step=step)

    for key, val in dict_metrics.items():
        writer.add_scalar(f"valid/{key}", val, global_step=step)

    fig = tsd.graph_tsne()
    writer.add_figure("stats/tsne", fig, global_step=step)
    plt.close(fig)

    fig = tsd.graph_mean_distribution()
    writer.add_figure("stats/mean_distribution", fig, global_step=step)
    plt.close(fig)

    fig = tsd.graph_quantile_distribution(q=0.2)
    writer.add_figure("stats/q0.2_distribution", fig, global_step=step)
    plt.close(fig)

    fig = tsd.graph_quantile_distribution(q=0.8)
    writer.add_figure("stats/q0.8_distribution", fig, global_step=step)
    plt.close(fig)

    fig = tsd.graph_daily_profile()
    writer.add_figure("profile/daily", fig, global_step=step)
    plt.close(fig)

    fig = tsd.graph_weekly_profile()
    writer.add_figure("profile/weekly", fig, global_step=step)
    plt.close(fig)

    fig = tsd.graph_monthly_profile()
    writer.add_figure("profile/monthly", fig, global_step=step)
    plt.close(fig)

    client_indices = [0, 1]
    plot_multiple_clients(pred, client_indices, step, writer)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting training, device: {device}")

    torch.manual_seed(args.training.global_seed)

    bool_col_names, num_classes, conditional, use_temperature = resolve_conditioning(args)
    logging.info(
        f"Conditioning: conditional={conditional} (num_classes={num_classes}, "
        f"labels={bool_col_names}), temperature={use_temperature}"
    )

    # Setup tensorboard / logging
    log_dir = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = f"{log_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(log_dir)
    logger.info(f"Experiment directory created at {log_dir}")

    # Model + diffusion
    model, ema, diffusion = build_model(args, num_classes, device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    logging.info("Data loading...")
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = build_dataloaders(args)
    logger.info(f"Dataset contains {len(train_dataset):,} curves ({args.data.data_path})")

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
    passed_steps = maybe_resume(args, model, ema, optimizer, scheduler, device, logger)

    # Step from which checkpoints are persisted (0 => always save).
    min_checkpoint_step = args.training.get("min_checkpoint_step", 0)

    # Variables for monitoring/logging purposes:
    epoch = 0
    train_losses = []
    valid_losses = []

    logger.info(f"Training for {args.training.max_steps} steps...")
    while passed_steps < args.training.max_steps:

        total_loss = 0.0
        epoch_passed_steps = 0

        logger.info(f"Beginning epoch {epoch}...")
        for batch in train_dataloader:

            # --- batch preparation + forward + losses ---
            x, model_kwargs = prepare_batch(batch, diffusion, device, conditional)
            loss = compute_losses(diffusion, model, x, model_kwargs)

            # --- backward pass + optimizer step ---
            backward_step(loss, optimizer)
            total_loss += loss.item()

            # Update EMA model
            update_ema(ema, model, decay=args.training.ema_decay)

            passed_steps += 1
            epoch_passed_steps += 1

            # --- logging ---
            log_train_step(writer, loss.item(), optimizer, passed_steps)
            scheduler.step()

            # --- validation + checkpoint ---
            if passed_steps % args.training.log_every == 0:
                logging.info(f"Train steps : {passed_steps}. Will save checkpoint!")

                logging.info("Validation loss ...")
                val_loss = evaluate_valid_loss(model, diffusion, valid_dataloader, device, conditional)
                valid_losses.append(val_loss)
                writer.add_scalar("Valid/loss", val_loss, passed_steps)

                if passed_steps >= min_checkpoint_step:
                    save_checkpoint(checkpoint_dir, passed_steps, model, ema,
                                    optimizer, scheduler, args, logger)

                run_validation(args, passed_steps, ema, diffusion, feature_extractor,
                               valid_dataset, train_dataloader, device, writer,
                               conditional, use_temperature)

        train_losses.append(total_loss / epoch_passed_steps)
        writer.add_scalar("Train/Epoch_total_loss", train_losses[-1], epoch)
        epoch += 1

    logger.info("Done!")


@hydra.main(version_base=None, config_path="../../configs", config_name="loadiff_with_conditioning.yaml")
def main(cfg: DictConfig) -> None:
    logging.info(cfg)
    train(cfg)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    main()