"""Conditioning contracts shared by training and checkpoint-compatible inference."""
from typing import Sequence

import torch


def conditioning_class_count(label_names: Sequence[str], multilabels: bool) -> int:
    if not label_names:
        return 0
    if multilabels:
        return len(label_names)
    if len(label_names) != 1:
        raise ValueError("Single-label conditioning requires exactly one binary appliance column.")
    return 2


def complete_unknown_labels(
    labels: torch.Tensor,
    trained_unknown_slots: Sequence[bool],
    *,
    policy: str = "auto",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete unsupported unknown states using a reproducible uniform binary prior.

    ``auto`` retains trained unknown states and samples untrained ones. ``sample``
    samples every unknown state; ``require_trained`` rejects untrained unknowns.
    Returns the realized labels and a mask of the entries that were sampled.
    A local generator keeps fully specified inference's diffusion RNG unchanged.
    """
    if policy not in {"auto", "sample", "require_trained"}:
        raise ValueError("unknown_label_policy must be auto, sample, or require_trained.")
    if labels.ndim != 2 or len(trained_unknown_slots) != labels.shape[1]:
        raise ValueError("Unknown-state support must match the label matrix's columns.")
    if not torch.isfinite(labels).all() or not ((labels == -1) | (labels == 0) | (labels == 1)).all():
        raise ValueError("Appliance labels must be -1 (unknown), 0 (absent), or 1 (present).")
    trained = torch.tensor(trained_unknown_slots, dtype=torch.bool, device=labels.device)
    unknown = labels == -1
    sampled = unknown if policy == "sample" else unknown & ~trained.unsqueeze(0)
    if sampled.any() and policy == "require_trained":
        raise ValueError(
            "The checkpoint does not record training for these unknown appliance states. "
            "Provide all labels or use unknown_label_policy=auto to sample them."
        )
    if not sampled.any():
        return labels, sampled
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.randint(0, 2, labels.shape, generator=generator).to(labels.device)
    return torch.where(sampled, draws, labels), sampled
