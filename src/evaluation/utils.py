import numpy as np
import torch

import json
from pathlib import Path
from typing import Any

from sklearn.neighbors import KernelDensity

def _kde_pdf_1d(kde: KernelDensity, x: float) -> float:
    """
    Evaluate sklearn KernelDensity (which outputs log-density) at scalar x.
    Returns density (not log-density).
    """
    x2 = np.array([[x]], dtype=np.float64)  # shape (1, 1)
    logp = kde.score_samples(x2)[0]
    return float(np.exp(logp))

def _sym_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """
    Symmetric matrix square root using eigen-decomposition.
    Clips small negatives in eigenvalues to zero.
    """
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, 0.0, None)
    sqrt_evals = np.sqrt(evals)
    return (evecs * sqrt_evals) @ evecs


def to_jsonable(x: Any) -> Any:
    """Recursively convert objects (numpy/torch/Path) into JSON-serializable types."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)

    # Torch
    if torch.is_tensor(x):
        # tensor scalar -> python number; tensor array -> list
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()

    # Numpy
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x.tolist()
    if isinstance(x, np.generic):  # np.float32, np.int64, ...
        return x.item()

    # Python scalars / strings are already JSON-friendly
    return x