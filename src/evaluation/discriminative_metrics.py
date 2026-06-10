from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


def _check_2d(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (B, L). Got shape={x.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 samples. Got B={x.shape[0]}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf values.")
    return x.astype(np.float64, copy=False)


def _stratified_split_indices(
    n_real: int,
    n_fake: int,
    test_size: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1). Got {test_size}.")

    # Ensure at least 1 sample of each class in test, and at least 1 in train.
    n_test_real = int(np.round(test_size * n_real))
    n_test_fake = int(np.round(test_size * n_fake))

    n_test_real = max(1, min(n_real - 1, n_test_real))
    n_test_fake = max(1, min(n_fake - 1, n_test_fake))

    real_idx = np.arange(n_real)
    fake_idx = np.arange(n_fake)

    test_real = rng.choice(real_idx, size=n_test_real, replace=False)
    test_fake = rng.choice(fake_idx, size=n_test_fake, replace=False)

    train_real = np.setdiff1d(real_idx, test_real, assume_unique=False)
    train_fake = np.setdiff1d(fake_idx, test_fake, assume_unique=False)

    return {
        "train_real": train_real,
        "test_real": test_real,
        "train_fake": train_fake,
        "test_fake": test_fake,
    }


def _standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mode: Optional[str],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardization options:
    - None: no normalization
    - "zscore_per_feature": z-score each feature dimension using TRAIN stats
    - "zscore_per_series": z-score each series independently (train/test separately)
    """
    if mode is None or mode == "none":
        return X_train, X_test

    if mode == "zscore_per_feature":
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True)
        sd = np.maximum(sd, eps)
        return (X_train - mu) / sd, (X_test - mu) / sd

    if mode == "zscore_per_series":
        def z_per_series(X: np.ndarray) -> np.ndarray:
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            sd = np.maximum(sd, eps)
            return (X - mu) / sd
        return z_per_series(X_train), z_per_series(X_test)

    raise ValueError(f"Unknown standardize mode: {mode!r}.")


def _predict_1nn_euclidean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    1-NN prediction with Euclidean distance using a stable and fast formula:
      ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    Uses chunking to limit memory peak.
    """
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dimensions do not match.")
    if X_train.shape[0] == 0:
        raise ValueError("Empty training set.")

    # Precompute norms
    train_norm2 = np.sum(X_train * X_train, axis=1)  # (N_train,)
    y_pred = np.empty(X_test.shape[0], dtype=y_train.dtype)

    for i in range(0, X_test.shape[0], chunk_size):
        Xc = X_test[i:i + chunk_size]  # (C, D)
        test_norm2 = np.sum(Xc * Xc, axis=1)  # (C,)

        # dist2 shape: (C, N_train)
        # Use matrix product for the cross term
        cross = Xc @ X_train.T  # (C, N_train)
        dist2 = test_norm2[:, None] + train_norm2[None, :] - 2.0 * cross

        nn_idx = np.argmin(dist2, axis=1)
        y_pred[i:i + Xc.shape[0]] = y_train[nn_idx]

    return y_pred


@dataclass
class DiscriminativeScore1NN:
    """
    Discriminative score : test accuracy of a post-hoc 1-NN
    classifier (Euclidean) separating real vs synthetic samples.

    Inputs:
      real:      np.ndarray (B_real, L)
      synthetic: np.ndarray (B_syn,  L)

    Output:
      mean/std accuracy over n_repeats stratified splits.
    """
    test_size: float = 0.2
    n_repeats: int = 10
    random_state: int = 0
    standardize: Optional[str] = None  # None | "zscore_per_feature" | "zscore_per_series"
    chunk_size: int = 1024

    def __call__(self, real: np.ndarray, synthetic: np.ndarray) -> Dict[str, Any]:
        real = _check_2d("real", real)
        fake = _check_2d("synthetic", synthetic)

        if real.shape[1] != fake.shape[1]:
            raise ValueError(f"Sequence length L must match. Got {real.shape[1]} vs {fake.shape[1]}.")

        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1. Got {self.n_repeats}.")

        rng = np.random.default_rng(self.random_state)
        accs = []

        for _ in range(self.n_repeats):
            idx = _stratified_split_indices(real.shape[0], fake.shape[0], self.test_size, rng)

            X_train = np.vstack([real[idx["train_real"]], fake[idx["train_fake"]]])
            y_train = np.concatenate([
                np.ones(len(idx["train_real"]), dtype=np.int64),
                np.zeros(len(idx["train_fake"]), dtype=np.int64),
            ])

            X_test = np.vstack([real[idx["test_real"]], fake[idx["test_fake"]]])
            y_test = np.concatenate([
                np.ones(len(idx["test_real"]), dtype=np.int64),
                np.zeros(len(idx["test_fake"]), dtype=np.int64),
            ])

            # Shuffle train/test sets (important for any downstream assumptions)
            tr_perm = rng.permutation(X_train.shape[0])
            te_perm = rng.permutation(X_test.shape[0])
            X_train, y_train = X_train[tr_perm], y_train[tr_perm]
            X_test, y_test = X_test[te_perm], y_test[te_perm]

            # Optional standardization
            X_train_s, X_test_s = _standardize(X_train, X_test, self.standardize)

            # 1-NN predict and accuracy
            y_pred = _predict_1nn_euclidean(X_train_s, y_train, X_test_s, chunk_size=self.chunk_size)
            acc = float(np.mean(y_pred == y_test))
            accs.append(acc)

        accs = np.asarray(accs, dtype=np.float64)
        return {
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "n_repeats": self.n_repeats,
            "test_size": self.test_size,
            "standardize": self.standardize,
        }

# functional API
def discriminative_score_1nn(
    real: np.ndarray,
    synthetic: np.ndarray,
    test_size: float = 0.2,
    n_repeats: int = 10,
    random_state: int = 0,
    standardize: Optional[str] = None,
    chunk_size: int = 1024,
) -> Dict[str, Any]:
    return DiscriminativeScore1NN(
        test_size=test_size,
        n_repeats=n_repeats,
        random_state=random_state,
        standardize=standardize,
        chunk_size=chunk_size,
    )(real, synthetic)


def compute_discriminative_metrics(
    real_data: np.ndarray,
    synth_data: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute discriminative metrics between real and synthetic data.

    This function is intentionally a thin wrapper to keep the public API stable.
    It currently returns the mean 1-NN discriminative accuracy under the key
    'discriminative_1nn'.

    Parameters
    ----------
    real_data:
        Real samples with shape (B, L).
    synth_data:
        Synthetic samples with shape (B, L).

    Returns
    -------
    dict
        Dictionary with:
        - 'discriminative_1nn': float
            Mean test accuracy of the 1-NN classifier across repeated splits.
    """
    res_1nn = discriminative_score_1nn(
        real=real_data,
        synthetic=synth_data,
        test_size=0.2,
    )

    return {"discriminative_1nn": res_1nn["accuracy_mean"]}


if __name__ == '__main__':
    # Example usage:
    # real_data and generated_data should be numpy arrays of shape (batch_size, length)
    real_data = np.random.normal(0, 1, (100, 100))  # example real data
    synth_data = np.random.normal(0, 1, (100, 100))  # example generated data

    res = compute_discriminative_metrics(
        real_data=real_data,
        synth_data=synth_data
    )

    print(res)