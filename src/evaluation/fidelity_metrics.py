import numpy as np
import torch

from typing import Callable
from torch import Tensor
from scipy.stats import skew, kurtosis, gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad

from .features_extractor import ROCKET

from .utils import _kde_pdf_1d, _sym_matrix_sqrt


def frechet_distance(
        real_data: np.ndarray, 
        fake_data: np.ndarray, 
        eps: float = 1e-6
    ) -> float:
    """
    Compute Fréchet Inception Distance (FID) between two populations of embeddings.
    
    Args:
        z_real: (N_r, D) real-sample embeddings.
        z_gen : (N_g, D) generated-sample embeddings.
        eps   : jitter added to covariances for numerical stability.
        
    Returns:
        fid (float)
    """
    z_real = np.asarray(real_data, dtype=np.float64)
    z_gen  = np.asarray(fake_data,  dtype=np.float64)

    if z_real.ndim != 2 or z_gen.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (num_samples, feature_dim).")
    if z_real.shape[1] != z_gen.shape[1]:
        raise ValueError("Feature dimensions must match: got "
                         f"{z_real.shape[1]} and {z_gen.shape[1]}.")

    # Means
    mu_r = z_real.mean(axis=0)
    mu_g = z_gen.mean(axis=0)

    # Covariances (rowvar=False => columns are features)
    cov_r = np.cov(z_real, rowvar=False, ddof=1)
    cov_g = np.cov(z_gen,  rowvar=False, ddof=1)

    # Jitter for numerical stability
    d = z_real.shape[1]
    cov_r = cov_r + eps * np.eye(d)
    cov_g = cov_g + eps * np.eye(d)

    # ||mu_r - mu_g||^2
    mean_diff = mu_r - mu_g
    m_dist2 = mean_diff.dot(mean_diff)

    # Trace(Σ_r + Σ_g - 2 * (Σ_r Σ_g)^{1/2})
    # Use the identity: Tr((Σ_r Σ_g)^{1/2}) = sum_i sqrt(eig_i( Σ_r^{1/2} Σ_g Σ_r^{1/2} ))
    sr_sqrt = _sym_matrix_sqrt(cov_r)
    middle  = sr_sqrt @ cov_g @ sr_sqrt

    # Eigenvalues of symmetric matrix; clip tiny negatives from numerical noise
    evals = np.linalg.eigvalsh(middle)
    evals = np.clip(evals, 0.0, None)
    trace_sqrt = np.sum(np.sqrt(evals))

    fid = m_dist2 + np.trace(cov_r) + np.trace(cov_g) - 2.0 * trace_sqrt
    # FID is non-negative by definition; clamp tiny negatives from floating-point
    # cancellation (near-identical distributions / ill-conditioned covariances).
    return float(max(0.0, np.real_if_close(fid).item()))


def skewness_difference(
        real_data: np.array,
        fake_data: np.array
    ) -> float:
    """Calculates the Skewness Difference (SD) between real and generated data."""
    real_skew = skew(real_data.reshape(-1))
    generated_skew = skew(fake_data.reshape(-1))
    
    sd = np.abs(real_skew - generated_skew)
    return sd


def kurtosis_difference(
        real_data: np.array,
        fake_data: np.array
    ) -> float:
    """Calculates the Kurtosis Difference (KD) between real and generated data."""
    real_kurt = kurtosis(real_data.reshape(-1))
    generated_kurt = kurtosis(fake_data.reshape(-1))
    
    kd = np.abs(real_kurt - generated_kurt)
    return kd


def marginal_distribution_difference_old(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    method: str = "kde",
    bandwidth: float = 0.05,
    bins: int = 100,
    integration_margin: float | None = None,
    quad_limit: int = 200,          # kept for backward-compatibility (unused in grid mode)
    grid_size: int = 2048,          # NEW: grid resolution for KDE integration
    robust_bounds: bool = True,     # NEW: avoid min/max outlier blow-up
    q_low: float = 0.001,           # NEW: lower quantile for robust bounds
    q_high: float = 0.999,          # NEW: upper quantile for robust bounds
    scale_aware_bandwidth: bool = True,  # NEW: adapt bandwidth to data scale if user leaves default
    min_bandwidth: float = 1e-6,    # NEW: numerical floor
) -> float:
    """
    Estimate distributions of a scalar feature in two datasets, then compute
    the L1 distance between the distributions:

        dist = ∫ |p_real(x) - p_fake(x)| dx

    - If method == "kde": Gaussian KDE (sklearn) + *grid* numerical integration (trapz)
      (more stable than quad for peaky/multimodal KDE differences)
    - Else: histogram approximation of the same integral

    Args:
        real_data, fake_data: arrays of any shape; flattened to 1D scalars.
        bandwidth: KDE bandwidth. If scale_aware_bandwidth=True and bandwidth==0.05 (default),
                   bandwidth is recalibrated using a Silverman-like rule on pooled data.
        integration_margin: extends integration bounds. If None, uses 3 * bandwidth.
        grid_size: number of grid points for trapz integration.
        robust_bounds: if True, integrate over [q_low, q_high] pooled quantiles rather than [min, max]
                       to reduce outlier sensitivity.
    Returns:
        dist (float)
    """
    # Compute scalar feature values
    real_f = np.asarray(real_data, dtype=np.float64).reshape(-1)
    fake_f = np.asarray(fake_data, dtype=np.float64).reshape(-1)

    # Filter non-finite values
    real_f = real_f[np.isfinite(real_f)]
    fake_f = fake_f[np.isfinite(fake_f)]
    if real_f.size == 0 or fake_f.size == 0:
        return float("nan")

    if method == "kde":
        # Optionally make bandwidth scale-aware when the user kept the default
        bw = float(bandwidth)
        if scale_aware_bandwidth and np.isclose(bw, 0.05):
            pooled = np.concatenate([real_f, fake_f], axis=0)
            n = pooled.size
            std = float(np.std(pooled))
            if n > 1 and np.isfinite(std) and std > 0:
                # Silverman-like (good default for 1D)
                bw = 1.06 * std * (n ** (-1.0 / 5.0))
        bw = max(bw, min_bandwidth)

        # Fit KDEs
        real_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(real_f[:, None])
        fake_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(fake_f[:, None])

        # Choose integration bounds
        pooled = np.concatenate([real_f, fake_f], axis=0)
        if robust_bounds:
            lo = float(np.quantile(pooled, q_low))
            hi = float(np.quantile(pooled, q_high))
        else:
            lo = float(pooled.min())
            hi = float(pooled.max())

        if integration_margin is None:
            integration_margin = 3.0 * bw

        lo -= float(integration_margin)
        hi += float(integration_margin)

        # Guard against degenerate intervals
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("nan")

        # Grid integration (stable replacement for quad)
        xs = np.linspace(lo, hi, int(grid_size))[:, None]
        pr = np.exp(real_kde.score_samples(xs))
        pf = np.exp(fake_kde.score_samples(xs))
        dist_val = float(np.trapezoid(np.abs(pr - pf), xs[:, 0]))
        return dist_val

    # Histogram branch
    lo = min(real_f.min(), fake_f.min())
    hi = max(real_f.max(), fake_f.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    bin_edges = np.linspace(lo, hi, int(bins) + 1)
    h_real, _ = np.histogram(real_f, bins=bin_edges, density=True)
    h_fake, _ = np.histogram(fake_f, bins=bin_edges, density=True)

    dx = bin_edges[1:] - bin_edges[:-1]
    dist_val = float(np.sum(dx * np.abs(h_real - h_fake)))
    return dist_val


def marginal_distribution_difference( 
    real_data: np.ndarray,
    fake_data: np.ndarray,
    method: str = "kde",
    bandwidth: float = 0.05,
    bins: int = 100,
    integration_margin: float | None = None,
    quad_limit: int = 200,          
    grid_size: int = 2048,          
    robust_bounds: bool = True,     
    q_low: float = 0.001,           
    q_high: float = 0.999,          
    scale_aware_bandwidth: bool = True, 
    min_bandwidth: float = 1e-6,    
    max_samples: int = 20000,
    seed: int = 0
) -> float:
    """
    Optimized L1 distance calculation. Uses subsampling for large datasets
    to ensure KDE remains fast.
    """
    # Compute scalar feature values
    real_f = np.asarray(real_data, dtype=np.float64).reshape(-1)
    fake_f = np.asarray(fake_data, dtype=np.float64).reshape(-1)

    # Filter non-finite values
    real_f = real_f[np.isfinite(real_f)]
    fake_f = fake_f[np.isfinite(fake_f)]
    
    if real_f.size == 0 or fake_f.size == 0:
        return float("nan")

    if method == "kde":
        # --- OPTIMIZATION START: Subsampling ---
        # If data is too large, KDE is too slow. We subsample for estimation.
        # This does not affect accuracy significantly for N > 10k.
        rng = np.random.RandomState(seed)
        
        if len(real_f) > max_samples:
            real_f_fit = rng.choice(real_f, size=max_samples, replace=False)
        else:
            real_f_fit = real_f
            
        if len(fake_f) > max_samples:
            fake_f_fit = rng.choice(fake_f, size=max_samples, replace=False)
        else:
            fake_f_fit = fake_f
        # ---------------------------------------

        # Optionally make bandwidth scale-aware
        bw = float(bandwidth)
        if scale_aware_bandwidth and np.isclose(bw, 0.05):
            # Use the fitted (subsampled) data for bandwidth heuristic 
            # to keep it fast, or the full data if you prefer robustness (std is fast).
            # Here we use full data for std calculation as numpy is fast.
            pooled = np.concatenate([real_f, fake_f], axis=0)
            n = pooled.size
            std = float(np.std(pooled))
            if n > 1 and np.isfinite(std) and std > 0:
                bw = 1.06 * std * (n ** (-1.0 / 5.0))
        
        bw = max(bw, min_bandwidth)

        # Fit KDEs on the SUBSAMPLED data
        real_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(real_f_fit[:, None])
        fake_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(fake_f_fit[:, None])

        # Choose integration bounds (using full data for bounds is safe & correct)
        pooled = np.concatenate([real_f, fake_f], axis=0)
        if robust_bounds:
            lo = float(np.quantile(pooled, q_low))
            hi = float(np.quantile(pooled, q_high))
        else:
            lo = float(pooled.min())
            hi = float(pooled.max())

        if integration_margin is None:
            integration_margin = 3.0 * bw

        lo -= float(integration_margin)
        hi += float(integration_margin)

        # Guard against degenerate intervals
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("nan")

        # Grid integration
        xs = np.linspace(lo, hi, int(grid_size))[:, None]
        
        # Scoring is now fast because the tree inside real_kde is small (max_samples)
        pr = np.exp(real_kde.score_samples(xs))
        pf = np.exp(fake_kde.score_samples(xs))
        
        # Use trapezoid (numpy 2.0+) or trapz (older)
        if hasattr(np, "trapezoid"):
             dist_val = float(np.trapezoid(np.abs(pr - pf), xs[:, 0]))
        else:
             dist_val = float(np.trapz(np.abs(pr - pf), xs[:, 0]))
             
        return dist_val

    # Histogram branch (unchanged, this is already O(N) and fast)
    lo = min(real_f.min(), fake_f.min())
    hi = max(real_f.max(), fake_f.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    bin_edges = np.linspace(lo, hi, int(bins) + 1)
    h_real, _ = np.histogram(real_f, bins=bin_edges, density=True)
    h_fake, _ = np.histogram(fake_f, bins=bin_edges, density=True)

    dx = bin_edges[1:] - bin_edges[:-1]
    dist_val = float(np.sum(dx * np.abs(h_real - h_fake)))
    return dist_val


def auto_correlation_difference(real, generated, use_channel=0, eps=1e-6):
    """
    Calculates the Auto-Correlation Difference (ACD) between real and generated data.
    - Per-series z-score before ACF
    - Normalized ACF (acf[0] = 1)
    - Accepts (B, L), (B, 1, L), or (B, C, L); selects channel `use_channel` if needed
    """
    # ---- helpers ----
    def _to_np_2d(x):
        # torch -> numpy if needed
        if "torch" in str(type(x)):
            try:
                x = x.detach().cpu().numpy()
            except Exception:
                x = np.asarray(x)
        x = np.asarray(x)
        if x.ndim == 3:      # (B, C, L)
            if x.shape[1] <= use_channel:
                raise ValueError(f"use_channel={use_channel} out of range for shape {x.shape}")
            x = x[:, use_channel, :]  # -> (B, L)
        elif x.ndim == 2:    # (B, L)
            pass
        elif x.ndim == 1:    # (L,)
            x = x[None, :]
        else:
            raise ValueError(f"Unsupported shape {x.shape}; expected (B,L), (B,1,L), or (B,C,L).")
        return x.astype(np.float64, copy=False)

    def _acf_norm_1d(x1d):
        # z-score per series
        x1d = np.asarray(x1d, dtype=np.float64)
        m = np.mean(x1d)
        s = np.std(x1d)
        if not np.isfinite(s) or s < eps:
            # degenerate: return [1, 0, 0, ...]
            acf = np.zeros_like(x1d, dtype=np.float64)
            acf[0] = 1.0
            return acf
        x = (x1d - m) / (s + eps)
        corr = np.correlate(x, x, mode='full')   # length 2L-1
        acf = corr[corr.size // 2:]              # lags 0..L-1
        acf = acf / (acf[0] + eps)               # normalize so acf[0] = 1
        return acf

    # ---- prepare data & align lengths ----
    real2 = _to_np_2d(real)        # (B, L)
    gen2  = _to_np_2d(generated)   # (B, L)
    L = min(real2.shape[1], gen2.shape[1])
    if L < 2:
        return 0.0  # trivial/degenerate

    real2 = real2[:, :L]
    gen2  = gen2[:, :L]

    # ---- mean ACFs across batch ----
    real_acf = np.mean([_acf_norm_1d(r) for r in real2], axis=0)   # (L,)
    gen_acf  = np.mean([_acf_norm_1d(g) for g in gen2], axis=0)    # (L,)

    # ---- ACD ----
    acd = float(np.mean(np.abs(real_acf - gen_acf)))
    return acd


def compute_fidelity_metrics(
        real_data: np.ndarray, 
        synth_data: np.ndarray,
        real_data_train: np.ndarray | None = None,
        features_extractor: torch.nn.Module | None = None
    ) -> dict:
    """
    Compute all distributional metrics between real and generated time series.
    """
    if features_extractor is None:
        return {
            "mdd": float(marginal_distribution_difference(real_data, synth_data)),
            "acd": float(auto_correlation_difference(real_data, synth_data)),
            "sd":  float(skewness_difference(real_data, synth_data)),
            "kd":  float(kurtosis_difference(real_data, synth_data)),
        }
    else:
        feat_real_data, feat_synth_data = features_extractor.get_features(real_data, synth_data)

        return {
            "fid": float(frechet_distance(feat_real_data, feat_synth_data)),
            "mdd": float(marginal_distribution_difference(real_data, synth_data)),
            "acd": float(auto_correlation_difference(real_data, synth_data)),
            "sd":  float(skewness_difference(real_data, synth_data)),
            "kd":  float(kurtosis_difference(real_data, synth_data)),
        }

        


if __name__ == '__main__':
    # Example usage:
    # real_data and generated_data should be numpy arrays of shape (batch_size, length)
    real_data = np.random.normal(0, 1, (10, 1, 100))  # example real data
    synth_data = np.random.normal(0, 1, (10, 1, 100))  # example generated data

    res = compute_fidelity_metrics(
        real_data=real_data,
        synth_data=synth_data,
        features_extractor=ROCKET(real_data.shape[-1])
    )

    print(res)