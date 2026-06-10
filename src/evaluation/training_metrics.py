import numpy as np
from scipy.stats import skew, kurtosis, gaussian_kde


def frechet_distance(real: np.ndarray, generated: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet Inception Distance (FID) between two populations of embeddings.
    
    Args:
        z_real: (N_r, D) real-sample embeddings.
        z_gen : (N_g, D) generated-sample embeddings.
        eps   : jitter added to covariances for numerical stability.
        
    Returns:
        fid (float)
    """
    z_real = np.asarray(real, dtype=np.float64)
    z_gen  = np.asarray(generated,  dtype=np.float64)

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


def _sym_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """
    Symmetric matrix square root using eigen-decomposition.
    Clips small negatives in eigenvalues to zero.
    """
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, 0.0, None)
    sqrt_evals = np.sqrt(evals)
    return (evecs * sqrt_evals) @ evecs.T


def marginal_distribution_difference(real, generated):
    """
    Calculates the Marginal Distribution Difference (MDD) between real and generated data.
    
    NB!
    - The original code from the TSBBench paper uses histogram differences, which may be sensitive to the choice of bins. 
      To improve this, we consider using kernel density estimation (KDE) for a smoother and more robust comparison.
    """
    real_values = real.reshape(-1)
    generated_values = generated.reshape(-1)
    
    real_kde = gaussian_kde(real_values)
    gen_kde = gaussian_kde(generated_values)
    
    x = np.linspace(min(real_values.min(), generated_values.min()), 
                    max(real_values.max(), generated_values.max()), 100)
    
    mdd = np.mean(np.abs(real_kde(x) - gen_kde(x)))
    return mdd


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

def skewness_difference(real, generated):
    """Calculates the Skewness Difference (SD) between real and generated data."""
    real_skew = skew(real.reshape(-1))
    generated_skew = skew(generated.reshape(-1))
    
    sd = np.abs(real_skew - generated_skew)
    return sd

def kurtosis_difference(real, generated):
    """Calculates the Kurtosis Difference (KD) between real and generated data."""
    real_kurt = kurtosis(real.reshape(-1))
    generated_kurt = kurtosis(generated.reshape(-1))
    
    kd = np.abs(real_kurt - generated_kurt)
    return kd


def get_all_metrics(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Compute all distributional metrics between real and generated time series.
    Expects arrays shaped like (B, C, L) or (B, 1, L).

    Returns:
        {
            "mdd": Marginal Distribution Difference (float),
            "acd": Auto-Correlation Difference (float),
            "sd":  Skewness Difference (float),
            "kd":  Kurtosis Difference (float),
        }
    """
    return {
        "mdd": float(marginal_distribution_difference(real, generated)),
        "acd": float(auto_correlation_difference(real, generated)),
        "sd":  float(skewness_difference(real, generated)),
        "kd":  float(kurtosis_difference(real, generated)),
    }


if __name__ == '__main__':
    # Example usage:
    # real_data and generated_data should be numpy arrays of shape (batch_size, 1, length)
    real_data = np.random.normal(0, 1, (10, 1, 100))  # example real data
    generated_data = np.random.normal(0, 1, (10, 1, 100))  # example generated data

    print("Marginal Distribution Difference:", marginal_distribution_difference(real_data, generated_data))
    print("Auto-Correlation Difference:", auto_correlation_difference(real_data, generated_data))
    print("Skewness Difference:", skewness_difference(real_data, generated_data))
    print("Kurtosis Difference:", kurtosis_difference(real_data, generated_data))