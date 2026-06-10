import numpy as np
from privacyeval import z_norm_timeseries, evaluate_metrics


def compute_privacy_metrics(
    real_data_train: np.array,
    synthetic_data: np.array,
    real_data_test: np.array,
    metrics=["ims", "dcr", "nndr", "authenticity", "neighbors_privacy"],
):
    """
    Compute privacy-related metrics for synthetic time series using `privacyeval`.

    This function is a thin, opinionated wrapper around `privacyeval.evaluate_metrics`
    for time-series data. It returns a flat dictionary of scalar scores.

    Notes on returned scores
    ------------------------
    - ims: Intersection Matching Score (exact match share) in [0, 1]. Lower is better (ideally ~0).
    - dcr: Absolute gap between DCR(synthetic -> real_train) and DCR(holdout -> real_train).
           Lower is better (synthetic should not be closer to train than holdout is).
    - nndr: Absolute gap between NNDR(synthetic) and NNDR(holdout). Lower is better.
    - neighbors_privacy: Proportion based on k-NN re-identification for synthetic set.
                         Interpretation depends on the library's definition/implementation.
    - authenticity: Frequency synthetic is closer to real than real is to itself; acceptable often ~0.5.

    Parameters
    ----------
    real_data_train : np.array
        Real training data used as the reference population.
    synthetic_data : np.array
        Synthetic data to evaluate.
    real_data_test : np.array
        Real holdout/test data used as a baseline for comparisons (e.g., for DCR/NNDR).
    metrics : tuple
        Metrics to compute, forwarded to `privacyeval.evaluate_metrics`.

    Returns
    -------
    dict
        Flat dict with keys among: {"ims", "dcr", "nndr", "neighbors_privacy", "authenticity"}.

    Raises
    ------
    KeyError
        If `privacyeval` returns an unexpected structure or missing metric keys.
    """

    results_ts_raw = evaluate_metrics(
        real_data=real_data_train,
        synthetic_data=synthetic_data,
        holdout_data=real_data_test,
        data_type="timeseries",
        metrics=metrics,
    )

    def _get_raw(metric_name: str, split: str):
        return results_ts_raw[metric_name][split]
    
    res = {}

    if metrics != ["ims"]:
        z_norm_real_data_train = z_norm_timeseries(real_data_train, orientation="row")
        z_norm_synthetic_data = z_norm_timeseries(synthetic_data, orientation="row")
        z_norm_real_data_test= z_norm_timeseries(real_data_test, orientation="row")

        results_ts_znorm = evaluate_metrics(
            real_data=z_norm_real_data_train,
            synthetic_data=z_norm_synthetic_data,
            holdout_data=z_norm_real_data_test,
            data_type="timeseries",
            metrics=metrics,
        )

        def _get_znorm(metric_name: str, split: str):
            return results_ts_znorm[metric_name][split]
    
        if "ims" in metrics:
            res["ims"] = _get_raw("ims", "synthetic")

        if "dcr" in metrics:
            res["dcr"] = abs(_get_raw("dcr", "synthetic") - _get_raw("dcr", "holdout"))
            res["dcr_znorm"] = abs(_get_znorm("dcr", "synthetic") - _get_znorm("dcr", "holdout"))

        if "nndr" in metrics:
            res["nndr"] = abs(_get_raw("nndr", "synthetic") - _get_raw("nndr", "holdout"))
            res["nndr_znorm"] = abs(_get_znorm("nndr", "synthetic") - _get_znorm("nndr", "holdout"))

        if "neighbors_privacy" in metrics:
            res["neighbors_privacy"] = _get_raw("neighbors_privacy", "synthetic")
            res["neighbors_privacy_znorm"] = _get_znorm("neighbors_privacy", "synthetic")

        if "authenticity" in metrics:
            res["authenticity"] = _get_raw("authenticity", "synthetic")
            res["authenticity_znorm"] = _get_znorm("authenticity", "synthetic")
        
    else:
        res["ims"] = _get_raw("ims", "synthetic")

    return res


if __name__ == "__main__":
    # Example usage:
    # Real/synthetic data should be numpy arrays of shape (batch_size, length)
    real_data_train = np.random.normal(0, 1, (100, 100))
    real_data_test = np.random.normal(0, 1, (100, 100))
    synth_data = np.random.normal(0, 1, (100, 100))

    res = compute_privacy_metrics(
        real_data_train=real_data_train,
        real_data_test=real_data_test,
        synthetic_data=synth_data,
        metrics=("ims", "dcr", "nndr", "authenticity", "neighbors_privacy"),
    )

    print(res)
