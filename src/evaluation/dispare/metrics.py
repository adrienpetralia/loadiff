import torch
import numpy as np
import scipy.linalg
from dispare.features import FeaturesExtractor, RawFeatures
from functools import partial
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.image.kid import maximum_mean_discrepancy

from abc import abstractmethod
from typing import Callable, Union, Tuple


class SharedData(Metric):
    """Data container that can be shared between metrics."""

    def __init__(self, features_extractor: FeaturesExtractor = RawFeatures(), **kwargs):
        """Initialize container.

        Args:
            features_extractor (FeaturesExtractor, optional): Features
                extractor to apply on data before saving them.
                Defaults to RawFeatures().
        """
        super().__init__(**kwargs)
        self.fe = features_extractor

        self.add_state("real_data", [], dist_reduce_fx="cat")
        self.add_state("fake_data", [], dist_reduce_fx="cat")

    def update(self, data: Tensor, real: bool, metadata=None):
        """Store the data for later use.

        Args:
            data (Tensor): time series or features
            real (bool): wether is comes from the real data or not
        """
        # Extract features
        data = self.fe(data, metadata)

        # Update stored data
        if real:
            self.real_data.append(data)
        else:
            self.fake_data.append(data)

    def compute(self):
        """Compute metric using current data."""
        raise NotImplementedError


class DistributionMetric:
    """Base class for metrics comparing two distributions"""

    def __init__(
        self,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        """Initialize metric

        Args:
            shared_data (SharedData | None, optional): Optional shared data
                object. If None, the metric will use its own one.
                Defaults to None.
            features_extractor (FeaturesExtractor | None, optional): In case
                no shared data is given, it is the features extractor that
                will be uses. Defaults to RawFeatures.
        """
        if shared_data is None:
            shared_data = SharedData(features_extractor)
        self.data = shared_data
        self.data_id = id(shared_data)

    def update(self, data: Tensor, real: bool):
        """Store the data for later use.

        Args:
            data (Tensor): time series or features
            real (bool): wether is comes from the real data or not
        """
        self.data.update(data, real)

    def reset(self):
        """Free stored data."""
        self.data.reset()

    @abstractmethod
    def compute(self):
        raise NotImplementedError(
            "You have to implement the `compute` method !"
        )  # nopep8 # noqa


def dist_fn(x, real_kde, fake_kde):
    """abs difference between densities"""
    return np.abs(
        np.exp(real_kde.score(np.array([[x]])))
        - np.exp(fake_kde.score(np.array([[x]])))
    )


def distance_distribution(
    real_data: Tensor,
    fake_data: Tensor,
    feature_fn: Callable,
    method="kde",
    bandwidth=0.05,
) -> Tensor:
    """Estimates distrubutions of given features in the two dataset. Then
    computes the distance between these distributions."""
    # Feature of interest
    real_f = feature_fn(real_data).cpu().numpy()
    fake_f = feature_fn(fake_data).cpu().numpy()

    if method == "kde":
        # Density estimation
        real_kde = KernelDensity(bandwidth=bandwidth)
        fake_kde = KernelDensity(bandwidth=bandwidth)
        real_kde.fit(real_f)
        fake_kde.fit(fake_f)

        # L1 distance estimation using scipy quad

        # # Support of the densities is inside [mini, maxi]
        mini = min(
            real_f.min() - real_kde.bandwidth_,
            fake_f.min() - fake_kde.bandwidth_,
        )
        maxi = max(
            real_f.max() - real_kde.bandwidth_,
            fake_f.max() - fake_kde.bandwidth_,
        )
        # # Integrate
        dist = torch.tensor(
            quad(
                partial(dist_fn, real_kde=real_kde, fake_kde=fake_kde),
                mini,
                maxi,
                limit=200,
            )[0]
        )
        return dist
    else:
        # Histograms
        h_real, bins = np.histogram(real_f, bins=100, density=True)
        h_fake, bins = np.histogram(fake_f, bins=bins, density=True)
        # Integrate
        dist = np.sum((bins[1:] - bins[:-1]) * np.abs(h_real - h_fake))
        dist = torch.tensor(dist)
        return dist


class DistDist(DistributionMetric):
    """Estimates the distribution of the feature of interest
    by KDE of histogram then computes L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        feature_fn,
        method="kde",
        bandwidth=0.05,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        super().__init__(shared_data, features_extractor)
        self.feature_fn = feature_fn
        self.bandwidth = bandwidth
        self.method = method
        assert method in [
            "kde",
            "hist",
        ], "Only two methods implemented for estimating densities: 'kde' and 'hist'"  # nopep8 # noqa

    def compute(self) -> Tensor:
        """Compute metric."""
        real_data = dim_zero_cat(self.data.real_data)
        fake_data = dim_zero_cat(self.data.fake_data)

        return distance_distribution(
            real_data, fake_data, self.feature_fn, self.method, self.bandwidth
        )


class DistMean(DistDist):
    """Estimates the mean distribution by KDE or histogram then computes
    L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        method="kde",
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        def feature_function(x: Tensor):
            return x.flatten(1).mean(1)[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


class DistQuantiles(DistDist):
    """Estimates the mean distribution by KDE or histogram then computes
    L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        q: float,
        method="kde",
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        self.q = q

        def feature_function(x: Tensor):
            return x.flatten(1).quantile(q, 1)[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


class DistStd(DistDist):
    """Estimates the std distribution by KDE or histogram then computes
    L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        method="kde",
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        def feature_function(x: Tensor):
            return x.flatten(1).std(1)[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


class DistAutocorrelation(DistDist):
    """Estimates the autocorrelation distribution by KDE or histogram then
    computes L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        method="kde",
        tau=1,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        def feature_function(x: Tensor):
            x = x.flatten(1)
            return torch.mean(x[:, tau:] * x[:, :-tau], 1)[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


class DistAbsAutocorrelation(DistDist):
    """Estimates the autocorrelation (absolute) distribution by KDE or
    histogram then computes L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        method="kde",
        tau=1,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        def feature_function(x: Tensor):
            x = x.flatten(1)
            return torch.mean(torch.abs(x[:, tau:] * x[:, :-tau]), 1)[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


class DistVariations(DistDist):
    """Estimates the avg variations distribution by KDE or histogram then
    computes L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        method="kde",
        tau=1,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        def feature_function(x: Tensor):
            x = x.flatten(1)
            return torch.mean(torch.abs(x[:, :-1] - x[:, 1:]), 1)[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


class DistNumMax(DistDist):
    """Estimates the number of local max distribution by KDE or histogram then
    computes L1 distance between the two densities."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        method="kde",
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        def feature_function(x: Tensor):
            x = x.flatten(1)
            num_maxi = (x[:, 1:-1] > x[:, 2:]) & (x[:, 1:-1] > x[:, :-2])
            num_maxi = num_maxi.to(torch.float32).mean(dim=1)
            return num_maxi[:, None]

        super().__init__(
            feature_fn=feature_function,
            method=method,
            shared_data=shared_data,
            features_extractor=features_extractor,
        )


def gaussian_kernel(f1: Tensor, f2: Tensor, sigma=1.0):
    raise NotImplementedError("Coming soon")


class MMD(DistributionMetric):
    """Maximum Mean Discrepancy.

    Based on the implementation of the Kernel Inception Distance in
    torchmetrics. (see torchmetrics.image.kid)
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        kernel,
        subsets: int = 100,
        subset_size: int = 1000,
        return_std=False,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ) -> None:
        """Initialize internal Module state, shared by both nn.Module and
        ScriptModule.

        Args:
            kernel (function): a semite definite kernel function that maps
                batch of features X, Y to the Gram matrix (k(x_i,y_j)).
            subsets (int): Number of subsets sampled to compute the metric.
            subset_size (int): Number of sampled element per subset.
            return_std (bool, optional): Wether to return the std of the
                metric beween subsets.
        """
        super().__init__(shared_data=shared_data, features_extractor=features_extractor)
        self.kernel = kernel
        self.subsets = subsets
        self.subset_size = subset_size
        self.return_std = return_std

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute metric."""
        real_data = dim_zero_cat(self.data.real_data).flatten(1)
        fake_data = dim_zero_cat(self.data.fake_data).flatten(1)

        n_samples_real = real_data.shape[0]
        if n_samples_real < self.subset_size:
            raise ValueError(
                "Argument `subset_size` should be smaller than the number of samples"
            )  # nopep8 # noqa
        n_samples_fake = fake_data.shape[0]
        if n_samples_fake < self.subset_size:
            raise ValueError(
                "Argument `subset_size` should be smaller than the number of samples"
            )  # nopep8 # noqa

        kid_scores_ = []
        for _ in range(self.subsets):
            perm = torch.randperm(n_samples_real)
            f_real = real_data[perm[: self.subset_size]]
            perm = torch.randperm(n_samples_fake)
            f_fake = fake_data[perm[: self.subset_size]]

            k_xx = self.kernel(f_real, f_real)
            k_xy = self.kernel(f_real, f_fake)
            k_yy = self.kernel(f_fake, f_fake)
            o = maximum_mean_discrepancy(k_xx, k_xy, k_yy)
            kid_scores_.append(o)

        kid_scores = torch.stack(kid_scores_)
        if self.return_std:
            return kid_scores.mean(), kid_scores.std(unbiased=False)
        else:
            return kid_scores.mean()


def __crps(x, real_data, int_cdf_2, int_cdf_minus_1_2):
    """Compute CRPS for one element of the data.

    Args:
        x (Tensor): One data point in the fake data set
        real_data: The real data set, sorted along batch dim
        int_cdf_2: Integral of the cdf^2 between real data point
        int_cdf_minus_1_2: Idem with cdf - 1

    Returns:
        CRPS(x)
    """
    # real_data = (N, T, F) or (N, T)
    # x = (T, F) or (T)
    mask = (real_data <= x[None, :]).to(torch.float32)
    crps_x = (int_cdf_2 * mask[1:] + int_cdf_minus_1_2 * (1 - mask[:-1])).sum(0)

    # There is missing the term \int_xi^x F(z)^2 + \int_x^xi+1 (F(z)-1)^2
    pos = torch.arange(0, mask.size(0), dtype=torch.float32)
    pos = 0.1 * pos / mask.size(0)
    if len(real_data.shape) == 3:
        pos = torch.unsqueeze(pos, -1)
    idx = torch.argmax(mask + pos[:, None], dim=0)
    idx = torch.where(mask[0] < 0.1, -1, idx)
    # Very small x
    crps_x = torch.where(idx == -1, crps_x + real_data[0] - x, crps_x)
    # Very big x
    crps_x = torch.where(idx == mask.size(0) - 1, crps_x + x - real_data[-1], crps_x)
    # Intermediate x
    mid_mask = (idx != -1) & (idx != mask.size(0) - 1)
    mid_idx = torch.where(mid_mask, idx, 0)
    xi = real_data[[mid_idx]].diag()
    xip1 = real_data[[mid_idx + 1]].diag()
    ci = ((mid_idx + 1) / real_data.size(0)) ** 2
    cip1 = ((mid_idx + 2) / real_data.size(0)) ** 2
    crps_x = torch.where(mid_mask, crps_x + ci * (x - xi) + cip1 * (xip1 - x), crps_x)
    return crps_x


def crps(real_data, fake_data):
    """Compute average CRPS of the fake data given real data."""
    real_data = torch.sort(real_data, dim=0)[0]

    # Flatten data if needed
    if len(real_data.shape) > 1:
        real_data = torch.flatten(real_data, 1, -1)
    if len(fake_data.shape) > 1:
        fake_data = torch.flatten(fake_data, 1, -1)

    # CDF is approximated by F(z) = 1/S \sum_i 1[x_i <= z]
    cdf = torch.linspace(1 / real_data.size(0), 1, real_data.size(0))
    if len(real_data.shape) == 3:
        cdf = cdf.tile((real_data.size(1), 1, 1))
        cdf = cdf.transpose(0, 1).transpose(0, 2)
        cdf = cdf.repeat((1, 1, 2))
    else:
        cdf = cdf.tile((real_data.size(1), 1)).T

    # \int_xi^xi+1 F(z)^2
    int_cdf_2 = (cdf[:-1] ** 2) * (real_data[1:] - real_data[:-1])
    # \int_xi^xi+1 (F(z)-1)^2
    int_cdf_minus_1_2 = ((cdf[:-1] - 1) ** 2) * (real_data[1:] - real_data[:-1])

    # Map __crps function that works for one value over the fake dataset
    crps_func = partial(
        __crps,
        real_data=real_data,
        int_cdf_2=int_cdf_2,
        int_cdf_minus_1_2=int_cdf_minus_1_2,
    )
    crps = torch.vmap(crps_func, chunk_size=32)(fake_data)
    return crps.mean()


class CRPS(DistributionMetric):
    """Continuous Ranked Probability Score"""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    @torch.no_grad()
    def compute(self):
        """Compute metric."""
        real_data = dim_zero_cat(self.data.real_data)
        fake_data = dim_zero_cat(self.data.fake_data)

        return crps(real_data, fake_data)


def frechet_distance(real_data: Tensor, fake_data: Tensor) -> Tensor:
    """Fréchet Distance between two samples of multidimentional Gaussian
    distributions."""
    # Flatten data if needed
    real_data = real_data.flatten(1)
    fake_data = fake_data.flatten(1)

    # Distance between mean values
    real_mu = real_data.mean(0)
    fake_mu = fake_data.mean(0)
    diff = real_mu - fake_mu
    dist_mu = torch.sum(diff * diff).detach().cpu().numpy()

    # Distance between variance co-variance matrices
    real_cov = real_data.T.cov()
    fake_cov = fake_data.T.cov()
    prod = real_cov.mm(fake_cov).cpu().numpy()
    prod, _ = scipy.linalg.sqrtm(prod, disp=False)
    dist_cov = np.trace(real_cov.cpu().numpy() + fake_cov.cpu().numpy() - 2 * prod)
    dist_cov = np.absolute(dist_cov)

    # Fréchet distance
    fd = dist_mu + dist_cov
    return torch.tensor(fd)


class FID(DistributionMetric):
    """Fréchet Inception Distance"""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    @torch.no_grad()
    def compute(self):
        """Compute metric."""
        real_data = dim_zero_cat(self.data.real_data)
        fake_data = dim_zero_cat(self.data.fake_data)

        return frechet_distance(real_data, fake_data)


class Authenticity(DistributionMetric):
    """Authenticity as described in https://arxiv.org/pdf/2102.08921.pdf"""

    def compute(self):
        """Compute metric"""
        real_data = dim_zero_cat(self.data.real_data).flatten(1)
        fake_data = dim_zero_cat(self.data.fake_data).flatten(1)

        # WARNING : could explode memory
        d_fake_real = torch.cdist(
            fake_data.unsqueeze(0), real_data.unsqueeze(0)
        ).squeeze()
        d_fake_real, amin = torch.min(d_fake_real, dim=-1)
        d_real_real = torch.cdist(
            real_data.unsqueeze(0), real_data.unsqueeze(0)
        ).squeeze()
        d_real_real += torch.eye(real_data.size(0)) * torch.max(d_real_real)
        d_real_real = torch.min(d_real_real, dim=-1)[0]
        return torch.mean((d_fake_real > d_real_real[amin]).to(torch.float32))


class AlphaPrecision(DistributionMetric):
    """alpha-Precision as described in https://arxiv.org/pdf/2102.08921.pdf"""

    def __init__(
        self,
        occ,
        shared_data: SharedData | None = None,
        features_extractor: FeaturesExtractor = RawFeatures(),
    ):
        """Initialize metric.

        Args:
            occ: One Class Classifier
        """
        super().__init__(shared_data=shared_data, features_extractor=features_extractor)
        self.occ = occ

    @torch.no_grad()
    def compute(self):
        """Compute metric"""
        real_data = dim_zero_cat(self.data.real_data)
        fake_data = dim_zero_cat(self.data.fake_data)

        if len(real_data.shape) < 3 or real_data.size(-1) != self.occ.in_channels:
            real_data = real_data.flatten(1)
            real_data = torch.split(real_data, self.occ.in_channels, dim=1)
            real_data = torch.stack(real_data, dim=1)
        if len(fake_data.shape) < 3 or fake_data.size(-1) != self.occ.in_channels:
            fake_data = fake_data.flatten(1)
            fake_data = torch.split(fake_data, self.occ.in_channels, dim=1)
            fake_data = torch.stack(fake_data, dim=1)

        # if not self.occ.fitted:
        real_data = real_data.to(self.occ.device)
        fake_data = fake_data.to(self.occ.device)
        self.occ.fit(real_data)

        return self.occ.alpha_precision(real_data, fake_data)

print("test")