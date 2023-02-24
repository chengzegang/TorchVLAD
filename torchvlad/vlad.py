from __future__ import annotations

from typing import Tuple

import torch
from pykeops.torch import LazyTensor  # type: ignore
from torch import nn

from .kmeans import KMeans  # type: ignore

from .sift import SIFT


class VLAD(nn.Module):
    def __init__(
        self,
        num_clusters: int = 128,
        num_features: int = 128,
        patch_size: int = 32,
        angle_bins: int = 8,
        spatial_bins: int = 4,
        niter: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()
        self.sift = SIFT(num_features, patch_size, angle_bins, spatial_bins)
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.patch_size = patch_size
        self._populations = nn.Parameter(
            torch.zeros(num_clusters).float(), requires_grad=False
        )
        self._centroids = nn.Parameter(
            torch.zeros(num_clusters).float(), requires_grad=False
        )
        self.kmeans = KMeans(num_clusters, niter)

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids / self._populations.view(-1, 1)

    def init_clusters(self, x: torch.Tensor) -> None:
        lafs, resps, descs = self.sift(x)
        B, N, D = descs.shape
        clusters, self._centroids.data, n_points = self.kmeans(
            descs.flatten(end_dim=-2), self.num_clusters
        )
        self._populations.data += n_points

    def update_clusters(self, x: torch.Tensor) -> None:
        lafs, resps, descs = self.sift(x)
        B, N, D = descs.shape
        descs = descs.flatten(end_dim=-2)
        distances = ((descs.unsqueeze(-2) - self.centroids.unsqueeze(-3)) ** 2).sum(-1)
        clusters = distances.argmin(dim=-1).long().view(-1)
        self._populations.data += torch.bincount(
            clusters.view(-1), minlength=self.num_clusters
        )
        self._centroids.scatter_add_(
            0, clusters.view(-1, 1).repeat(1, D), descs.view(-1, D)
        )

    def residuals(self, x: torch.Tensor) -> torch.Tensor:
        lafs, resps, descs = self.sift(x)
        B, N, D = descs.shape
        distances = ((descs.unsqueeze(-2) - self.centroids.unsqueeze(-3)) ** 2).sum(-1)
        clusters = distances.argmin(dim=-1).long()
        desc_sums = torch.zeros(B, self.num_clusters, D).to(descs.device)
        desc_sums.scatter_add_(1, clusters.view(B, N, 1).repeat(1, 1, D), descs)
        pops = torch.stack(
            [torch.bincount(clusters[i], minlength=self.num_clusters) for i in range(B)]
        )

        center_sums = self.centroids * pops.unsqueeze(-1)
        residuals = center_sums - desc_sums
        if self.training:
            self._populations.data += pops.sum(0)
            self._centroids.data += desc_sums.sum(0)

        residuals = residuals / torch.linalg.matrix_norm(
            residuals, dim=(-1, -2), ord=2, keepdim=True
        )
        return residuals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residuals(x)
