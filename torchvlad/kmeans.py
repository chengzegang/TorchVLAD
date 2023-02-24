from __future__ import annotations

from typing import Tuple

import torch
from pykeops.torch import LazyTensor  # type: ignore
from torch import nn  # type: ignore


class KMeans(nn.Module):
    def __init__(self, k: int, niters: int = 10) -> None:
        super().__init__()
        self.k = k
        self.niters = niters

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return kmeans(x, self.k, self.niters)


def kmeans(x: torch.Tensor, K: int = 10, niters: int = 10) -> Tuple[torch.Tensor, ...]:
    """Implements Lloyd's algorithm for the Euclidean metric."""
    N, D = x.shape  # Number of samples, dimension of the ambient space
    centroids = x[:K, :].clone()  # Simplistic initialization for the centroids
    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(centroids.view(1, K, D))  # (1, K, D) centroids
    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(niters):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        clusters = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        centroids.zero_()
        centroids.scatter_add_(0, clusters[:, None].repeat(1, D), x)
        # Divide by the number of points per cluster:
        n_points = torch.bincount(clusters, minlength=K).type_as(centroids).view(K, 1)
        centroids /= n_points  # in-place division to compute the average
    return clusters, centroids, n_points.view(-1)
