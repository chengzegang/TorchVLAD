from typing import Tuple

import torch
from pykeops.torch import LazyTensor  # type: ignore
from torch import nn  # type: ignore


class KMeans(nn.Module):
    def __init__(self, k: int, n_iters: int = 10) -> None:
        super().__init__()
        self.k = k
        self.n_iters = n_iters
        self.__initialized = False
        self._centroids = nn.Parameter(torch.zeros(k, 1), requires_grad=False)
        self._n_points = nn.Parameter(torch.zeros(k, 1), requires_grad=False)

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids / self._n_points.clamp_min(1.0)

    def forward(
        self, x: torch.Tensor, k: int | None = None, n_iters: int | None = None
    ) -> Tuple[torch.Tensor, ...]:
        clusters, centroids, n_points = kmeans(
            x,
            self.k if k is None else k,
            self.n_iters if n_iters is None else n_iters,
            self.centroids if self.__initialized else None,
        )
        x = x.flatten(end_dim=-2)
        if not self.__initialized:
            self._centroids.data = self._centroids.expand(-1, x.shape[-1])
            self.__initialized = True

        if self.training:
            self._centroids.data = self._centroids.scatter_add(
                0, clusters.view(-1, 1).repeat(1, x.shape[-1]), x.flatten(end_dim=-2)
            )
            self._n_points.data += (
                torch.bincount(clusters.view(-1), minlength=self.k)
                .type_as(self._n_points.data)
                .view(-1, 1)
            )
        return clusters, centroids, n_points


def kmeans(
    x: torch.Tensor,
    K: int = 10,
    niters: int = 10,
    centroids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, ...]:
    """Implements Lloyd's algorithm for the Euclidean metric."""
    N, D = x.shape  # Number of samples, dimension of the ambient space
    centroids = x[:K, :].clone() if centroids is None else centroids
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
