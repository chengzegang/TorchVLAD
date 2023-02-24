from __future__ import annotations

import os
import shutil
from typing import Callable, List, Tuple

import pandas as pd
import torch
from kornia.color import rgb_to_grayscale
from kornia.feature import (
    BlobHessian,
    LAFOrienter,
    ScaleSpaceDetector,
    SIFTDescriptor,
    extract_patches_from_pyramid,
)
from kornia.geometry import RANSAC, ConvQuadInterp3d, ScalePyramid
from PIL import Image, ImageFile
from pykeops.torch import LazyTensor  # type: ignore
from torch import nn  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize  # type: ignore
from torchvision.transforms.functional import pil_to_tensor  # type: ignore
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

__SHARED_VLAD: VLAD | None = None


class SIFT(nn.Module):
    def __init__(
        self,
        num_features: int = 128,
        patch_size: int = 32,
        angle_bins: int = 8,
        spatial_bins: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.descriptor = SIFTDescriptor(
            patch_size, angle_bins, spatial_bins, rootsift=True
        )
        self.detector = ScaleSpaceDetector(
            num_features=num_features,
            resp_module=BlobHessian(),
            nms_module=ConvQuadInterp3d(),
            scale_pyr_module=ScalePyramid(
                3, 1.6, min_size=patch_size, double_image=True
            ),
            ori_module=LAFOrienter(patch_size),
            mr_size=6.0,
        )
        self.sampler = RANSAC()

    def detect(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.detector.to(x.device)
        with torch.no_grad():
            lafs, resps = self.detector(x.contiguous())
            return lafs, resps

    def describe(self, x: torch.Tensor, lafs: torch.Tensor) -> torch.Tensor:
        self.descriptor.to(x.device)
        with torch.no_grad():
            patches = extract_patches_from_pyramid(x, lafs, self.patch_size)
            B, N, CH, H, W = patches.size()
            descs = self.descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
            return descs

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = rgb_to_grayscale(x).float()
        lafs, resps = self.detect(x)
        descs = self.describe(x, lafs)
        return lafs, resps, descs


class VLAD(nn.Module):
    def __init__(
        self,
        num_clusters: int = 128,
        num_features: int = 128,
        patch_size: int = 32,
        angle_bins: int = 8,
        spatial_bins: int = 4,
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

    def KMeans(
        self, x: torch.Tensor, K: int = 10, Niter: int = 10
    ) -> Tuple[torch.Tensor, ...]:
        """Implements Lloyd's algorithm for the Euclidean metric."""

        N, D = x.shape  # Number of samples, dimension of the ambient space

        centroids = x[:K, :].clone()  # Simplistic initialization for the centroids

        x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(centroids.view(1, K, D))  # (1, K, D) centroids

        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(Niter):
            # E step: assign points to the closest cluster -------------------------
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
            clusters = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            centroids.zero_()
            centroids.scatter_add_(0, clusters[:, None].repeat(1, D), x)

            # Divide by the number of points per cluster:
            n_points = (
                torch.bincount(clusters, minlength=K).type_as(centroids).view(K, 1)
            )
            centroids /= n_points  # in-place division to compute the average

        return clusters, centroids, n_points.view(-1)

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids / self._populations.view(-1, 1)

    def init_clusters(self, x: torch.Tensor) -> None:
        lafs, resps, descs = self.sift(x)
        B, N, D = descs.shape
        clusters, self._centroids.data, n_points = self.KMeans(
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


class ImageFolder(Dataset):
    def __init__(
        self, root: str, transform: Callable | nn.Module | None = None
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.imgs = []
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(("jpg", "png", "jpeg")):
                    img_path = os.path.join(dirpath, filename)
                    self.imgs.append(img_path)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.imgs[index])
        tensor = pil_to_tensor(img)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def get_img(self, idx: int) -> Image.Image:
        return Image.open(self.imgs[idx])

    def get_path(self, idx: int) -> str:
        return self.imgs[idx]


def train(
    imgdir: str,
    batch_size: int = 1,
    image_size: Tuple[int, int] | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
    verbose: bool = False,
    device: str = "cpu",
    **kwargs,
) -> VLAD:
    global __SHARED_VLAD
    if __SHARED_VLAD is None:
        __SHARED_VLAD = VLAD(**kwargs)

    transforms = Resize(image_size, antialias=True) if image_size is not None else None

    dataset = ImageFolder(imgdir, transforms)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    __SHARED_VLAD.to(device)
    __SHARED_VLAD.train()
    for idx, batch in enumerate(tqdm(dataloader, disable=not verbose)):
        batch = batch.to(device).float()
        if idx == 0:
            __SHARED_VLAD.init_clusters(batch)
        __SHARED_VLAD.update_clusters(batch)
    return __SHARED_VLAD


def index(
    index_img_dir: str,
    db_dir: str = "db",
    batch_size: int = 1,
    image_size: Tuple[int, int] | None = None,
    num_workers: int = 0,
    verbose: bool = False,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, str]:
    global __SHARED_VLAD
    if __SHARED_VLAD is None:
        raise ValueError("You must train the model first")

    os.makedirs(db_dir, exist_ok=True)

    records = []
    transforms = Resize(image_size, antialias=True) if image_size is not None else None

    dataset = ImageFolder(index_img_dir, transforms)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    __SHARED_VLAD.to(device)
    __SHARED_VLAD.eval()

    for idx, batch in enumerate(tqdm(dataloader, disable=not verbose)):
        with torch.no_grad():
            batch = batch.to(device).float()
            residuals = __SHARED_VLAD(batch)
        for i, res in enumerate(residuals.cpu().unbind(0)):
            num = idx * batch_size + i
            index_res_name = os.path.join(db_dir, f"index_res_{num}.pt")
            torch.save(res, index_res_name)
            records.append(dict(img_path=dataset.get_path(i), res_path=index_res_name))

    record_df = pd.DataFrame(records)
    record_df_path = os.path.join(db_dir, "index.csv")
    record_df.to_csv(record_df_path, index=False)
    return record_df, record_df_path


def test(
    query_dir: str,
    index_df: pd.DataFrame | str,
    cache_dir: str = "cache",
    clear: bool = False,
    n_retrivals: int = 10,
    batch_size: int = 1,
    image_size: Tuple[int, int] | None = None,
    num_workers: int = 0,
    verbose: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, pd.DataFrame]:
    global __SHARED_VLAD

    if __SHARED_VLAD is None:
        raise ValueError("You must train the model first")

    if isinstance(index_df, str):
        index_df = pd.read_csv(index_df)
    else:
        assert isinstance(
            index_df, pd.DataFrame
        ), "index_df must be a pandas DataFrame or a path to a csv file"

    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
        if not clear:
            raise ValueError("Cache directory is not empty")
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    transforms = Resize(image_size, antialias=True) if image_size is not None else None

    dataset = ImageFolder(query_dir, transforms)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    __SHARED_VLAD.to(device)
    __SHARED_VLAD.eval()

    total = len(dataloader)
    for idx, batch in enumerate(tqdm(dataloader, disable=not verbose)):
        with torch.no_grad():
            residuals = __SHARED_VLAD(batch.to(device).float())
        torch.save(residuals.cpu(), os.path.join(cache_dir, f"residuals_{idx}.pt"))

    index_chks = []
    for i, row in tqdm(index_df.iterrows(), total=len(index_df), disable=not verbose):
        index_res = torch.load(row["res_path"]).to(device)
        index_chks.append(index_res)
    index = torch.stack(index_chks)

    retrival_chks = []
    for j in tqdm(range(total), disable=not verbose):
        path = os.path.join(cache_dir, f"residuals_{j}.pt")
        residuals = torch.load(path).to(device)
        distances = torch.cdist(residuals.flatten(-2), index.flatten(-2), p=2)
        retr_vals, retr_indices = torch.topk(
            distances, k=n_retrivals, dim=-1, largest=False
        )
        retrival_chks.append(retr_indices.cpu())

    retrivals = torch.cat(retrival_chks, dim=0)
    query_paths = [dataset.get_path(i) for i in range(len(dataset))]
    results = []
    for i, retr_indices in enumerate(retrivals.unbind(0)):
        retr_paths = index_df["img_path"].iloc[retr_indices.numpy()].tolist()
        results.append(dict(query=query_paths[i], retrivals=retr_paths))
    result_df = pd.DataFrame(results)

    return retrivals, result_df
