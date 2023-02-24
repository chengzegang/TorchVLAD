from __future__ import annotations

from typing import Tuple

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
from torch import nn  # type: ignore


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
