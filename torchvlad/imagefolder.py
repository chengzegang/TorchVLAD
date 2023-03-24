from __future__ import annotations

import os
from typing import Callable

import torch
from PIL import Image, ImageFile
from torch import nn  # type: ignore
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor  # type: ignore

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        tensor: torch.Tensor = pil_to_tensor(img)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def get_img(self, idx: int) -> Image.Image:
        return Image.open(self.imgs[idx])

    def get_path(self, idx: int) -> str:
        return self.imgs[idx]
