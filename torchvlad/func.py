from __future__ import annotations

import os
import shutil
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize  # type: ignore
from tqdm.auto import tqdm

from .vlad import VLAD
from .imagefolder import ImageFolder


__SHARED_VLAD: VLAD | None = None


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
