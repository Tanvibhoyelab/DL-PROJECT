"""LEVIR-CD style dataset loader.

Expected layout (this is the official LEVIR-CD structure):

data/LEVIR-CD/
    train/A/*.png   train/B/*.png   train/label/*.png
    val/A/...       val/B/...       val/label/...
    test/A/...      test/B/...      test/label/...
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class DatasetNotFound(FileNotFoundError):
    pass


class LevirCDDataset(Dataset):
    """Paired before/after images with a binary change mask.

    ``augment`` applies flips, rotations, temporal swaps and mild photometric
    jitter. The jitter is applied independently to the two dates so the model
    learns to ignore illumination differences rather than treating them as
    change — that is the failure mode that hurts most on real Sentinel-2 pairs.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        size: int = 256,
        augment: bool = False,
        limit: int | None = None,
    ):
        self.root = Path(root) / split
        self.size = size
        self.augment = augment and split == "train"

        a_dir, b_dir, l_dir = self.root / "A", self.root / "B", self.root / "label"
        if not a_dir.is_dir() or not b_dir.is_dir() or not l_dir.is_dir():
            raise DatasetNotFound(
                f"Expected LEVIR-CD folders at {self.root}/A, /B and /label. "
                "Run `python download_data.py` (or extract LEVIR-CD into data/LEVIR-CD/)."
            )

        self.samples = []
        for a_path in sorted(p for p in a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            b_path = b_dir / a_path.name
            l_path = l_dir / a_path.name
            if b_path.exists() and l_path.exists():
                self.samples.append((a_path, b_path, l_path))
        if not self.samples:
            raise DatasetNotFound(f"No matching image triples found in {self.root}.")
        if limit is not None and limit < len(self.samples):
            step = len(self.samples) / limit
            self.samples = [self.samples[int(i * step)] for i in range(limit)]

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path, mode: str) -> Image.Image:
        img = Image.open(path).convert(mode)
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR if mode == "RGB" else Image.NEAREST)
        return img

    @staticmethod
    def _jitter(image: np.ndarray) -> np.ndarray:
        gain = random.uniform(0.9, 1.1)
        bias = random.uniform(-0.05, 0.05)
        return np.clip(image * gain + bias, 0.0, 1.0)

    def __getitem__(self, idx: int):
        a_path, b_path, l_path = self.samples[idx]
        a = np.array(self._load(a_path, "RGB"), dtype=np.float32) / 255.0
        b = np.array(self._load(b_path, "RGB"), dtype=np.float32) / 255.0
        lab = (np.array(self._load(l_path, "L"), dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                a, b, lab = a[:, ::-1], b[:, ::-1], lab[:, ::-1]
            if random.random() < 0.5:
                a, b, lab = a[::-1], b[::-1], lab[::-1]
            k = random.randint(0, 3)
            if k:
                a, b, lab = np.rot90(a, k), np.rot90(b, k), np.rot90(lab, k)
            if random.random() < 0.5:  # temporal order swap
                a, b = b, a
            if random.random() < 0.5:
                a = self._jitter(a)
            if random.random() < 0.5:
                b = self._jitter(b)

        a = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)).astype(np.float32))
        b = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)).astype(np.float32))
        lab = torch.from_numpy(np.ascontiguousarray(lab).astype(np.float32)).unsqueeze(0)
        return a, b, lab


def positive_fraction(dataset: LevirCDDataset, sample_size: int = 200) -> float:
    """Fraction of changed pixels, measured on a sample of the label masks.

    Used to set the BCE positive-class weight from the data instead of guessing.
    """
    paths = [s[2] for s in dataset.samples]
    if len(paths) > sample_size:
        step = len(paths) / sample_size
        paths = [paths[int(i * step)] for i in range(sample_size)]
    positives = total = 0
    for path in paths:
        mask = np.array(Image.open(path).convert("L")) > 127
        positives += int(mask.sum())
        total += int(mask.size)
    return positives / max(total, 1)


def build_loaders(
    root: str | Path,
    size: int = 256,
    batch_size: int = 8,
    workers: int = 2,
    train_limit: int | None = None,
    val_limit: int | None = None,
):
    train = LevirCDDataset(root, "train", size, augment=True, limit=train_limit)
    val = LevirCDDataset(root, "val", size, limit=val_limit)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            persistent_workers=workers > 0,
        ),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers),
        train,
        val,
    )
