"""
splits.py

Utilities to load the precomputed F3 Netherlands splits from NumPy arrays.

This repo assumes the splits already exist on disk as .npy files, e.g.:

- train/train_seismic.npy and train/train_labels.npy
- validation/test2_seismic.npy and validation/test2_labels.npy
- test/test1_seismic.npy and test/test1_labels.npy  (Benchmark Test set #1)

Paths are defined in `configs/split.yaml`.

Important:
- The dataset is NOT redistributed in this repository.
- Users must set `data.base_dir` in the YAML to their local path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required to read configs. Install with: pip install pyyaml") from e


@dataclass(frozen=True)
class SplitArrays:
    """Container for one split."""
    seismic: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train/val/test."""
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_npy(base_dir: Path, rel_path: str) -> np.ndarray:
    fpath = base_dir / rel_path
    if not fpath.exists():
        raise FileNotFoundError(f"Missing file: {fpath}")
    return np.load(fpath)


def load_splits_from_config(split_config_path: str | Path) -> DatasetSplits:
    """
    Load train/val/test arrays using `configs/split.yaml`.

    Parameters
    ----------
    split_config_path : str | Path
        Path to `configs/split.yaml`.

    Returns
    -------
    DatasetSplits
        train/val/test arrays.
    """
    cfg = _read_yaml(split_config_path)

    if "data" not in cfg:
        raise KeyError("Expected top-level key `data` in split config.")

    data_cfg = cfg["data"]
    base_dir = data_cfg.get("base_dir", None)
    if not base_dir or "PATH/TO/YOUR" in str(base_dir):
        raise ValueError(
            "Please set `data.base_dir` in configs/split.yaml to your local data directory."
        )

    base_dir = Path(base_dir).expanduser().resolve()

    # Train
    train_seis = _load_npy(base_dir, data_cfg["train"]["seismic"])
    train_lbls = _load_npy(base_dir, data_cfg["train"]["labels"])

    # Validation
    val_seis = _load_npy(base_dir, data_cfg["val"]["seismic"])
    val_lbls = _load_npy(base_dir, data_cfg["val"]["labels"])

    # Test
    test_seis = _load_npy(base_dir, data_cfg["test"]["seismic"])
    test_lbls = _load_npy(base_dir, data_cfg["test"]["labels"])

    return DatasetSplits(
        train=SplitArrays(seismic=train_seis, labels=train_lbls),
        val=SplitArrays(seismic=val_seis, labels=val_lbls),
        test=SplitArrays(seismic=test_seis, labels=test_lbls),
    )


def validate_split_shapes(splits: DatasetSplits) -> None:
    """
    Basic sanity checks for split arrays.

    Expected:
    - seismic shape: [N, H, W] (or [H, W] if single section, but usually [N,H,W])
    - labels shape:  [N, H, W] matching seismic in N/H/W
    """
    for name, split in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        x, y = split.seismic, split.labels

        if x.shape != y.shape:
            raise ValueError(f"{name}: seismic and labels shapes differ: {x.shape} vs {y.shape}")

        if x.ndim not in (2, 3):
            raise ValueError(f"{name}: expected seismic ndim 2 or 3, got {x.ndim}")

        # Optional: label dtype sanity
        if not np.issubdtype(y.dtype, np.integer):
            # Many pipelines store labels as int; warn by raising only if really wrong
            raise TypeError(f"{name}: labels dtype should be integer, got {y.dtype}")


def print_split_summary(splits: DatasetSplits) -> None:
    """
    Print a short summary (shapes + unique classes count) to help debugging.
    """
    for name, split in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        x, y = split.seismic, split.labels
        uniq = np.unique(y)
        print(f"{name}: X {x.shape} | Y {y.shape} | classes={len(uniq)} | min={uniq.min()} max={uniq.max()}")
