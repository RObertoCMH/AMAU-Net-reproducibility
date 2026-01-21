#!/usr/bin/env python3
"""
03_evaluate.py

Evaluate a trained variant on the held-out test split and export:
- per-seed metrics (CSV)
- mean ± std across seeds (CSV)

Normalization (paper-aligned):
- Loads per-channel z-score statistics (mu, sigma) computed on TRAIN and saved by training
  as: outputs/checkpoints/<variant>/norm_stats.npz
- Applies the same mu/sigma to TEST (no leakage, no re-fitting)

Metrics from test confusion matrix:
- PA  (pixel accuracy)
- MCA (macro class accuracy / mean per-class recall)
- mIoU
- Dice_fw (frequency-weighted Dice)
- FWIU (frequency-weighted IoU)
- per-class IoU

Input selection:
- in_channels=1: amplitude-only from split arrays
- in_channels=4: cached attributes from outputs/attributes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

from amaunet.data import load_splits_from_config, validate_split_shapes
from amaunet.models import UNetWithAttention


# -----------------------------
# Config / device
# -----------------------------
def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_cfg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Normalization helpers
# -----------------------------
def apply_channel_stats(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Apply per-channel z-score stats.
    x: [N,C,H,W]
    mu,sigma: broadcastable to x
    """
    return ((x - mu) / sigma).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class NpySegmentationDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # x: [N,C,H,W], y: [N,H,W]
        if x.ndim != 4:
            raise ValueError(f"Expected x [N,C,H,W], got {x.shape}")
        if y.ndim != 3:
            raise ValueError(f"Expected y [N,H,W], got {y.shape}")
        if x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[1:]:
            raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).long()


# -----------------------------
# Loading inputs for variants
# -----------------------------
def load_inputs_for_variant(
    variant: str,
    model_cfg: Dict[str, Any],
    split_cfg_path: str | Path,
    attr_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x_test, y_test

    Inputs:
    - if in_channels=1: amplitude-only from split arrays
    - if in_channels=4: cached attributes from attr_dir
    """
    splits = load_splits_from_config(split_cfg_path)
    validate_split_shapes(splits)

    variants = model_cfg["variants"]
    if variant not in variants:
        raise KeyError(f"Variant '{variant}' not found in configs/model.yaml. Available: {list(variants.keys())}")

    vcfg = variants[variant]
    in_channels = int(vcfg["in_channels"])

    y_test = splits.test.labels.astype(np.int64)

    if in_channels == 1:
        x_test = splits.test.seismic.astype(np.float32)[:, None, :, :]
    elif in_channels == 4:
        attr_dir = Path(attr_dir).expanduser().resolve()
        test_path = attr_dir / "test_attributes.npy"
        if not test_path.exists():
            raise FileNotFoundError(
                f"Missing cached attributes file: {test_path}. "
                f"Run scripts/01_compute_attributes.py first."
            )
        x_test = np.load(test_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported in_channels={in_channels}. Expected 1 or 4.")

    return x_test, y_test


# -----------------------------
# Confusion + metrics
# -----------------------------
def confusion_from_logits(logits: torch.Tensor, y_true: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Confusion matrix for one batch.
    cm[i,j] = #pixels with true class i predicted as j
    """
    y_pred = torch.argmax(logits, dim=1)  # [B,H,W]
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    mask = (y_true >= 0) & (y_true < n_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    idx = n_classes * y_true + y_pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return cm


def metrics_from_confusion(cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """
    cm[i,j] = count of true class i predicted as j
    """
    eps = 1e-12
    gt = cm.sum(axis=1)          # true pixels per class
    pred = cm.sum(axis=0)        # predicted pixels per class
    tp = np.diag(cm)
    total = cm.sum()

    pa = float(tp.sum() / max(eps, total))

    recall = tp / np.maximum(gt, eps)
    mca = float(np.mean(recall))

    union = gt + pred - tp
    iou = tp / np.maximum(union, eps)
    miou = float(np.mean(iou))

    dice = (2 * tp) / np.maximum(gt + pred, eps)
    freq = gt / np.maximum(gt.sum(), eps)
    dice_fw = float((freq * dice).sum())
    fwiu = float((freq * iou).sum())

    out: Dict[str, Any] = {
        "PA": pa,
        "MCA": mca,
        "mIoU": miou,
        "Dice_fw": dice_fw,
        "FWIU": fwiu,
    }
    for k, name in enumerate(class_names):
        out[f"IoU_{name}"] = float(iou[k])

    return out


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split_cfg", type=str, default="configs/split.yaml")
    p.add_argument("--model_cfg", type=str, default="configs/model.yaml")
    p.add_argument("--train_cfg", type=str, default="configs/train.yaml")
    p.add_argument("--eval_cfg", type=str, default="configs/eval.yaml")
    p.add_argument("--variant", type=str, required=True)
    p.add_argument("--attr_dir", type=str, default="outputs/attributes")
    p.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    p.add_argument("--out_dir", type=str, default="outputs/eval")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    model_cfg = read_yaml(args.model_cfg)
    train_cfg = read_yaml(args.train_cfg)
    eval_cfg = read_yaml(args.eval_cfg)

    device = select_device(train_cfg["training"].get("device", "auto"))
    print(f"[info] device = {device}")

    # Load test inputs
    x_test, y_test = load_inputs_for_variant(
        variant=args.variant,
        model_cfg=model_cfg,
        split_cfg_path=args.split_cfg,
        attr_dir=args.attr_dir,
    )

    # --- Normalization (paper-aligned): load TRAIN stats from training run ---
    norm_cfg = train_cfg.get("normalization", None)
    if norm_cfg is None or not bool(norm_cfg.get("enabled", True)):
        raise ValueError(
            "Normalization must be enabled to match the manuscript. "
            "Set `normalization.enabled: true` in configs/train.yaml."
        )

    ckpt_base = Path(args.ckpt_dir).expanduser().resolve() / args.variant
    stats_path = ckpt_base / "norm_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing normalization stats: {stats_path}. "
            "Run training first to generate norm_stats.npz."
        )

    stats = np.load(stats_path)
    mu = stats["mu"]
    sigma = stats["sigma"]
    x_test = apply_channel_stats(x_test, mu, sigma)
    print(f"[info] loaded normalization stats: {stats_path}")

    # DataLoader
    batch_size = int(train_cfg["training"]["batch_size"])
    test_ds = NpySegmentationDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluation config
    class_names = eval_cfg["evaluation"]["class_names"]
    n_classes = int(eval_cfg["evaluation"]["n_classes"])

    # Model params from defaults + variant toggles
    defaults = model_cfg["defaults"]
    vcfg = model_cfg["variants"][args.variant]
    model_params = dict(
        n_classes=int(defaults["n_classes"]),
        n_filters=int(defaults["n_filters"]),
        dropout=float(defaults["dropout"]),
        batchnorm=bool(defaults["batchnorm"]),
        in_channels=int(vcfg["in_channels"]),
        cbam_in_layers=list(vcfg.get("cbam_in_layers", [])),
        use_self_attention=bool(vcfg.get("use_self_attention", False)),
    )

    # Seeds to evaluate (must match training)
    seeds_cfg = train_cfg["training"]["seeds"]
    seeds = seeds_cfg.get("values", None)
    if seeds is None:
        n_runs = int(seeds_cfg.get("n_runs", 1))
        seeds = list(range(n_runs))

    # Output dirs
    out_base = Path(args.out_dir).expanduser().resolve() / args.variant
    out_base.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() / args.variant

    rows = []
    for seed in seeds:
        seed = int(seed)
        ckpt_path = ckpt_dir / f"seed_{seed:03d}_last.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        # Load model
        model = UNetWithAttention(**model_params).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Accumulate confusion
        cm = torch.zeros((n_classes, n_classes), dtype=torch.int64)
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                cm += confusion_from_logits(logits, yb, n_classes=n_classes).cpu()

        metrics = metrics_from_confusion(cm.numpy(), class_names=class_names)
        metrics["seed"] = seed
        rows.append(metrics)

        print(f"[seed {seed:03d}] PA={metrics['PA']:.4f} mIoU={metrics['mIoU']:.4f} MCA={metrics['MCA']:.4f}")

    # Save per-seed metrics
    df = pd.DataFrame(rows).sort_values("seed")
    per_seed_path = out_base / "metrics_per_seed.csv"
    df.to_csv(per_seed_path, index=False)

    # Save mean ± std across seeds
    num_cols = [c for c in df.columns if c != "seed"]
    summary = df[num_cols].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    summary_path = out_base / "metrics_summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    print("[ok] saved:", per_seed_path)
    print("[ok] saved:", summary_path)


if __name__ == "__main__":
    main()
