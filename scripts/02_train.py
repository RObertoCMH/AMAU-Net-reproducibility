#!/usr/bin/env python3
"""
02_train.py

Train a selected model variant (unet_amp / unet_attrs / amau_net) under a
controlled multi-seed protocol.

Design choice:
- Attribute computation is done separately (01_compute_attributes.py).
- Preprocessing/normalization is OPTIONAL here (for simplicity). If enabled,
  normalization is fit on TRAIN only and applied to VAL/TEST (no leakage).

Expected:
- Split arrays from configs/split.yaml (train/val/test .npy)
- If in_channels=4: cached attributes exist in --attr_dir:
    train_attributes.npy, val_attributes.npy, test_attributes.npy
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

from amaunet.data import load_splits_from_config, validate_split_shapes
from amaunet.models import UNetWithAttention


# -----------------------------
# Utilities
# -----------------------------
def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_cfg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(y: np.ndarray, n_classes: int, eps: float = 1e-6, normalize: bool = True) -> torch.Tensor:
    """
    Inverse-frequency class weights from TRAIN labels only.
    w_c ∝ 1/(f_c + eps), optionally renormalized.
    """
    flat = y.reshape(-1)
    counts = np.bincount(flat.astype(np.int64), minlength=n_classes).astype(np.float64)
    freqs = counts / max(1.0, counts.sum())
    w = 1.0 / (freqs + eps)
    if normalize:
        w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)


def fit_train_channel_stats(x_train: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit per-channel mean/std on TRAIN only.
    x_train: [N,C,H,W] or [C,H,W]
    Returns (mu, sigma) broadcastable to x.
    """
    if x_train.ndim == 3:
        # [C,H,W]
        mu = x_train.mean(axis=(1, 2), keepdims=True)
        sigma = x_train.std(axis=(1, 2), keepdims=True)
    elif x_train.ndim == 4:
        # [N,C,H,W]
        mu = x_train.mean(axis=(0, 2, 3), keepdims=True)
        sigma = x_train.std(axis=(0, 2, 3), keepdims=True)
    else:
        raise ValueError(f"Expected x_train ndim 3 or 4, got {x_train.ndim}")

    sigma = np.maximum(sigma, eps)
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_channel_stats(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((x - mu) / sigma).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class NpySegmentationDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        x: [N,C,H,W]
        y: [N,H,W]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x shape [N,C,H,W], got {x.shape}")
        if y.ndim != 3:
            raise ValueError(f"Expected y shape [N,H,W], got {y.shape}")
        if x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[1:]:
            raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).long()
        return x, y


# -----------------------------
# Loading inputs
# -----------------------------
def load_inputs_for_variant(
    variant: str,
    model_cfg: Dict[str, Any],
    splits_cfg_path: str | Path,
    attr_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      x_train, y_train, x_val, y_val, x_test, y_test, variant_cfg
    """
    splits = load_splits_from_config(splits_cfg_path)
    validate_split_shapes(splits)

    variants = model_cfg["variants"]
    if variant not in variants:
        raise KeyError(f"Variant '{variant}' not found in configs/model.yaml variants: {list(variants.keys())}")

    vcfg = variants[variant]
    in_channels = int(vcfg["in_channels"])

    y_train = splits.train.labels.astype(np.int64)
    y_val   = splits.val.labels.astype(np.int64)
    y_test  = splits.test.labels.astype(np.int64)

    if in_channels == 1:
        # Use amplitude-only from split arrays: [N,H,W] -> [N,1,H,W]
        x_train = splits.train.seismic.astype(np.float32)[:, None, :, :]
        x_val   = splits.val.seismic.astype(np.float32)[:, None, :, :]
        x_test  = splits.test.seismic.astype(np.float32)[:, None, :, :]
    elif in_channels == 4:
        # Use cached attributes from attr_dir
        attr_dir = Path(attr_dir).expanduser().resolve()
        train_path = attr_dir / "train_attributes.npy"
        val_path   = attr_dir / "val_attributes.npy"
        test_path  = attr_dir / "test_attributes.npy"
        for p in [train_path, val_path, test_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing cached attributes file: {p}. "
                    f"Run scripts/01_compute_attributes.py first (or set --attr_dir)."
                )
        x_train = np.load(train_path).astype(np.float32)  # [N,4,H,W]
        x_val   = np.load(val_path).astype(np.float32)
        x_test  = np.load(test_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported in_channels={in_channels}. Expected 1 or 4.")

    return x_train, y_train, x_val, y_val, x_test, y_test, vcfg


# -----------------------------
# Training
# -----------------------------
def train_one_seed(
    seed: int,
    device: torch.device,
    model_params: Dict[str, Any],
    train_cfg: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: Path,
) -> Path:
    """
    Train one model for one seed. Saves the final checkpoint.
    Returns the checkpoint path.
    """
    deterministic = bool(train_cfg["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    batch_size = int(train_cfg["training"]["batch_size"])
    epochs = int(train_cfg["training"]["epochs"])

    # Dataset + loader
    train_ds = NpySegmentationDataset(x_train, y_train)
    val_ds   = NpySegmentationDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = UNetWithAttention(**model_params).to(device)

    # Loss weights from TRAIN only
    n_classes = int(model_params["n_classes"])
    loss_cfg = train_cfg["loss"]["class_weights"]
    w = compute_class_weights(
        y_train,
        n_classes=n_classes,
        eps=float(loss_cfg.get("eps", 1e-6)),
        normalize=bool(loss_cfg.get("normalize", True)),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=w)

    # Optimizer
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )

    # Scheduler (optional CLR)
    sched_cfg = train_cfg.get("scheduler", {"name": "none"})
    scheduler = None
    if sched_cfg.get("name", "none") == "cyclical_lr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=float(sched_cfg["base_lr"]),
            max_lr=float(sched_cfg["max_lr"]),
            mode=str(sched_cfg.get("mode", "triangular")),
            cycle_momentum=bool(sched_cfg.get("cycle_momentum", False)),
        )

    # Train loop
    best_val = None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # CLR per batch

            running += float(loss.item())
            n_batches += 1

        # Simple val loss (optional)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item())
                val_batches += 1

        train_loss = running / max(1, n_batches)
        val_loss = val_loss / max(1, val_batches)

        # Track best val loss (not used for early stopping; only informative)
        if best_val is None or val_loss < best_val:
            best_val = val_loss

        print(f"[seed {seed:03d}] epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # Save final checkpoint
    ckpt_path = out_dir / f"seed_{seed:03d}_last.pth"
    torch.save(
        {
            "seed": seed,
            "model_state": model.state_dict(),
            "model_params": model_params,
        },
        ckpt_path,
    )
    return ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split_cfg", type=str, default="configs/split.yaml")
    p.add_argument("--model_cfg", type=str, default="configs/model.yaml")
    p.add_argument("--train_cfg", type=str, default="configs/train.yaml")
    p.add_argument("--variant", type=str, required=True, help="unet_amp | unet_attrs | amau_net")
    p.add_argument("--attr_dir", type=str, default="outputs/attributes", help="Where *_attributes.npy live")
    p.add_argument("--out_dir", type=str, default="outputs/checkpoints", help="Base output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = read_yaml(args.model_cfg)
    train_cfg = read_yaml(args.train_cfg)

    device = select_device(train_cfg["training"].get("device", "auto"))
    print(f"[info] device = {device}")

    # Load inputs for selected variant
    x_train, y_train, x_val, y_val, x_test, y_test, vcfg = load_inputs_for_variant(
        variant=args.variant,
        model_cfg=model_cfg,
        splits_cfg_path=args.split_cfg,
        attr_dir=args.attr_dir,
    )

    # Build model parameters (merge defaults + variant)
    defaults = model_cfg["defaults"]
    model_params = {
        "n_classes": int(defaults["n_classes"]),
        "n_filters": int(defaults["n_filters"]),
        "dropout": float(defaults["dropout"]),
        "batchnorm": bool(defaults["batchnorm"]),
        "in_channels": int(vcfg["in_channels"]),
        "cbam_in_layers": list(vcfg.get("cbam_in_layers", [])),
        "use_self_attention": bool(vcfg.get("use_self_attention", False)),
    }

    # Output dir for this variant
    base_out = Path(args.out_dir).expanduser().resolve()
    run_out = base_out / args.variant
    run_out.mkdir(parents=True, exist_ok=True)

    # Save a small run manifest
    manifest = {
        "variant": args.variant,
        "model_params": model_params,
        "training": train_cfg.get("training", {}),
        "optimizer": train_cfg.get("optimizer", {}),
        "scheduler": train_cfg.get("scheduler", {}),
        "loss": train_cfg.get("loss", {}),
        "normalization": train_cfg.get("normalization", {}),
    }
    (run_out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- Normalization (paper-aligned): fit on TRAIN only, apply to TRAIN/VAL/TEST ---
    norm_cfg = train_cfg.get("normalization", None)
    if norm_cfg is None or not bool(norm_cfg.get("enabled", True)):
        raise ValueError(
            "Normalization must be enabled to match the manuscript. "
            "Set `normalization.enabled: true` in configs/train.yaml."
        )

    eps = float(norm_cfg.get("eps", 1e-6))

    mu, sigma = fit_train_channel_stats(x_train, eps=eps)
    x_train = apply_channel_stats(x_train, mu, sigma)
    x_val   = apply_channel_stats(x_val,   mu, sigma)
    x_test  = apply_channel_stats(x_test,  mu, sigma)

    # Save stats once (used later by evaluation)
    stats_path = run_out / "norm_stats.npz"
    np.savez(stats_path, mu=mu, sigma=sigma, eps=np.float32(eps))
    print(f"[info] saved normalization stats: {stats_path}")

    # Multi-seed
    seeds_cfg = train_cfg["training"]["seeds"]
    seeds = seeds_cfg.get("values", None)
    if seeds is None:
        n_runs = int(seeds_cfg.get("n_runs", 1))
        seeds = list(range(n_runs))

    ckpts = []
    for s in seeds:
        ckpt = train_one_seed(
            seed=int(s),
            device=device,
            model_params=model_params,
            train_cfg=train_cfg,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            out_dir=run_out,
        )
        ckpts.append(str(ckpt))

    print("[done] checkpoints:")
    for c in ckpts:
        print(" -", c)

    # Note: Evaluation is handled in a separate script (03_evaluate.py)

if __name__ == "__main__":
    main()
