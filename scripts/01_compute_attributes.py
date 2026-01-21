#!/usr/bin/env python3
"""
Compute and cache 4-channel inputs (amplitude + instantaneous attributes).

Output for each split:
  {out_dir}/{split}_attributes.npy

Channel order:
  0: amplitude
  1: instantaneous phase (wrapped, rad)
  2: angular instantaneous frequency (rad/sample)
  3: envelope

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from amaunet.data import (
    load_splits_from_config,
    validate_split_shapes,
    print_split_summary,
)
from amaunet.data.seismic_attributes import instantaneous_attributes


def stack_4ch(amplitude: np.ndarray, phase: np.ndarray, angfreq: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Stack channels as [N, 4, H, W] for [N,H,W] inputs, or [4,H,W] for [H,W]."""
    amp = amplitude.astype(np.float32)
    phase = phase.astype(np.float32)
    angfreq = angfreq.astype(np.float32)
    env = env.astype(np.float32)

    if amp.ndim == 2:
        return np.stack([amp, phase, angfreq, env], axis=0)
    if amp.ndim == 3:
        return np.stack([amp, phase, angfreq, env], axis=1)
    raise ValueError(f"Expected amplitude ndim=2 or 3, got {amp.ndim}")


def compute_and_save(split_name: str, amp: np.ndarray, out_dir: Path) -> Path:
    """Compute attributes and save a 4-channel .npy file for a given split."""
    phase, angfreq, env = instantaneous_attributes(amp, axis=-1)
    x4 = stack_4ch(amp, phase, angfreq, env)

    out_path = out_dir / f"{split_name}_attributes.npy"
    np.save(out_path, x4)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split_cfg", type=str, default="configs/split.yaml", help="Path to split.yaml")
    p.add_argument("--out_dir", type=str, default="outputs/attributes", help="Output directory for cached attributes")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cached files")
    p.add_argument("--quiet", action="store_true", help="Suppress split summary prints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits_from_config(args.split_cfg)
    validate_split_shapes(splits)

    if not args.quiet:
        print_split_summary(splits)

    for name, amp in [
        ("train", splits.train.seismic),
        ("val", splits.val.seismic),
        ("test", splits.test.seismic),
    ]:
        out_path = out_dir / f"{name}_attributes.npy"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} already exists (use --overwrite to recompute).")
            continue

        saved = compute_and_save(name, amp, out_dir)
        shape = np.load(saved, mmap_mode="r").shape
        print(f"[ok] saved {name} attributes: {saved}  | shape={shape}")


if __name__ == "__main__":
    main()
