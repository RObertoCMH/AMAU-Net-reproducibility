"""
seismic_attributes.py

Hilbert-based instantaneous attributes used in AMAU-Net.

This module matches the attribute computation in the original notebook:

    analytic_signal = hilbert(seismic, axis=-1)
    envelope        = abs(analytic_signal)
    phase_wrapped   = angle(analytic_signal)
    ang_freq        = gradient(unwrap(phase_wrapped), axis=-1)

The attribute computation is applied trace-wise along the last axis (axis=-1),
which must correspond to the time/sample dimension in the stored arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import hilbert


def instantaneous_attributes(
    seismic: np.ndarray,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute instantaneous phase (wrapped), angular instantaneous frequency,
    and envelope from a seismic amplitude array.
    
    """
    if not isinstance(seismic, np.ndarray):
        raise TypeError(f"`seismic` must be a numpy array, got {type(seismic)}")

    analytic_signal = hilbert(seismic, axis=axis)

    envelope = np.abs(analytic_signal).astype(np.float32)

    # Wrapped phase in radians
    phase_wrapped = np.angle(analytic_signal).astype(np.float32)

    # Unwrap ONLY for derivative computation
    phase_unwrapped = np.unwrap(phase_wrapped, axis=axis).astype(np.float32)

    # Angular instantaneous frequency in rad/sample
    ang_inst_freq = np.gradient(phase_unwrapped, axis=axis).astype(np.float32)

    return phase_wrapped, ang_inst_freq, envelope


def build_four_channel_input(
    amplitude: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Build the 4-channel input used by U-Net + attributes / AMAU-Net:

        [amplitude, phase_wrapped, ang_inst_freq, envelope]

    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude array with shape [H, W] or [N, H, W].
    axis : int
        Time/sample axis for Hilbert/unwrap/gradient. Default: -1.

    Returns
    -------
    x4 : np.ndarray
        Four-channel array with shape:
        - [4, H, W] for a single section input, or
        - [N, 4, H, W] for a stack of sections.
    """
    phase, ang_freq, env = instantaneous_attributes(amplitude, axis=axis)

    amp = amplitude.astype(np.float32)

    if amp.ndim == 2:
        # [H, W] -> [4, H, W]
        return np.stack([amp, phase, ang_freq, env], axis=0)

    if amp.ndim == 3:
        # [N, H, W] -> [N, 4, H, W]
        return np.stack([amp, phase, ang_freq, env], axis=1)

    raise ValueError(f"Expected amplitude ndim=2 or 3, got {amp.ndim}")


