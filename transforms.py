"""
Offline perturbations of fixed-length traffic direction representations.

These functions support sensitivity checks on trained classifiers.
"""

from typing import Tuple
import numpy as np


def add_timing_jitter(
    sequence: np.ndarray,
    jitter_std: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """
    Return a copy of the unchanged input. Parameters are retained for legacy runs.
    """
    return sequence.copy()


def front_padding(
    sequence: np.ndarray,
    pad_length: int = 100,
    pad_value: int = 0,
) -> np.ndarray:
    """Prepend constant direction symbols to the stored input array."""
    pad = np.full(pad_length, pad_value, dtype=sequence.dtype)
    return np.concatenate([pad, sequence])


def burst_smoothing(
    sequence: np.ndarray,
    window: int = 8,
) -> np.ndarray:
    """
    In windows with more than half as many transitions as positions, replace
    the midpoint with the first direction symbol from the original window.
    """
    if len(sequence) < window:
        return sequence.copy()

    smoothed = sequence.copy()
    for i in range(len(sequence) - window):
        window_vals = sequence[i : i + window]
        # If the window is highly alternating, flatten toward the first value
        if len(set(window_vals)) >= 2 and np.sum(np.diff(window_vals) != 0) > window // 2:
            smoothed[i + window // 2] = window_vals[0]
    return smoothed


def apply_defense(
    sequence: np.ndarray,
    defense: str = "none",
    **kwargs,
) -> np.ndarray:
    """Convenience dispatcher for all supported defensive transformations."""
    if defense == "front_padding":
        return front_padding(sequence, **kwargs)
    elif defense == "burst_smoothing":
        return burst_smoothing(sequence, **kwargs)
    elif defense == "timing_jitter":
        return add_timing_jitter(sequence, **kwargs)
    else:
        return sequence.copy()


def normalize_length(seq: np.ndarray, target_len: int, pad_value: int = 0) -> np.ndarray:
    """
    Truncate or right-pad the representation to the classifier's input length.
    """
    if len(seq) == target_len:
        return seq.copy()
    if len(seq) > target_len:
        return seq[:target_len].copy()
    # pad on the right
    pad = np.full(target_len - len(seq), pad_value, dtype=seq.dtype)
    return np.concatenate([seq, pad])
