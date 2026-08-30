"""PSNR and SSIM wrappers around scikit-image, with data_range handled here.

Both are worth quoting: PSNR is a per-pixel error measure and is blind to
structure, so it cannot tell uniform noise from a coherent ghost of the same
energy. SSIM compares local means, variances and covariance, so it responds to
lost structure and tracks human judgement better.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """PSNR in dB. Higher is better; +6 dB is roughly half the error."""
    original, reconstruction = _as_float_pair(original, reconstruction)
    # data_range comes from the original, not the reconstruction: otherwise a
    # few bright artifact pixels would inflate the peak and flatter the score.
    data_range = float(original.max() - original.min())
    if data_range == 0:
        raise ValueError("original image is constant; PSNR is undefined")
    return float(
        peak_signal_noise_ratio(original, reconstruction, data_range=data_range)
    )


def compute_ssim(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """SSIM in [-1, 1]; 1.0 is a perfect match."""
    original, reconstruction = _as_float_pair(original, reconstruction)
    data_range = float(original.max() - original.min())
    if data_range == 0:
        raise ValueError("original image is constant; SSIM is undefined")
    return float(
        structural_similarity(original, reconstruction, data_range=data_range)
    )


def compute_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    """Both metrics as {"psnr": dB, "ssim": float} -- one row of main.py's table."""
    return {
        "psnr": compute_psnr(original, reconstruction),
        "ssim": compute_ssim(original, reconstruction),
    }


def _as_float_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape-check and cast to float64, rejecting complex input."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    # Catches passing a raw ifft2 result instead of its magnitude.
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError(
            "metrics expect real images; take the magnitude of the "
            "reconstruction first"
        )
    return a.astype(np.float64), b.astype(np.float64)
