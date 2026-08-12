"""
metrics.py -- image quality metrics.

Thin, documented wrappers around scikit-image's implementations so that the
rest of the code never has to think about `data_range` bookkeeping.

Both metrics compare a reconstruction against the original (ground truth)
image. They measure different things and it is worth quoting both:

PSNR (Peak Signal-to-Noise Ratio, in dB)
    A logarithmic restatement of mean squared error:

        PSNR = 10 * log10( peak^2 / MSE )

    Purely a per-pixel error measure. Higher is better; +6 dB is roughly
    "half the error". It is completely blind to *structure* -- it cannot
    tell a bit of uniform noise from a coherent ghost artifact of the same
    energy, even though the second is far more damaging to a radiologist.

SSIM (Structural Similarity Index, unitless, in [-1, 1])
    Compares local means, variances and covariance over a sliding window, so
    it responds to loss of structure, contrast and texture rather than raw
    pixel error. 1.0 is a perfect match. It tracks human judgement of image
    quality much better than PSNR, which is why both are reported.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """
    PSNR in dB between `original` and `reconstruction`.

    `data_range` is fixed to the dynamic range of the *original* image rather
    than of the reconstruction. That is deliberate: the peak in the PSNR
    formula must be a property of the reference signal, otherwise a
    reconstruction with a few bright artifact pixels would inflate its own
    denominator and score better than it deserves.
    """
    original, reconstruction = _as_float_pair(original, reconstruction)
    data_range = float(original.max() - original.min())
    if data_range == 0:
        raise ValueError("original image is constant; PSNR is undefined")
    return float(
        peak_signal_noise_ratio(original, reconstruction, data_range=data_range)
    )


def compute_ssim(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """
    SSIM between `original` and `reconstruction`.

    Same `data_range` reasoning as above.
    """
    original, reconstruction = _as_float_pair(original, reconstruction)
    data_range = float(original.max() - original.min())
    if data_range == 0:
        raise ValueError("original image is constant; SSIM is undefined")
    return float(
        structural_similarity(original, reconstruction, data_range=data_range)
    )


def compute_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    """
    Both metrics at once, as a dict -- the form main.py collects into a table.

    Returns
    -------
    {"psnr": float (dB), "ssim": float}
    """
    return {
        "psnr": compute_psnr(original, reconstruction),
        "ssim": compute_ssim(original, reconstruction),
    }


def _as_float_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shape-check and cast both images to float64.

    Guards against the easy mistake of passing a complex array straight out
    of an inverse FFT -- reconstructions must have had `np.abs` applied first
    (see `kspace.from_kspace`).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError(
            "metrics expect real images; take the magnitude of the "
            "reconstruction first"
        )
    return a.astype(np.float64), b.astype(np.float64)
