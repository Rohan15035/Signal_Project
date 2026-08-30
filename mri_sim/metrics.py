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
from scipy.ndimage import minimum_filter
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


# ---------------------------------------------------------------------------
# ROI-restricted metrics (used by the reduced-FOV work in mri_sim/roi.py)
# ---------------------------------------------------------------------------
#
# Whole-image PSNR/SSIM answer "how good is this picture?". For a targeted
# scan that is the wrong question. A reduced-FOV acquisition deliberately does
# not reconstruct anything outside the excited box -- there is no signal out
# there to reconstruct -- so a whole-image score would grade it on pixels it
# was never trying to produce, and report a triumph as a failure.
#
# The clinical question is "how good is the picture *of the thing we care
# about*?". A scan that resolves the lesion perfectly and leaves the skull a
# smear is a good scan. These functions score only inside a mask.

# Side length of the sliding window SSIM compares statistics over. Fixed
# explicitly rather than left to scikit-image's default, because the ROI
# erosion below has to use exactly the same window scikit-image did.
SSIM_WINDOW = 7


def compute_psnr_in_roi(
    original: np.ndarray, reconstruction: np.ndarray, roi_mask: np.ndarray
) -> float:
    """
    PSNR in dB over the pixels where `roi_mask` is True.

    PSNR is a per-pixel error measure, so restricting it is simply a matter of
    averaging the squared error over fewer pixels -- no windows, no borders.

    `data_range` is taken from the **whole** original image, not from the ROI
    crop. Two reasons: it keeps the number directly comparable with the
    whole-image PSNR reported everywhere else, and a small ROI sitting in a
    dim part of the anatomy would otherwise get a tiny peak value and a
    flattering score for the same absolute error.
    """
    original, reconstruction = _as_float_pair(original, reconstruction)
    selection = _as_roi_mask(roi_mask, original.shape)

    data_range = float(original.max() - original.min())
    if data_range == 0:
        raise ValueError("original image is constant; PSNR is undefined")

    mse = float(np.mean((original[selection] - reconstruction[selection]) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def compute_ssim_in_roi(
    original: np.ndarray, reconstruction: np.ndarray, roi_mask: np.ndarray
) -> float:
    """
    SSIM averaged over the pixels where `roi_mask` is True.

    SSIM cannot be evaluated on a bag of scattered pixels the way PSNR can: it
    is a *local* statistic, comparing the mean, variance and covariance inside
    a sliding window. So we compute the full SSIM **map** over the whole image
    and then average that map over the ROI.

    The ROI is **eroded by half a window** first, i.e. we keep only the pixels
    whose entire 7x7 window lies inside the mask. That is not conservatism for
    its own sake -- it is what makes the number mean what it says. The SSIM
    value at a pixel is a property of its whole window, so a window straddling
    the ROI boundary is partly scored on pixels outside the region we claimed
    to care about. In a reduced-FOV scan those outside pixels are the aliased
    replicas the method deliberately does not reconstruct, and leaving them in
    drags a numerically exact ROI down from 1.00 to ~0.93 for no reason that
    concerns the ROI. Erosion also removes the image border, where
    scikit-image's windows are incomplete anyway.

    The cost is that the ROI must be at least `SSIM_WINDOW` pixels across;
    below that there is no window that fits, and the function raises.
    """
    original, reconstruction = _as_float_pair(original, reconstruction)
    selection = _as_roi_mask(roi_mask, original.shape)

    data_range = float(original.max() - original.min())
    if data_range == 0:
        raise ValueError("original image is constant; SSIM is undefined")

    _, ssim_map = structural_similarity(
        original, reconstruction,
        data_range=data_range, win_size=SSIM_WINDOW, full=True,
    )

    # Erosion by a square structuring element == a minimum filter on a 0/1
    # array. `cval=0` treats everything off the edge of the image as outside
    # the ROI, which is what handles the image border for free.
    interior = minimum_filter(
        selection.astype(np.uint8), size=SSIM_WINDOW, mode="constant", cval=0
    ).astype(bool)
    if not interior.any():
        raise ValueError(
            f"ROI is too small for SSIM: no {SSIM_WINDOW}x{SSIM_WINDOW} window "
            f"fits entirely inside it. Use a larger region, or report PSNR only."
        )

    return float(ssim_map[interior].mean())


def compute_metrics_in_roi(
    original: np.ndarray, reconstruction: np.ndarray, roi_mask: np.ndarray
) -> dict:
    """
    Both metrics, scored only inside `roi_mask`.

    Same return shape as `compute_metrics`, so the two are interchangeable in
    the tables and figures.
    """
    return {
        "psnr": compute_psnr_in_roi(original, reconstruction, roi_mask),
        "ssim": compute_ssim_in_roi(original, reconstruction, roi_mask),
    }


def _as_roi_mask(roi_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Shape-check the ROI mask and cast it to boolean (non-zero = inside)."""
    mask = np.asarray(roi_mask)
    if mask.shape != shape:
        raise ValueError(f"ROI mask shape {mask.shape} does not match image {shape}")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("ROI mask is empty; there is nothing to score")
    return mask
