"""Stage 2: compressed sensing by iterative soft thresholding.

Zero-filling assumes every unmeasured k-space point is zero. It is not zero,
we just did not look, and that lie is what makes the Stage 1 artifacts.
Compressed sensing instead asks: of all images consistent with what we
measured, which is the simplest?

It needs three things. Sparsity -- medical images have few large wavelet
coefficients and thousands of negligible ones (the same fact JPEG-2000 uses).
Incoherent sampling -- regular Cartesian skipping makes crisp ghosts that are
as sparse as the anatomy, so sparsity cannot tell them apart; random
variable-density sampling makes incoherent grain, which it can. And a
nonlinear solve:

    minimise  || M F x - y ||^2  +  lambda * || W x ||_1

ISTA alternates two steps: data consistency (fold the measured samples back
into the current guess) and denoising (soft-threshold the wavelet
coefficients). F is unitary, so the gradient step size is just 1. FISTA adds
Nesterov momentum for roughly quadratically faster convergence.
"""

from __future__ import annotations

import numpy as np
import pywt

from .kspace import from_kspace, to_kspace


def _ifft_complex(kspace_centered: np.ndarray) -> np.ndarray:
    """Inverse transform that keeps the phase.

    kspace.from_kspace returns abs(...), which would be a bug inside an
    iterative loop: the estimated image is complex, and FFT(|x|) != FFT(x), so
    data consistency would compare the k-space of the magnitude image against
    measurements from the complex one and fight itself. Magnitude is taken
    once, at the very end.
    """
    return np.fft.ifft2(np.fft.ifftshift(kspace_centered))

# db4 is the usual default in the CS-MRI literature. Haar is faster but its
# blocky basis leaves staircase artifacts on smooth anatomy.
DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 3


def _wavedec(image: np.ndarray, wavelet: str, level: int):
    """Forward 2-D wavelet transform, flattened so it can be thresholded in one call."""
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode="periodization")
    return pywt.coeffs_to_array(coeffs)


def _waverec(array: np.ndarray, slices, wavelet: str, level: int) -> np.ndarray:
    """Inverse of _wavedec."""
    coeffs = pywt.array_to_coeffs(array, slices, output_format="wavedec2")
    return pywt.waverec2(coeffs, wavelet=wavelet, mode="periodization")


def _threshold_details(
    coeff_array: np.ndarray,
    slices,
    threshold: float,
) -> np.ndarray:
    """Soft-threshold the detail bands only, leaving the approximation band alone.

    Correctness issue, not a refinement. The detail bands are near-zero over
    smooth tissue, so sparsity applies to them. The approximation band is a
    dense, high-energy summary of the anatomy with the largest coefficients in
    the transform; shrinking it every iteration drains energy, contrast drops,
    and PSNR ends up below the zero-filled baseline.
    """
    thresholded = soft_threshold(coeff_array, threshold)
    # slices[0] indexes the approximation block in the flattened array.
    thresholded[slices[0]] = coeff_array[slices[0]]
    return thresholded


def _detail_scale(coeff_array: np.ndarray, slices) -> float:
    """Largest detail coefficient, used to scale the threshold.

    The global maximum always comes from the exempt approximation band and is
    an order of magnitude larger, so it would make lambda_ mean different
    things on different images.
    """
    details = coeff_array.copy()
    details[slices[0]] = 0.0
    return float(np.abs(details).max())


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """soft(v, t) = sign(v) * max(|v| - t, 0).

    Anything smaller than t lands exactly on zero, which is what creates
    sparsity. Hard thresholding would too, but soft is the exact proximal
    operator of the L1 norm, which is what makes ISTA provably convergent.
    On complex input it shrinks the magnitude and leaves the phase.
    """
    if np.iscomplexobj(values):
        magnitude = np.abs(values)
        shrunk = np.maximum(magnitude - threshold, 0.0)
        # where= guards the division; magnitude == 0 gives 0 anyway.
        scale = np.divide(shrunk, magnitude, out=np.zeros_like(shrunk), where=magnitude > 0)
        return values * scale
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def ista_reconstruct(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    lambda_: float = 0.01,
    n_iter: int = 60,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    use_fista: bool = True,
    final_data_consistency: bool = True,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compressed-sensing reconstruction from undersampled k-space.

    lambda_ is a fraction of the largest wavelet detail coefficient; 0.01-0.05
    suits these images, and too large thresholds away real anatomy. n_iter of
    40-80 is plenty at 256x256. use_fista adds momentum and is strictly better
    -- the flag only exists so the demo can compare. final_data_consistency
    ends on a data step instead of a thresholding one, so the returned image
    still agrees with the scanner data. reference is diagnostic only, used to
    record per-iteration PSNR; the algorithm never looks at it.

    Returns (magnitude image, history of per-iteration dicts).
    """
    mask = mask.astype(bool)

    # Start from the zero-filled reconstruction -- the baseline this is trying
    # to beat. Kept complex; see _ifft_complex.
    current = _ifft_complex(kspace_measured * mask)

    # Threshold relative to the actual largest coefficient, so lambda_ means
    # the same thing across images and ratios.
    start_coeffs, slices = _wavedec(current, wavelet, level)
    threshold = lambda_ * _detail_scale(start_coeffs, slices)

    momentum_image = current.copy()
    t_previous = 1.0

    history: list[dict] = []

    for iteration in range(n_iter):
        # (a) Data consistency: replace the measured points with the true
        # measurements. F is unitary, so this is a gradient step of size 1.
        predicted = to_kspace(momentum_image)
        residual = np.where(mask, kspace_measured - predicted, 0.0)
        consistent = _ifft_complex(predicted + residual)

        # (b) Sparsity.
        coeffs, slices = _wavedec(consistent, wavelet, level)
        coeffs = _threshold_details(coeffs, slices, threshold)
        updated = _waverec(coeffs, slices, wavelet, level)

        if use_fista:
            t_current = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_previous ** 2))
            momentum_image = updated + ((t_previous - 1.0) / t_current) * (updated - current)
            t_previous = t_current
        else:
            momentum_image = updated

        current = updated

        data_error = float(
            np.sum(np.abs(np.where(mask, to_kspace(current) - kspace_measured, 0.0)) ** 2)
        )
        sparsity_cost = float(np.sum(np.abs(_wavedec(current, wavelet, level)[0])))
        entry = {
            "iteration": iteration + 1,
            "objective": data_error + threshold * sparsity_cost,
            "data_error": data_error,
        }
        if reference is not None:
            mse = float(np.mean((reference - np.abs(current)) ** 2))
            entry["psnr"] = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
        history.append(entry)

    if final_data_consistency:
        predicted = to_kspace(current)
        current = _ifft_complex(np.where(mask, kspace_measured, predicted))

        # Record this step too, or a convergence plot would stop one step short
        # of the image actually returned and seem to disagree with the PSNR.
        if reference is not None and history:
            mse = float(np.mean((reference - np.abs(current)) ** 2))
            history.append({
                "iteration": history[-1]["iteration"] + 1,
                "objective": history[-1]["objective"],
                "data_error": 0.0,          # measured points now match exactly
                "psnr": float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse),
            })

    # The one and only magnitude operation.
    return np.abs(current), history


def compare_with_zero_fill(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray,
    **kwargs,
) -> dict:
    """The Stage 2 headline: same mask, same samples, two answers to "what was
    not measured?" -- zero-fill says zero, CS says whatever is sparsest while
    still matching the measurements.
    """
    from .metrics import compute_metrics

    zero_filled = from_kspace(kspace_measured * mask)
    cs_image, history = ista_reconstruct(
        kspace_measured, mask, reference=reference, **kwargs
    )

    return {
        "zero_fill_image": zero_filled,
        "cs_image": cs_image,
        "zero_fill_metrics": compute_metrics(reference, zero_filled),
        "cs_metrics": compute_metrics(reference, cs_image),
        "history": history,
    }
