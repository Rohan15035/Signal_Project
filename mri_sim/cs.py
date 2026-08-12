"""
cs.py -- Stage 2: compressed sensing by iterative soft thresholding.

THE PROBLEM
-----------
Zero-filling is the honest but naive reconstruction: it assumes every
unmeasured k-space point is zero. It is not zero, we simply did not look. That
lie is what produces the artifacts in Stage 1.

Compressed sensing asks a better question. Instead of "what image has these
measurements and zeros elsewhere?", it asks:

    of all the images consistent with what we measured,
    which one is the *simplest*?

THREE INGREDIENTS
-----------------
1. SPARSITY. Medical images are compressible: in a wavelet basis, a handful of
   large coefficients describe the image and thousands of tiny ones describe
   almost nothing. (This is exactly why JPEG-2000 works.) "Simplest" therefore
   means "fewest significant wavelet coefficients".

2. INCOHERENT SAMPLING. The undersampling artifacts must look like *noise* in
   the sparsifying basis, not like structure. Regular Cartesian skipping fails
   here: it produces crisp ghost copies of the anatomy, which are just as
   sparse as the anatomy itself, so no sparsity-based method can tell them
   apart. Random variable-density sampling spreads the error into low-level
   incoherent grain, which sparsity *can* remove. This is why compressed
   sensing is demonstrated on the random mask and not the Cartesian one.

3. A NONLINEAR RECONSTRUCTION. We solve

       minimise  || M F x - y ||^2  +  lambda * || W x ||_1

   where F is the 2-D FFT, M the sampling mask, y the measured samples, W the
   wavelet transform, and ||.||_1 the sum of absolute values -- the term that
   pushes small coefficients to exactly zero.

THE ALGORITHM (ISTA / FISTA)
----------------------------
Iterative Soft-Thresholding alternates two intuitive steps:

    a) DATA CONSISTENCY -- put back what we actually measured:
           x <- x + F^-1 ( M (y - F x) )
       i.e. compute what our current guess predicts for the measured samples,
       take the difference from the real measurements, and fold it back in.

    b) DENOISE BY SPARSIFYING -- shrink every wavelet coefficient toward zero
       by lambda, setting to zero anything smaller:
           x <- W^-1 ( soft( W x, lambda ) )

Step (a) is a gradient step on the data term (the FFT's unitarity makes the
Lipschitz constant 1, so the step size is simply 1 -- no tuning needed). Step
(b) is the proximal operator of the L1 term. Alternating them is provably
convergent.

FISTA adds Nesterov momentum: extrapolate past the previous iterate before
the next step. Same cost per iteration, roughly quadratically faster
convergence, three extra lines of code.
"""

from __future__ import annotations

import numpy as np
import pywt

from .kspace import from_kspace, to_kspace


def _ifft_complex(kspace_centered: np.ndarray) -> np.ndarray:
    """
    Inverse transform that **keeps the phase**.

    `kspace.from_kspace` returns `abs(...)`, because that is what a scanner
    console displays and what Stage 1 compares against. Inside an iterative
    reconstruction that would be a serious bug: the image being estimated is
    complex (real MRI images have phase from B0 inhomogeneity and coil
    sensitivities, and the sample store models this explicitly), and

        FFT( |x| )  !=  FFT( x )

    So if we collapsed to magnitude on every iteration, the data-consistency
    step would compare the k-space of the *magnitude* image against
    measurements taken from the *complex* one, and the iteration would fight
    itself -- converging to something worse than the zero-filled image it
    started from.

    The magnitude is taken exactly once, at the very end.
    """
    return np.fft.ifft2(np.fft.ifftshift(kspace_centered))

# db4 balances smoothness against support width and is the usual default in
# the CS-MRI literature. Haar ('db1') is faster but its blocky basis leaves
# visible staircase artifacts on smooth anatomy.
DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 3


# ---------------------------------------------------------------------------
# The sparsifying transform
# ---------------------------------------------------------------------------


def _wavedec(image: np.ndarray, wavelet: str, level: int):
    """Forward 2-D wavelet transform, flattened to a coefficient array."""
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode="periodization")
    # coeffs_to_array gives one flat array we can threshold in a single call,
    # plus the bookkeeping needed to put it back together.
    return pywt.coeffs_to_array(coeffs)


def _waverec(array: np.ndarray, slices, wavelet: str, level: int) -> np.ndarray:
    """Inverse of :func:`_wavedec`."""
    coeffs = pywt.array_to_coeffs(array, slices, output_format="wavedec2")
    return pywt.waverec2(coeffs, wavelet=wavelet, mode="periodization")


def _threshold_details(
    coeff_array: np.ndarray,
    slices,
    threshold: float,
) -> np.ndarray:
    """
    Soft-threshold the wavelet *detail* bands only, leaving the coarse
    approximation band untouched.

    This is not a refinement, it is a correctness issue. `wavedec2` splits an
    image into one low-resolution approximation band (a shrunken copy of the
    image) plus detail bands (edges at each scale). The sparsity assumption
    applies to the detail bands: they are near-zero over smooth tissue and
    large only at boundaries. The approximation band is the opposite -- it is
    a dense, high-energy summary of the anatomy, and its coefficients are the
    largest in the whole transform.

    Shrinking that band by the same threshold on every iteration steadily
    drains energy out of the image: contrast drops, the reconstruction dims,
    and PSNR ends up *below* the zero-filled baseline it started from. Every
    practical CS-MRI implementation exempts the approximation band.
    """
    thresholded = soft_threshold(coeff_array, threshold)
    # slices[0] indexes the approximation block inside the flattened array.
    thresholded[slices[0]] = coeff_array[slices[0]]
    return thresholded


def _detail_scale(coeff_array: np.ndarray, slices) -> float:
    """
    Largest *detail* coefficient magnitude, used to scale the threshold.

    Scaling against the global maximum would be misleading, because that
    maximum always comes from the (now exempt) approximation band and is an
    order of magnitude larger. Referencing the detail bands keeps a given
    `lambda_` meaning roughly the same thing across different images.
    """
    details = coeff_array.copy()
    details[slices[0]] = 0.0
    return float(np.abs(details).max())


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding (the "shrinkage" operator):

        soft(v, t) = sign(v) * max(|v| - t, 0)

    Every coefficient moves `t` toward zero; anything already smaller than `t`
    lands exactly on zero. That last part is what actually creates sparsity --
    hard thresholding would zero small coefficients too, but soft thresholding
    is the exact proximal operator of the L1 norm, which is what makes the
    iteration provably convergent.

    Written to work on complex input as well: there the operator shrinks the
    magnitude and leaves the phase untouched.
    """
    if np.iscomplexobj(values):
        magnitude = np.abs(values)
        shrunk = np.maximum(magnitude - threshold, 0.0)
        # Guard the division: where magnitude == 0 the result is 0 anyway.
        scale = np.divide(shrunk, magnitude, out=np.zeros_like(shrunk), where=magnitude > 0)
        return values * scale
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


# ---------------------------------------------------------------------------
# The reconstruction
# ---------------------------------------------------------------------------


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
    """
    Compressed-sensing reconstruction from undersampled k-space.

    Parameters
    ----------
    kspace_measured : complex array
        The acquired k-space (already masked, and noisy if noise was
        simulated), in the centered convention used throughout the package.
    mask : array
        The sampling mask that produced it. Needed because data consistency
        must only enforce the points that were actually measured.
    lambda_ : float
        Regularisation strength, as a fraction of the largest wavelet
        coefficient of the zero-filled starting image. Larger = sparser =
        smoother; too large and real anatomy gets thresholded away. 0.01-0.05
        is a sensible range for these images.
    n_iter : int
        Number of iterations. 40-80 is plenty at 256x256; the curve is flat
        long before that.
    wavelet, level : str, int
        The sparsifying basis.
    use_fista : bool
        Add Nesterov momentum (FISTA). Strictly better; the flag exists so the
        two can be compared in the demo.
    final_data_consistency : bool
        Finish on a data-consistency step rather than on a thresholding step.
        The loop ends by shrinking wavelet coefficients, which perturbs the
        samples we actually measured -- so without this the returned image
        does not quite agree with the scanner data. Re-imposing the measured
        points costs one FFT pair and never hurts: those values are the one
        part of the reconstruction we know to be true.
    reference : array or None
        Ground-truth image. If given, PSNR is recorded per iteration so the
        convergence can be plotted. Purely diagnostic -- the algorithm never
        looks at it.

    Returns
    -------
    (image, history)
        image : float array, the reconstructed magnitude image
        history : list of {"iteration", "objective", "psnr"} dicts
    """
    mask = mask.astype(bool)

    # Start from the zero-filled reconstruction -- the Stage 1 baseline. The
    # algorithm's job is to improve on exactly this image. Kept complex; see
    # _ifft_complex for why that matters.
    current = _ifft_complex(kspace_measured * mask)

    # Scale lambda to the data. Wavelet coefficients of an image in [0, 1] are
    # O(1), but making the threshold relative to the actual largest coefficient
    # keeps the same lambda_ meaningful across different images and ratios.
    start_coeffs, slices = _wavedec(current, wavelet, level)
    threshold = lambda_ * _detail_scale(start_coeffs, slices)

    # FISTA bookkeeping.
    momentum_image = current.copy()
    t_previous = 1.0

    history: list[dict] = []

    for iteration in range(n_iter):
        # --- (a) data consistency ------------------------------------------
        # Fourier-transform the current guess, replace the measured points with
        # the true measurements, and transform back. Because F is unitary this
        # is exactly a gradient-descent step of size 1 on ||M F x - y||^2.
        predicted = to_kspace(momentum_image)
        residual = np.where(mask, kspace_measured - predicted, 0.0)
        consistent = _ifft_complex(predicted + residual)

        # --- (b) sparsity ---------------------------------------------------
        # On complex coefficients, soft_threshold shrinks the magnitude and
        # leaves the phase alone -- the correct proximal operator here.
        coeffs, slices = _wavedec(consistent, wavelet, level)
        coeffs = _threshold_details(coeffs, slices, threshold)
        updated = _waverec(coeffs, slices, wavelet, level)

        # --- FISTA momentum --------------------------------------------------
        if use_fista:
            t_current = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_previous ** 2))
            momentum_image = updated + ((t_previous - 1.0) / t_current) * (updated - current)
            t_previous = t_current
        else:
            momentum_image = updated

        current = updated

        # --- diagnostics -----------------------------------------------------
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
        # Put the measured samples back exactly as they were measured.
        predicted = to_kspace(current)
        current = _ifft_complex(np.where(mask, kspace_measured, predicted))

        # Record the result of this step too, otherwise a convergence plot
        # would stop one step short of the image actually returned and appear
        # to disagree with the reported PSNR.
        if reference is not None and history:
            mse = float(np.mean((reference - np.abs(current)) ** 2))
            history.append({
                "iteration": history[-1]["iteration"] + 1,
                "objective": history[-1]["objective"],
                "data_error": 0.0,          # measured points now match exactly
                "psnr": float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse),
            })

    # The one and only magnitude operation, matching what a scanner displays
    # and what Stage 1's zero-fill reconstruction returns.
    return np.abs(current), history


def compare_with_zero_fill(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray,
    **kwargs,
) -> dict:
    """
    Run both reconstructions on identical data and score them.

    This is the Stage 2 headline comparison: *same* mask, *same* samples, two
    different answers to the question "what was not measured?".

        zero-fill : "it was zero"          (linear, instant, artifact-ridden)
        CS        : "it was whatever makes the image sparsest while still
                     matching what we measured"   (nonlinear, iterative)

    Returns a dict with both images, both metric sets, and the CS history.
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
