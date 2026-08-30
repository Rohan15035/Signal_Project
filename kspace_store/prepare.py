"""Turn a raw 2-D slice into a store-ready k-space sample.

Every sample takes the same four steps, so any two are directly comparable:
square-crop (so the resize does not stretch the anatomy), robust percentile
normalisation into [0, 1], anti-aliased resize to 256, then a synthetic phase
map and forward FFT into centered k-space.

Step 4 is the one place this simulation has to invent something -- see
synthetic_phase.
"""

from __future__ import annotations

import numpy as np
from skimage import filters, transform

# Imported rather than reimplemented so the store and the simulator can never
# drift apart on the k-space convention.
from mri_sim.kspace import to_kspace

# 256x256 keeps a full FFT round trip under a millisecond, which is what makes
# a slider-driven demo feel instant.
DEFAULT_SIZE = 256


def square_crop_box(image: np.ndarray, mode: str = "center") -> tuple[int, int, int]:
    """Work out the square crop box as (top, left, side), without cropping.

    Computed once and applied to every array in a sample, image and tumour
    mask alike -- cropping the mask by its own calculation could shift it a
    pixel or two and break the overlay.
    """
    if mode == "content":
        return _content_box(image)
    if mode == "center":
        height, width = image.shape
        side = min(height, width)
        return (height - side) // 2, (width - side) // 2, side
    raise ValueError(f"unknown crop mode '{mode}', expected 'center' or 'content'")


def apply_crop(array: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Apply a (top, left, side) box from square_crop_box."""
    top, left, side = box
    return array[top:top + side, left:left + side]


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """Centre-crop to a square.

    Source slices are not all square (the abdominal volume is 320x300), and
    resizing one straight to 256x256 would squash the anatomy along one axis.
    Cropping first costs a little field of view at the long edges instead.
    """
    return apply_crop(image, square_crop_box(image, "center"))


def content_crop_square(image: np.ndarray) -> np.ndarray:
    """Crop to a square centred on the signal; see _content_box."""
    return apply_crop(image, square_crop_box(image, "content"))


def _content_box(image: np.ndarray, margin: float = 0.06) -> tuple[int, int, int]:
    """Square crop box centred on the signal rather than on the frame.

    The lumbar-spine DICOMs put the spine in the right-hand third of a 384x384
    frame. A plain centre crop keeps all that air, and since normalize_intensity
    then stretches the histogram, the empty region's noise is amplified into a
    grey haze -- and into broadband high-frequency k-space content that has
    nothing to do with the anatomy.

    Threshold above the noise floor, bound what survives, grow by margin, then
    square it off. If the content already fills the frame this is a no-op.
    """
    height, width = image.shape

    # Blur first, or isolated background speckles land in the bounding box and
    # drag it back out to the full frame, silently making this a no-op.
    smoothed = filters.gaussian(image, sigma=2.0, preserve_range=True)

    # Relative to the intensity range, not a pixel percentile: a percentile
    # always selects a fixed fraction of pixels however empty the frame is,
    # which is exactly wrong here.
    low, high = np.percentile(smoothed, [1.0, 99.0])
    if high <= low:
        return square_crop_box(image, "center")
    foreground = smoothed > (low + 0.15 * (high - low))
    if not foreground.any():
        return square_crop_box(image, "center")

    # Keep the band where row/column occupancy is a real fraction of its peak,
    # so a few stray bright pixels cannot stretch the box.
    def _extent(profile: np.ndarray) -> tuple[int, int]:
        keep = np.flatnonzero(profile > 0.10 * profile.max())
        return int(keep[0]), int(keep[-1]) + 1

    top, bottom = _extent(foreground.mean(axis=1))
    left, right = _extent(foreground.mean(axis=0))

    pad_y = int(round((bottom - top) * margin))
    pad_x = int(round((right - left) * margin))
    top, bottom = max(top - pad_y, 0), min(bottom + pad_y, height)
    left, right = max(left - pad_x, 0), min(right + pad_x, width)

    # Grow the shorter side around its centre, then slide back inside the image.
    side = min(max(bottom - top, right - left), height, width)
    center_y = (top + bottom) // 2
    center_x = (left + right) // 2
    top = int(np.clip(center_y - side // 2, 0, height - side))
    left = int(np.clip(center_x - side // 2, 0, width - side))

    return top, left, side


def normalize_intensity(
    image: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> np.ndarray:
    """Scale into [0, 1] by percentile clipping rather than min/max.

    CT volumes have a few extremely bright voxels (metal, bone, the table) and
    MRI slices have hot pixels from fat or flow; one outlier would compress all
    the soft tissue into the bottom few percent and the image would look black.

    The [0, 1] range is load-bearing, not cosmetic: mri_sim.metrics assumes
    data_range = 1.0, so this is what makes PSNR comparable across samples.
    """
    image = image.astype(np.float64)

    lo = float(np.percentile(image, p_low))
    hi = float(np.percentile(image, p_high))

    if hi <= lo:                                   # constant / degenerate slice
        return np.zeros_like(image)

    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo)


def resize_square(image: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Resize a square image to (size, size), anti-aliased.

    Matters more than usual here: without the low-pass, downsampling would
    alias high frequencies into the image and we would be measuring our own
    aliasing on top of the undersampling artifacts we mean to demonstrate.
    """
    if image.shape == (size, size):
        return image
    return transform.resize(
        image, (size, size), anti_aliasing=True, preserve_range=True
    ).astype(np.float64)


def to_store_image(
    image: np.ndarray,
    size: int = DEFAULT_SIZE,
    crop: str = "center",
) -> np.ndarray:
    """Crop -> normalise -> resize -> clip.

    crop="content" re-centres on the signal, used for the off-centre spine
    DICOMs. The final clip catches interpolation overshoot from the resize,
    which can nudge values just outside [0, 1] near sharp edges.
    """
    image = np.asarray(image, dtype=np.float64)
    image = apply_crop(image, square_crop_box(image, crop))
    image = normalize_intensity(image)
    image = resize_square(image, size)
    return np.clip(image, 0.0, 1.0)


def synthetic_phase(
    shape: tuple[int, int],
    seed: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Smooth, slowly-varying phase map in radians.

    A real scanner measures a complex signal whose phase comes from B0
    inhomogeneity, coil sensitivities, chemical shift and flow. Our sources are
    magnitude images -- the phase was discarded before the files were saved.

    That matters: a purely real image has exactly Hermitian k-space,
    K(-k) = conj(K(k)), so half of it would be a free copy of the other half
    and any partial-Fourier demo would be unrealistically perfect. A plausible
    phase map removes that artificial symmetry.

    Three smooth terms: a quadratic bowl for B0 inhomogeneity, a linear ramp
    for gradient timing error (a ramp in image space is a shift in k-space),
    and a few low-order ripples for coil sensitivity. All deliberately low
    frequency, so the energy they add stays near the k-space centre instead of
    faking detail the anatomy does not have.

    seed fixes the coefficients so the store is rebuildable. strength 1.0 gives
    a couple of radians peak-to-peak, typical of a real scan; 0.0 gives a real
    image with Hermitian k-space if you want to show the difference.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    # Normalised coordinates in [-1, 1] -- resolution-independent.
    y = np.linspace(-1.0, 1.0, ny).reshape(-1, 1)
    x = np.linspace(-1.0, 1.0, nx).reshape(1, -1)

    # B0-like bowl, off-centre by a random amount.
    y0, x0 = rng.uniform(-0.3, 0.3, size=2)
    phase = rng.uniform(0.8, 1.6) * ((y - y0) ** 2 + (x - x0) ** 2)

    phase += rng.uniform(-0.6, 0.6) * y + rng.uniform(-0.6, 0.6) * x

    # Coil-like ripples: two or three full cycles at most.
    for _ in range(3):
        fy, fx = rng.uniform(0.5, 2.0, size=2)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        phase += rng.uniform(0.1, 0.35) * np.sin(np.pi * (fy * y + fx * x) + phi)

    return strength * phase


def build_kspace(
    image: np.ndarray,
    seed: int,
    phase_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude image (+ synthetic phase) -> (centered k-space, phase in radians).

    The stored k-space is the FFT of image * exp(i * phase), so an inverse FFT
    and abs() returns the original magnitude image exactly to float precision
    -- which is what the Stage 1 reconstruction does.
    """
    phase = synthetic_phase(image.shape, seed=seed, strength=phase_strength)
    complex_image = image * np.exp(1j * phase)
    return to_kspace(complex_image), phase


def center_energy_fraction(kspace: np.ndarray, radius_fraction: float) -> float:
    """Fraction of total k-space energy inside a central disc.

    The quantitative version of the whole protect-the-centre argument: for a
    typical slice, a disc covering the central 10% of the radius -- about 1% of
    the samples -- holds well over 90% of the energy. radius_fraction is
    measured against half the array size, so 1.0 is the inscribed circle.
    """
    ny, nx = kspace.shape
    cy, cx = ny // 2, nx // 2

    y = np.arange(ny).reshape(-1, 1) - cy
    x = np.arange(nx).reshape(1, -1) - cx
    radius = np.sqrt((y / (ny / 2)) ** 2 + (x / (nx / 2)) ** 2)

    energy = np.abs(kspace) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    return float(energy[radius <= radius_fraction].sum() / total)
