"""
prepare.py -- turn a raw 2-D slice into a store-ready k-space sample.

Every sample in the store goes through exactly the same four steps, so that
any two samples are directly comparable (same size, same value range, same
k-space convention):

    1. square-crop      : centre-crop the slice to a square, so the resize
                          below does not stretch the anatomy
    2. normalise        : robust percentile scaling into [0, 1]
    3. resize           : anti-aliased resize to the store resolution (256)
    4. k-space          : multiply by a synthetic phase map, forward FFT,
                          fftshift  ->  centered complex k-space

Step 4 deserves a comment, because it is the one place where this simulation
has to *invent* something (see `synthetic_phase` below).
"""

from __future__ import annotations

import numpy as np
from skimage import filters, transform

# The k-space convention (centered, DC in the middle) is defined once, in
# mri_sim.kspace. We import it rather than re-implementing the FFT so that the
# store and the Stage 1 simulator can never drift apart.
from mri_sim.kspace import to_kspace

# Default store resolution. 256x256 keeps a full forward+inverse FFT under a
# millisecond, which is what makes a slider-driven web demo feel instant.
DEFAULT_SIZE = 256


# ---------------------------------------------------------------------------
# 1-3. Geometry and intensity normalisation
# ---------------------------------------------------------------------------


def square_crop_box(image: np.ndarray, mode: str = "center") -> tuple[int, int, int]:
    """
    Work out the square crop box as (top, left, side), without cropping.

    The box is computed once and then applied to every array belonging to a
    sample -- the image *and* its tumour mask. If the mask were cropped by its
    own separate calculation it could end up shifted by a pixel or two
    relative to the image, and an overlay would no longer line up.
    """
    if mode == "content":
        return _content_box(image)
    if mode == "center":
        height, width = image.shape
        side = min(height, width)
        return (height - side) // 2, (width - side) // 2, side
    raise ValueError(f"unknown crop mode '{mode}', expected 'center' or 'content'")


def apply_crop(array: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Apply a (top, left, side) box from :func:`square_crop_box`."""
    top, left, side = box
    return array[top:top + side, left:left + side]


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """
    Centre-crop a 2-D array to a square.

    Source slices are not all square (the abdominal volume is 320x300, for
    instance). Resizing a non-square image straight to 256x256 would squash
    the anatomy along one axis, which would make the reconstructions look
    subtly wrong. Cropping first keeps the aspect ratio honest; we lose a
    little of the field of view at the long edges, which is harmless here.
    """
    return apply_crop(image, square_crop_box(image, "center"))


def content_crop_square(image: np.ndarray) -> np.ndarray:
    """Crop to a square centred on the signal; see :func:`_content_box`."""
    return apply_crop(image, square_crop_box(image, "content"))


def _content_box(image: np.ndarray, margin: float = 0.06) -> tuple[int, int, int]:
    """
    Square crop box centred on the *signal*, not on the middle of the frame.

    Some scans do not put the anatomy in the middle. The lumbar-spine DICOMs
    are the clear case: the spine sits in the right-hand third of a 384x384
    frame and the rest is air. A plain centre crop keeps all that air, and
    because `normalize_intensity` then stretches the histogram, the empty
    region's noise gets amplified into a grey haze that dominates the image --
    and, worse, adds broadband high-frequency noise to k-space that has
    nothing to do with the anatomy.

    Method: threshold well above the noise floor, take the bounding box of
    what survives, grow it by `margin`, then expand the shorter side to make
    it square (clamped to the image). If the content already fills the frame,
    this is a no-op and behaves exactly like `center_crop_square`.
    """
    height, width = image.shape

    # Blur first. Without it, isolated noise speckles anywhere in the
    # background land inside the bounding box and drag it back out to the full
    # frame -- which silently turns this whole function into a no-op.
    smoothed = filters.gaussian(image, sigma=2.0, preserve_range=True)

    # Threshold relative to the intensity *range*, not to a pixel percentile.
    # A percentile threshold always selects a fixed fraction of pixels no
    # matter how much of the frame is actually empty, which is exactly the
    # wrong behaviour here.
    low, high = np.percentile(smoothed, [1.0, 99.0])
    if high <= low:
        return square_crop_box(image, "center")
    foreground = smoothed > (low + 0.15 * (high - low))
    if not foreground.any():
        return square_crop_box(image, "center")

    # Collapse to row and column occupancy profiles and keep the band where
    # occupancy is a meaningful fraction of its peak. A whole row of the image
    # has to be reasonably full of signal to count, so a few stray bright
    # pixels cannot stretch the box.
    def _extent(profile: np.ndarray) -> tuple[int, int]:
        keep = np.flatnonzero(profile > 0.10 * profile.max())
        return int(keep[0]), int(keep[-1]) + 1

    top, bottom = _extent(foreground.mean(axis=1))
    left, right = _extent(foreground.mean(axis=0))

    # Grow the box a little so we do not shave off the edge of the anatomy.
    pad_y = int(round((bottom - top) * margin))
    pad_x = int(round((right - left) * margin))
    top, bottom = max(top - pad_y, 0), min(bottom + pad_y, height)
    left, right = max(left - pad_x, 0), min(right + pad_x, width)

    # Make it square by growing the shorter side around its own centre, then
    # sliding the box back inside the image if it overhangs an edge.
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
    """
    Scale an image into [0, 1] using percentile clipping.

    Why percentiles instead of plain (x - min) / (max - min):

    * CT volumes contain a handful of extremely bright voxels (metal, bone,
      the scanner table). A single outlier voxel would compress all the soft
      tissue into the bottom few percent of the range and the image would look
      black.
    * MRI slices often have a few hot pixels from fat or flow artifacts, with
      the same effect.

    Clipping at the 0.5th / 99.5th percentile throws away that top and bottom
    half-percent and lets the tissue of interest use the full display range.

    The [0, 1] range is not cosmetic: PSNR needs a known data range, and
    mri_sim.metrics assumes data_range = 1.0 everywhere. Forcing every sample
    onto the same range is what makes PSNR numbers comparable across samples.
    """
    image = image.astype(np.float64)

    lo = float(np.percentile(image, p_low))
    hi = float(np.percentile(image, p_high))

    if hi <= lo:                                   # constant / degenerate slice
        return np.zeros_like(image)

    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo)


def resize_square(image: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """
    Resize a square image to (size, size).

    anti_aliasing=True low-pass filters before downsampling. That matters more
    than usual in this project: without it, downsampling would itself alias
    high spatial frequencies into the image, and we would then be measuring
    *our own* aliasing artifacts on top of the undersampling artifacts we are
    actually trying to demonstrate.
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
    """Run steps 1-3 in order: crop -> normalise -> resize -> clip.

    Parameters
    ----------
    crop : "center" | "content"
        "content" re-centres the crop on the signal; see
        :func:`content_crop_square`. Used for the off-centre spine DICOMs.

    The final clip is there because the anti-aliased resize can nudge values
    a hair outside [0, 1] (interpolation overshoot near sharp edges).
    Clipping guarantees the stored image really is in [0, 1], which is what
    PSNR's data_range=1.0 assumes.
    """
    image = np.asarray(image, dtype=np.float64)
    image = apply_crop(image, square_crop_box(image, crop))
    image = normalize_intensity(image)
    image = resize_square(image, size)
    return np.clip(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Phase and k-space
# ---------------------------------------------------------------------------


def synthetic_phase(
    shape: tuple[int, int],
    seed: int,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Build a smooth, slowly-varying phase map in radians.

    WHY WE NEED THIS AT ALL
    -----------------------
    A real scanner measures a *complex* signal. The image it encodes is
    complex too: its magnitude is the anatomy you look at, and its phase comes
    from B0 field inhomogeneity, coil sensitivities, chemical shift and flow.
    Our source data are magnitude images -- the phase was thrown away long
    before the files were saved.

    If we fed a purely real image into the FFT, its k-space would be exactly
    Hermitian-symmetric: K(-k) = conj(K(k)). Half of k-space would be a free
    copy of the other half, and any demo of partial-Fourier sampling, phase
    correction or "half the data is redundant" would be trivially,
    unrealistically perfect. Adding a plausible phase map removes that
    artificial symmetry and makes the simulated k-space behave like real data.

    THE MODEL
    ---------
    Three physically-motivated, smooth terms:

    * a quadratic bowl        -- stands in for B0 field inhomogeneity, which
                                 is smooth and roughly bowl-shaped over the
                                 field of view
    * a linear ramp           -- a small spatial offset / gradient timing
                                 error; a linear phase ramp in image space is
                                 a shift in k-space
    * a few low-order ripples -- stands in for receive-coil sensitivity phase

    Every term is deliberately *low spatial frequency*. Real image phase is
    smooth, and a smooth phase means the extra k-space energy it introduces
    stays near the centre instead of faking high-frequency detail that the
    anatomy does not have.

    Parameters
    ----------
    shape : (ny, nx)
    seed : int
        Fixes the random coefficients so a given sample always gets the same
        phase map. Reproducibility matters: the store must be rebuildable.
    strength : float
        Overall scale. 1.0 gives a peak-to-peak swing of roughly a couple of
        radians, which is typical of a real scan. 0.0 gives a real-valued
        image (Hermitian k-space) if you want to demonstrate the difference.

    Returns
    -------
    float64 array of the same shape, in radians.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    # Normalised coordinates in [-1, 1] -- resolution-independent.
    y = np.linspace(-1.0, 1.0, ny).reshape(-1, 1)
    x = np.linspace(-1.0, 1.0, nx).reshape(1, -1)

    # B0-like quadratic bowl, off-centre by a random amount.
    y0, x0 = rng.uniform(-0.3, 0.3, size=2)
    phase = rng.uniform(0.8, 1.6) * ((y - y0) ** 2 + (x - x0) ** 2)

    # Linear ramp (gradient/shift term).
    phase += rng.uniform(-0.6, 0.6) * y + rng.uniform(-0.6, 0.6) * x

    # Coil-sensitivity-like ripples: two or three full cycles at most.
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
    """
    Forward step: magnitude image (+ synthetic phase) -> centered k-space.

    Returns
    -------
    (kspace, phase)
        kspace : complex, fftshifted so DC sits at the middle of the array
        phase  : the phase map that was used, in radians

    The stored k-space is therefore the FFT of  image * exp(i * phase).
    A consumer that wants the original magnitude image back just does an
    inverse FFT and takes the absolute value -- which is exactly what the
    Stage 1 reconstruction does, so the round trip is exact to float
    precision.
    """
    phase = synthetic_phase(image.shape, seed=seed, strength=phase_strength)
    complex_image = image * np.exp(1j * phase)
    return to_kspace(complex_image), phase


# ---------------------------------------------------------------------------
# Derived statistics -- stored in the manifest, useful for teaching
# ---------------------------------------------------------------------------


def center_energy_fraction(kspace: np.ndarray, radius_fraction: float) -> float:
    """
    Fraction of total k-space energy inside a central disc.

    This single number is the quantitative version of the whole "the centre of
    k-space matters most" argument. For a typical MRI slice, a disc covering
    the central 10% of the k-space radius -- about 1% of the samples -- holds
    well over 90% of the signal energy. That is *why* every sampling mask in
    this project keeps a fully-sampled centre, and why an image reconstructed
    from the centre alone still looks like the right organ (just blurry).

    Parameters
    ----------
    kspace : centered complex k-space
    radius_fraction : float in (0, 1]
        Disc radius as a fraction of half the array size.
    """
    ny, nx = kspace.shape
    cy, cx = ny // 2, nx // 2

    y = np.arange(ny).reshape(-1, 1) - cy
    x = np.arange(nx).reshape(1, -1) - cx
    # Normalise by half the array size so radius_fraction=1.0 is the inscribed
    # circle touching the edges of k-space.
    radius = np.sqrt((y / (ny / 2)) ** 2 + (x / (nx / 2)) ** 2)

    energy = np.abs(kspace) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    return float(energy[radius <= radius_fraction].sum() / total)
