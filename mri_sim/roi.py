"""
roi.py -- reduced field-of-view ("inner volume") imaging.

THE QUESTION THIS MODULE ANSWERS
--------------------------------
"We only care about the pituitary / the hypothalamus / one lesion. Can we
scan just that bit and finish in a fraction of the time?"

The answer is yes -- but not in the way almost everyone first guesses.

THE WRONG ANSWER (and it is worth showing)
------------------------------------------
The intuitive move is "the lesion is in the top-left of the image, so keep
the top-left of k-space". That is wrong, because **k-space is not spatially
local**. Every single k-space sample is an inner product of the *whole*
object with one global 2-D sinusoid, so every sample carries information
about every pixel. Delete a quadrant of k-space and the damage does not land
in one quadrant of the image -- it spreads over the entire image, roughly
evenly. `kspace_locality_demo()` measures exactly that, and it is the first
figure this module produces.

Knowing *where* the target is tells you nothing about *which k-space samples
to keep*. Position in image space is encoded in the **phase** of the samples,
not in their location in k-space.

THE RIGHT ANSWER: SHRINK THE FOV, NOT THE k-SPACE REGION
--------------------------------------------------------
Two independent Fourier relationships govern a Cartesian scan, and keeping
them apart is the whole point of this task:

    dk    = spacing between k-space samples  ->  FOV = 1 / dk
    k_max = how far out the samples go       ->  dx  = 1 / (2 * k_max)

    +---------------------+------------------+
    | change dk (spacing) | changes the FOV  |
    | change k_max (edge) | changes detail   |
    +---------------------+------------------+

Low-pass (centre-only) sampling shrinks `k_max`, so it throws away
*resolution* -- a blurry picture of the whole head. That is the wrong knob.

Reduced-FOV imaging turns the *other* knob. Use a spatially selective RF
excitation (a 2-D selective pulse, or the intersection of two slice-selective
pulses in an inner-volume sequence) so that **only a small box around the
target ever produces signal**. The object being imaged is now that small box,
so its FOV requirement is `R` times smaller, so the samples may be spaced `R`
times further apart without the object folding onto itself. `k_max` -- and
therefore resolution -- is untouched.

Sample every `R`-th point along both axes and you measure `1/R^2` of the
data: at `R = 4` that is 6.25% of k-space, a **16x** faster scan, at full
resolution inside the box.

In this simulator the RF excitation is emulated by multiplying the image by a
box **before** the forward FFT. That is a legitimate model, not a cheat:
spatially restricting the excitation is precisely what the physical pulse
does, and everything after it (FFT, decimation, inverse FFT) is the real
pipeline.

THE TWO THINGS THAT WILL BITE YOU
---------------------------------
1. `R` must divide `N` exactly. Undersampling by `R` makes the image fold
   with a period of `N/R` pixels; if that is not a whole number of pixels the
   replicas land off-grid, smear, and the reconstruction quietly degrades to
   ~40 dB instead of failing loudly. `reduced_fov_acquire` asserts it.
2. The RF suppression is not a refinement, it *is* the method. Skip it and
   the rest of the head -- which is still producing signal -- folds directly
   on top of the ROI. Measured on `brain-pituitary-1111` at R = 4: 329 dB
   with suppression, -13 dB without. `compare_roi_strategies()` puts those
   two side by side with the two other things one might try instead.

CONVENTIONS
-----------
Same as the rest of the package: every k-space array here is **centered**
(`fftshift` applied, DC at `[ny//2, nx//2]`), images are real and in `[0, 1]`.
Nothing in this module modifies Stage 1 or Stage 2 behaviour; it is only ever
reached through the `--roi` flag or the Streamlit ROI tab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .kspace import build_mask, from_kspace, sampling_ratio, to_kspace

# The demo acceleration used when nothing else is specified. 4 divides 256,
# gives a 64x64 box (large enough to hold a pituitary or a small glioma with
# margin), and samples 1/16 of k-space.
DEFAULT_REDUCTION = 4


# ---------------------------------------------------------------------------
# 1. Where the box goes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ROIBox:
    """
    The excited box, in image pixels: rows `y0 .. y0+size`, cols `x0 .. x0+size`.

    Square by construction, because the reduction factor `R` is applied to
    both axes at once (we decimate k-space in both directions).
    """

    y0: int
    x0: int
    size: int

    @property
    def slices(self) -> tuple[slice, slice]:
        """`image[box.slices]` -> the ROI crop."""
        return (slice(self.y0, self.y0 + self.size),
                slice(self.x0, self.x0 + self.size))

    @property
    def center(self) -> tuple[int, int]:
        """Centre of the box as actually placed (may differ from the request
        if the box had to be pushed inside the image border)."""
        return (self.y0 + self.size // 2, self.x0 + self.size // 2)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean array, True inside the box. Used for ROI-only metrics."""
        out = np.zeros(shape, dtype=bool)
        out[self.slices] = True
        return out


def reduction_factors(shape: tuple[int, int], min_box: int = 8) -> list[int]:
    """
    Every reduction factor that is legal for an image of this shape.

    Legal means: divides *both* axes exactly (so the aliasing period is a
    whole number of pixels -- see the module docstring), and leaves a box of
    at least `min_box` pixels, since a 2-pixel ROI is not a scan.

    On the 256x256 store this returns [2, 4, 8, 16, 32] with the default
    `min_box`; the ones actually worth demonstrating are 2, 4, 8, 16.
    """
    ny, nx = shape
    return [
        r for r in range(2, min(ny, nx) + 1)
        if ny % r == 0 and nx % r == 0 and min(ny, nx) // r >= min_box
    ]


def roi_box(shape: tuple[int, int], center: tuple[int, int], R: int) -> ROIBox:
    """
    Place the excitation box of size `N/R` around `center`, clamped to the image.

    The clamp matters: the ROI centre often sits near the edge of the head,
    and a box that hangs off the image would (a) have fewer than `(N/R)^2`
    real pixels and (b) break the "the object fits inside one aliasing
    period" guarantee that the whole method rests on. Sliding it back inside
    keeps the box full-size; `ROIBox.center` reports where it really ended up.
    """
    ny, nx = shape
    _validate_reduction(shape, R)

    size = ny // R
    y0 = min(max(0, int(center[0]) - size // 2), ny - size)
    x0 = min(max(0, int(center[1]) - size // 2), nx - size)
    return ROIBox(y0=y0, x0=x0, size=size)


def roi_center_from_mask(binary_mask: np.ndarray) -> tuple[int, int]:
    """
    Centroid of a segmentation mask, as `(y, x)` integer pixels.

    Fed by the expert tumour masks that ship with the 12 brain samples in the
    store::

        sample = KSpaceStore("data/kspace_store").load("brain-pituitary-1111")
        center = roi_center_from_mask(sample.tumor_mask)

    This is the only place the demo uses prior knowledge of *where* the
    target is -- and note what it is used for: aiming the **RF excitation**,
    not choosing k-space samples. That distinction is the lesson of this
    module.
    """
    ys, xs = np.nonzero(binary_mask)
    if ys.size == 0:
        raise ValueError("segmentation mask is empty; no ROI centre to take")
    return (int(round(float(ys.mean()))), int(round(float(xs.mean()))))


# ---------------------------------------------------------------------------
# 2. The acquisition itself
# ---------------------------------------------------------------------------


def excitation_profile(shape: tuple[int, int], box: ROIBox) -> np.ndarray:
    """
    The RF excitation profile: 1.0 inside the box, 0.0 outside.

    A real 2-D selective pulse has soft edges (a sinc-ish transition band of a
    few pixels) and imperfect suppression outside (a few percent of residual
    signal). We use the ideal box because the point being taught is the FOV /
    sample-spacing relationship, and a soft edge would blur it with a second
    effect. `compare_roi_strategies` covers the opposite extreme -- *no*
    suppression at all -- which is where the interesting failure lives.
    """
    profile = np.zeros(shape, dtype=np.float64)
    profile[box.slices] = 1.0
    return profile


def coarse_grid_mask(shape: tuple[int, int], R: int) -> np.ndarray:
    """
    Keep every `R`-th k-space sample along both axes: the reduced-FOV mask.

    This is *not* one of the Stage 1 masks. Those keep a dense region and
    throw the rest away (changing `k_max`, i.e. resolution). This one keeps
    the full extent of k-space -- the outermost samples are still measured,
    so resolution is unchanged -- and only increases the **spacing** `dk`,
    which by `FOV = 1/dk` shrinks the field of view by `R`.

    The grid is anchored on the DC sample rather than on index 0. On a 256px
    image with R dividing 256 the two are the same thing, but anchoring on DC
    is the correct statement in general: the sample at k = 0 is the one that
    must always be measured, and it also makes the compact reconstruction in
    `compact_reconstruct` land DC in the middle of the small array.
    """
    ny, nx = shape
    _validate_reduction(shape, R)

    cy, cx = ny // 2, nx // 2
    mask = np.zeros(shape, dtype=np.float64)
    mask[cy % R::R, cx % R::R] = 1.0
    return mask


def reduced_fov_acquire(
    image: np.ndarray,
    center: tuple[int, int],
    R: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate an inner-volume (reduced-FOV) scan.

    Two steps, in this order:

      1. **Excite only a box** of size `(N/R, N/R)` around `center`. Emulates
         the spatially selective RF pulse: tissue outside the box is never
         tipped, so it contributes no signal at all.
      2. **Sample every `R`-th point of k-space** in both directions. Because
         the object now fits inside a FOV that is `R` times smaller, this
         coarser spacing does not cause the object to fold onto itself, and
         `k_max` is unchanged so the resolution is the same as a full scan.

    Parameters
    ----------
    image : 2-D real array, the full object (the whole head)
    center : (y, x) pixel the ROI should be centred on
    R : integer reduction factor; must divide the image size exactly

    Returns
    -------
    (excited_object, kspace_sampled, mask)
        `excited_object` is what the scanner "sees" after the RF pulse,
        `kspace_sampled` is centered k-space with the unmeasured points
        zero-filled, and `mask` is the coarse grid itself.

    Reconstruct the result with `reduced_fov_reconstruct` (which adds the
    density compensation), or `compact_reconstruct` for the small image a
    real scanner would hand back.
    """
    _validate_reduction(image.shape, R)

    box = roi_box(image.shape, center, R)
    excited = image * excitation_profile(image.shape, box)   # <-- the RF pulse

    # Forward FFT of the *excited* object, not of the whole head: after the
    # pulse, the head outside the box genuinely is not there.
    kspace = to_kspace(excited)

    mask = coarse_grid_mask(image.shape, R)
    return excited, kspace * mask, mask


def reduced_fov_reconstruct(kspace_sampled: np.ndarray, R: int) -> np.ndarray:
    """
    Reconstruct a reduced-FOV acquisition on the original full-size grid.

    Ordinary inverse FFT, then a **density compensation** factor of `R^2`.

    Why the factor: zero-filling `1 - 1/R^2` of k-space also removes that
    fraction of the signal energy, so the inverse FFT comes back scaled by
    `1/R^2`. (Formally, decimating k-space by `R` and inverse-transforming
    gives `(1/R^2)` times the sum of image replicas spaced `N/R` apart. The
    excitation guarantees only one replica is non-zero inside the box, so
    multiplying by `R^2` restores the true intensity there.) Without it the
    ROI would be correct in structure but 16x too dark at R = 4, and PSNR
    would report a disaster that is really just a gain error.

    Outside the box the result is *not* the object -- it is the periodic
    replicas of the ROI. That is expected and harmless: nothing outside the
    box was excited, so there is no ground truth out there to get wrong.
    Score this reconstruction with `metrics.compute_metrics_in_roi`.
    """
    return from_kspace(kspace_sampled) * (R * R)


def compact_reconstruct(
    kspace_sampled: np.ndarray,
    R: int,
    box: ROIBox | None = None,
) -> np.ndarray:
    """
    The honest reconstruction: an `(N/R, N/R)` image, which is all we measured.

    A real scanner does not zero-fill back up to the full matrix. It measured
    `(N/R)^2` samples, so it inverse-transforms exactly those into a small
    image -- same resolution (`k_max` never changed), smaller field of view.
    That small image *is* the reduced-FOV scan; the full-size version above is
    a convenience for comparing against the ground truth pixel by pixel.

    Note there is no `R^2` factor here: the small transform normalises by
    `(N/R)^2` instead of `N^2`, so the amplitude comes out right on its own.

    Where the ROI lands
    -------------------
    Decimating k-space keeps the origin of the *full* FOV as the origin of the
    small FOV, so the box appears **wrapped** -- it starts at
    `(y0 mod N/R, x0 mod N/R)` in the small image and rolls around the edges.
    Passing `box` rolls that offset out. That is the discrete half of the
    Fourier shift theorem -- a circular shift in image space is a linear phase
    ramp on k-space, `f(x - a) <-> F(k) * exp(-2*pi*i*k*a/N)` -- applied after
    the transform, where it is exact and needs no sign conventions. Doing it as
    a phase ramp on k-space *before* decimation is equivalent and equally
    valid; the roll is simply harder to get wrong.

    Verified against `image[box.slices]` on `brain-pituitary-1111` at R = 4:
    64x64 output, ~329 dB.
    """
    ny, nx = kspace_sampled.shape
    _validate_reduction(kspace_sampled.shape, R)
    cy, cx = ny // 2, nx // 2

    # Same anchoring as `coarse_grid_mask`, so we pick up exactly the measured
    # points. DC lands at index (N/R)//2 of the small array, which is where
    # `ifftshift` expects it.
    k_small = kspace_sampled[cy % R::R, cx % R::R]
    small = np.abs(np.fft.ifft2(np.fft.ifftshift(k_small)))

    if box is None:
        return small
    return np.roll(small, shift=(-(box.y0 % box.size), -(box.x0 % box.size)),
                   axis=(0, 1))


# ---------------------------------------------------------------------------
# 3. Demo 1 -- k-space is not spatially local
# ---------------------------------------------------------------------------


QUADRANTS = ("top-left", "top-right", "bottom-left", "bottom-right")


def kspace_locality_demo(image: np.ndarray, quadrant: str = "top-left") -> dict:
    """
    Prove that deleting *part* of k-space does not delete *part* of the image.

    Zero one quadrant of k-space, reconstruct, and measure the mean absolute
    error in each of the four image quadrants. If k-space were spatially local
    -- if the top-left of k-space "held" the top-left of the image -- one
    number would be large and the other three would be zero. They are not: on
    `brain-pituitary-1111` the four errors come out as

        0.0388   0.0223   0.0273   0.0334

    i.e. the same order of magnitude everywhere. The mild variation tracks
    where the anatomy is bright, not where the deleted quadrant was.

    The reason: each k-space sample is `sum over all pixels of
    image[y,x] * exp(-2*pi*i*(ky*y/N + kx*x/N))`. Every pixel appears in every
    sample. Location in the image is carried by the *phase* relationships
    between samples, not by the position of samples in k-space -- which is
    exactly why "the tumour is over there, so keep that corner" cannot work,
    and why the real answer has to come from the excitation instead.

    Returns
    -------
    dict with `kspace_damaged`, `reconstruction`, `error`, `quadrant_errors`
    (a 2x2 array, `[[TL, TR], [BL, BR]]`), `quadrant_labels`, `quadrant`,
    `kept_fraction` and `metrics`.
    """
    if quadrant not in QUADRANTS:
        raise ValueError(f"quadrant must be one of {QUADRANTS}, got {quadrant!r}")

    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    rows = slice(0, ny // 2) if quadrant.startswith("top") else slice(ny // 2, ny)
    cols = slice(0, nx // 2) if quadrant.endswith("left") else slice(nx // 2, nx)

    kspace = to_kspace(image)
    damaged = kspace.copy()
    damaged[rows, cols] = 0.0

    reconstruction = from_kspace(damaged)
    error = np.abs(image - reconstruction)

    # Mean |error| per image quadrant, laid out [[TL, TR], [BL, BR]] so the
    # array reads like the picture.
    half_y, half_x = ny // 2, nx // 2
    quadrant_errors = np.array([
        [error[:half_y, :half_x].mean(), error[:half_y, half_x:].mean()],
        [error[half_y:, :half_x].mean(), error[half_y:, half_x:].mean()],
    ])

    return {
        "quadrant": quadrant,
        "kspace_damaged": damaged,
        "reconstruction": reconstruction,
        "error": error,
        "quadrant_errors": quadrant_errors,
        "quadrant_labels": np.array([["top-left", "top-right"],
                                     ["bottom-left", "bottom-right"]]),
        "kept_fraction": 0.75,  # we deleted one quadrant of four
        "metrics": metrics.compute_metrics(image, reconstruction),
    }


# ---------------------------------------------------------------------------
# 4. Demo 2 -- four ways to spend the same 1/R^2 of k-space
# ---------------------------------------------------------------------------


def compare_roi_strategies(
    image: np.ndarray,
    center: tuple[int, int],
    R: int = DEFAULT_REDUCTION,
    seed: int = 0,
) -> dict:
    """
    Four acquisitions, all using ~`1/R^2` of k-space, scored **inside the ROI**.

    Same scan time, same number of samples, four different ideas about how to
    spend them:

    ==============================  ====================================
    `reduced_fov`                   excite the box, then the coarse grid
    `undersampled`                  whole head, ordinary random mask
    `low_pass`                      whole head, keep the k-space centre
    `no_suppression`                coarse grid but **no** RF excitation
    ==============================  ====================================

    Measured on `brain-pituitary-1111` at R = 4 (6.25% of k-space, 16x):

        reduced_fov       ~ +330 dB      (exact -- error is float round-off)
        undersampled      ~  +24 dB
        low_pass          ~  +25 dB
        no_suppression    ~  -13 dB

    The last row is the headline. Without the RF excitation the coarse grid is
    a catastrophe: the object still occupies the full FOV, so the `R`-fold
    aliasing folds 15 other pieces of the head directly on top of the ROI. A
    *negative* PSNR means the error is larger than the signal. The excitation
    is not an optimisation on top of the method -- it is the method.

    Note also that `undersampled` and `low_pass` are not catastrophic, just
    mediocre. They are the honest competitors, and reduced-FOV beats them by
    300 dB because it is not approximating anything: within the box it is a
    complete, critically-sampled measurement.

    All four are scored with `metrics.compute_metrics_in_roi`, i.e. only
    inside the box. Whole-image metrics are meaningless for `reduced_fov` --
    it deliberately does not reconstruct anything outside the box.

    Returns
    -------
    dict with `box`, `roi_mask`, `R`, and `variants`: a list of dicts, each
    with `key`, `label`, `note`, `object` (what was excited), `mask`,
    `kspace`, `reconstruction`, `ratio`, `acceleration`, `psnr`, `ssim`.
    """
    image = np.asarray(image, dtype=np.float64)
    _validate_reduction(image.shape, R)

    box = roi_box(image.shape, center, R)
    roi_mask = box.mask(image.shape)
    target_ratio = 1.0 / (R * R)

    # The full-head k-space, computed once: the three "no excitation" variants
    # all draw from it and differ only in which samples they keep.
    kspace_full = to_kspace(image)

    variants = []

    # --- 1. The method ------------------------------------------------------
    excited, k_roi, grid = reduced_fov_acquire(image, center, R)
    variants.append({
        "key": "reduced_fov",
        "label": f"Reduced FOV: excite the box, sample every {R}th point",
        "note": "resolution untouched (same k_max), FOV shrunk by R",
        "object": excited,
        "mask": grid,
        "kspace": k_roi,
        "reconstruction": reduced_fov_reconstruct(k_roi, R),
    })

    # --- 2. Ordinary undersampling of the whole head ------------------------
    # Variable-density is used because it is the strongest of the three Stage 1
    # strategies at this ratio -- comparing against the weakest would be
    # rigging the demo. (Cartesian scores ~17 dB, radial ~19 dB here.)
    vd_mask = build_mask("variable_density", image.shape, target_ratio, seed=seed)
    variants.append({
        "key": "undersampled",
        "label": f"Ordinary undersampling of the whole head ({R * R}x)",
        "note": "same number of samples, spread over the whole FOV",
        "object": image,
        "mask": vd_mask,
        "kspace": kspace_full * vd_mask,
        "reconstruction": from_kspace(kspace_full * vd_mask),
    })

    # --- 3. Keep the centre instead (the other intuitive wrong answer) ------
    # This is the knob that changes k_max, so it costs resolution, not FOV.
    lp_mask = build_mask("center_only", image.shape, target_ratio)
    variants.append({
        "key": "low_pass",
        "label": "Keep the k-space centre instead (ideal low-pass)",
        "note": "shrinks k_max -> loses resolution everywhere, including the ROI",
        "object": image,
        "mask": lp_mask,
        "kspace": kspace_full * lp_mask,
        "reconstruction": from_kspace(kspace_full * lp_mask),
    })

    # --- 4. The same coarse grid with the RF excitation removed -------------
    # Identical samples to variant 1. The *only* difference is that the rest
    # of the head is still producing signal.
    variants.append({
        "key": "no_suppression",
        "label": "Same coarse grid, but no RF suppression",
        "note": "the whole head folds onto the ROI -- error exceeds signal",
        "object": image,
        "mask": grid,
        "kspace": kspace_full * grid,
        # Same density compensation as variant 1, so the comparison is of the
        # excitation and nothing else.
        "reconstruction": reduced_fov_reconstruct(kspace_full * grid, R),
    })

    for variant in variants:
        scores = metrics.compute_metrics_in_roi(
            image, variant["reconstruction"], roi_mask
        )
        variant["ratio"] = sampling_ratio(variant["mask"])
        variant["acceleration"] = 1.0 / variant["ratio"]
        variant["psnr"] = scores["psnr"]
        variant["ssim"] = scores["ssim"]

    return {
        "R": R,
        "box": box,
        "roi_mask": roi_mask,
        "target_ratio": target_ratio,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# 5. Guard rails
# ---------------------------------------------------------------------------


def _validate_reduction(shape: tuple[int, int], R: int) -> None:
    """
    Enforce "R must divide N exactly", loudly.

    Undersampling k-space by `R` makes the image alias with a period of `N/R`
    pixels. When `R` does not divide `N` that period is fractional, the
    replicas land between grid points, and instead of a clean fold you get a
    smeared one that no excitation box can avoid. It does not crash -- it just
    quietly returns ~35-40 dB where a correct run returns 330 dB. Measured on
    the 256px store:

        R = 2   256 % 2 = 0   331 dB          R = 3   256 % 3 = 1    39 dB
        R = 4   256 % 4 = 0   330 dB          R = 5   256 % 5 = 1    35 dB
        R = 8   256 % 8 = 0   329 dB          R = 6   256 % 6 = 4    41 dB

    A silent 40 dB is far more dangerous than an exception, hence this check.
    """
    if not isinstance(R, (int, np.integer)) or R < 1:
        raise ValueError(f"reduction factor R must be a positive integer, got {R!r}")

    ny, nx = shape
    if ny % R or nx % R:
        raise ValueError(
            f"reduction factor R={R} must divide the image size exactly "
            f"(got {ny}x{nx}: {ny} % {R} = {ny % R}, {nx} % {R} = {nx % R}).\n"
            f"Otherwise the aliasing period N/R is not a whole number of "
            f"pixels and the reconstruction silently degrades to ~40 dB "
            f"instead of being exact. Legal factors here: "
            f"{reduction_factors(shape)}"
        )
