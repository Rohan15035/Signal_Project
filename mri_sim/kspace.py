"""
kspace.py -- the heart of the simulator.

Contains four things:

1. The forward step   : image  -> centered k-space          (`to_kspace`)
2. Sampling masks     : which k-space points the scanner bothers to measure
                        (`cartesian_mask`, `radial_mask`, `variable_density_mask`)
3. Masking            : zero-fill everything not sampled    (`apply_mask`)
4. The inverse step   : centered k-space -> image magnitude (`from_kspace`)

CONVENTION USED THROUGHOUT THIS FILE
------------------------------------
Every k-space array that crosses a function boundary in this package is
**centered**, i.e. `fftshift` has already been applied so that the DC term
(zero spatial frequency) sits at pixel [ny//2, nx//2] instead of at [0, 0].

Why: numpy's `fft2` puts DC in the corner and wraps the negative frequencies
around to the far edges. That layout is awkward both to look at and to write
masks for. The MRI convention is DC in the middle, low frequencies near the
middle, high frequencies out toward the edges -- which is what `fftshift`
gives us. Because every array is centered, all the mask functions can simply
reason in terms of "distance from the middle of the array".

The single place where this convention is undone is `from_kspace`, which
applies `ifftshift` right before `ifft2` to hand numpy back the corner-DC
layout it expects.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. Forward and inverse transforms
# ---------------------------------------------------------------------------


def to_kspace(image: np.ndarray) -> np.ndarray:
    """
    Forward step: turn an image into synthetic (centered) k-space data.

    This is the step that a real scanner does *not* do -- a real scanner
    measures k-space directly. We fake it so that we have ground truth to
    compare our reconstructions against.

    Parameters
    ----------
    image : 2-D real array

    Returns
    -------
    2-D complex array, fftshifted so DC is at the center.
    """
    # fft2  : the 2-D discrete Fourier transform. Output has DC at [0, 0].
    # fftshift: roll the array by half its size along both axes so DC moves
    #           to the middle. Purely a re-indexing -- no information changes.
    return np.fft.fftshift(np.fft.fft2(image))


def from_kspace(kspace_centered: np.ndarray) -> np.ndarray:
    """
    Inverse step: reconstruct an image from centered k-space.

    Three sub-steps, and the order matters:

    1. `ifftshift` -- undo the centering, putting DC back at [0, 0] where
       numpy's `ifft2` expects it. (Note: `ifftshift`, not `fftshift`. They
       are identical for even-sized arrays but differ for odd sizes, and
       using the wrong one there produces a subtly shifted image.)
    2. `ifft2`     -- the inverse 2-D DFT.
    3. `np.abs`    -- take the magnitude.

    Why magnitude? The inverse FFT returns a complex image. For fully-sampled
    data the imaginary part is ~0 (numerical noise) because our input was
    real. But once we zero out parts of k-space, the masked data is no longer
    the transform of a real image, so the result genuinely has an imaginary
    component. Real MRI has the same issue for physical reasons (B0
    inhomogeneity, coil phase), and real scanners solve it exactly this way:
    they display the magnitude image.

    Returns
    -------
    2-D real, non-negative float array.
    """
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_centered)))


def apply_mask(kspace_centered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a sampling mask to k-space by zero-filling everything not sampled.

    This models an accelerated scan: the scanner only spends time acquiring
    the k-space points where mask == 1, and the reconstruction simply assumes
    the unmeasured points are zero. That assumption is wrong, and the way in
    which it is wrong is exactly what produces the artifacts we are studying.

    Note we keep the array the same size rather than deleting rows -- an
    array of measured samples plus zeros is what the inverse FFT needs.
    """
    if kspace_centered.shape != mask.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match k-space shape "
            f"{kspace_centered.shape}"
        )
    return kspace_centered * mask


def reconstruct(kspace_centered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper: mask the k-space, then inverse FFT it.

    This is the whole "zero-filled reconstruction" in one line, and it is the
    baseline every fancier method (e.g. compressed sensing) is measured
    against.
    """
    return from_kspace(apply_mask(kspace_centered, mask))


# ---------------------------------------------------------------------------
# 2. Helpers shared by the mask builders
# ---------------------------------------------------------------------------


def sampling_ratio(mask: np.ndarray) -> float:
    """Fraction of k-space points actually sampled (1.0 = fully sampled)."""
    return float(np.count_nonzero(mask)) / mask.size


def acceleration_factor(mask: np.ndarray) -> float:
    """
    Scan-time speed-up implied by a mask, i.e. 1 / sampling_ratio.

    A ratio of 0.25 means a 4x acceleration: the scan takes a quarter of the
    time because only a quarter of the samples are acquired.
    """
    ratio = sampling_ratio(mask)
    return float("inf") if ratio == 0 else 1.0 / ratio


def _center_index(n: int) -> int:
    """
    Index of the DC ("zero frequency") position along an axis of length n,
    for an fftshift-centered array. This is n // 2 for both even and odd n,
    matching numpy's fftshift definition.
    """
    return n // 2


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    """
    Normalized distance-from-center map for a centered k-space array.

    Returns an array the same shape as k-space where 0.0 is the DC point and
    1.0 is roughly the edge of the inscribed circle. Corners exceed 1.0.
    Used by the variable-density mask to decide sampling probability.
    """
    ny, nx = shape
    cy, cx = _center_index(ny), _center_index(nx)

    # Distances in pixels from the center along each axis...
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx

    # ...normalized by the half-width of each axis, so that a non-square
    # image still gets a sensible circular (rather than stretched) profile.
    y = y / max(cy, 1)
    x = x / max(cx, 1)

    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.sqrt(yy**2 + xx**2)


def _full_mask(shape: tuple[int, int]) -> np.ndarray:
    """A fully-sampled mask (all ones) -- the 100% / reference case."""
    return np.ones(shape, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. The three sampling strategies
# ---------------------------------------------------------------------------


def cartesian_mask(
    shape: tuple[int, int],
    ratio: float,
    center_fraction: float | None = None,
) -> np.ndarray:
    """
    Cartesian (regular line-skipping) undersampling.

    This is the classic accelerated MRI acquisition. k-space is filled one
    horizontal line at a time; each line is one "phase-encoding step" and
    costs one repetition of the pulse sequence, so scan time is essentially
    proportional to the number of lines. Skipping every Nth line therefore
    gives a genuine Nx speed-up. Sampling is uniform *along* each line (the
    readout direction is free), which is why we keep or drop whole rows
    rather than individual points.

    Two-part mask:

      * A fully-sampled block of lines around the k-space center.
      * Every Nth line elsewhere.

    WHY THE CENTER BLOCK IS NON-NEGOTIABLE:
    k-space is the Fourier transform of the image, so the *center* of k-space
    holds the low spatial frequencies: overall brightness, contrast, and the
    coarse shape of the object. Nearly all of the image's energy lives there.
    The outer regions hold high spatial frequencies: edges and fine detail.
    If you undersample the center you lose the image's basic contrast and
    structure and the reconstruction is unusable. If you undersample the
    edges you lose sharpness but still get a recognizable image. So every
    practical undersampling scheme -- and every one in this project --
    fully samples a central block and spends its remaining budget on the
    periphery.

    Parameters
    ----------
    shape : (ny, nx)
    ratio : float in (0, 1]
        Target fraction of k-space to sample. 1.0 returns a full mask.
    center_fraction : float or None
        Fraction of all lines to keep fully sampled at the center. Defaults
        to 0.32 * ratio, the convention used by the fastMRI benchmark
        (8% of lines at 4x acceleration, 4% at 8x). Roughly a third of the
        sampling budget is spent on the center.

    Returns
    -------
    Float mask of `shape` containing 0.0 / 1.0.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    ny, nx = shape
    if center_fraction is None:
        center_fraction = 0.32 * ratio

    # Total number of lines we are allowed to acquire.
    n_target = max(1, int(round(ratio * ny)))

    # --- Part 1: the fully-sampled center block -----------------------------
    n_center = max(1, int(round(center_fraction * ny)))
    n_center = min(n_center, n_target)  # never overspend the budget

    cy = _center_index(ny)
    start = cy - n_center // 2
    center_lines = np.arange(start, start + n_center)
    center_lines = center_lines[(center_lines >= 0) & (center_lines < ny)]

    mask = np.zeros((ny, nx), dtype=np.float64)
    mask[center_lines, :] = 1.0

    # --- Part 2: every Nth line in the remaining (outer) region -------------
    n_outer = n_target - len(center_lines)
    if n_outer > 0:
        outer_lines = np.setdiff1d(np.arange(ny), center_lines)
        # Uniform stride over the outer lines. stride is the "N" in
        # "keep every Nth line"; it is chosen so that the center block plus
        # the outer lines hits the sampling budget exactly.
        stride = len(outer_lines) / n_outer
        picks = np.unique(np.floor(np.arange(n_outer) * stride).astype(int))
        mask[outer_lines[picks], :] = 1.0

    return mask


def radial_mask(
    shape: tuple[int, int],
    ratio: float | None = None,
    n_spokes: int | None = None,
) -> np.ndarray:
    """
    Radial ("spoke" / projection) sampling.

    Instead of horizontal lines, the scanner traverses k-space along straight
    lines that all pass through the center, like spokes of a wheel. Real
    scanners do this with non-Cartesian gradient waveforms; it is the basis
    of PROPELLER and of most fast dynamic/abdominal imaging.

    Two properties make it attractive:

      * Every spoke passes through the k-space center, so the center is
        heavily oversampled "for free" -- the low frequencies that carry
        contrast are always well measured (same principle as the Cartesian
        center block, but automatic).
      * Its undersampling artifacts are incoherent streaks rather than the
        coherent ghosts/replicas produced by regular Cartesian skipping.
        Streaks look like noise and are visually less destructive; they are
        also what makes radial data friendly to compressed sensing.

    Implementation note: this is a *rasterized* approximation. A real radial
    trajectory lands on off-grid points and needs gridding/NUFFT to
    reconstruct. Here we snap each point along a spoke to the nearest
    Cartesian grid point, so we can keep using the plain inverse FFT. That is
    a simplification, but it reproduces the sampling pattern and its
    characteristic artifacts faithfully.

    Parameters
    ----------
    shape : (ny, nx)
    ratio : float in (0, 1], optional
        Target fraction of k-space to cover. The number of spokes needed is
        found by a binary search. Note that the *achieved* ratio is usually
        slightly above the target (spokes overlap near the center in ways
        that are hard to predict exactly), and that the far corners of
        k-space can never be reached by spokes -- so a target of 1.0 is
        special-cased to a full mask for use as the reference.
    n_spokes : int, optional
        Give this instead of `ratio` to specify the spoke count directly.

    Returns
    -------
    Float mask of `shape` containing 0.0 / 1.0.
    """
    if (ratio is None) == (n_spokes is None):
        raise ValueError("give exactly one of `ratio` or `n_spokes`")

    if n_spokes is not None:
        return _draw_spokes(shape, n_spokes)

    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    # Binary search for the smallest spoke count that reaches the target
    # coverage. Coverage grows monotonically with spoke count, so bisection
    # is safe. The upper bound is generous: 4x the image width is far more
    # spokes than are ever needed for near-full coverage.
    lo, hi = 1, 4 * max(shape)
    best = _draw_spokes(shape, hi)
    if sampling_ratio(best) < ratio:
        return best  # target unreachable with spokes; return the densest one

    while lo < hi:
        mid = (lo + hi) // 2
        candidate = _draw_spokes(shape, mid)
        if sampling_ratio(candidate) >= ratio:
            hi, best = mid, candidate
        else:
            lo = mid + 1

    return best


def _draw_spokes(shape: tuple[int, int], n_spokes: int) -> np.ndarray:
    """
    Rasterize `n_spokes` lines through the k-space center onto the grid.

    Angles are spread evenly over [0, pi) rather than [0, 2*pi) because a
    spoke and its 180-degree rotation trace the same set of grid points --
    using the full circle would just draw every line twice.
    """
    ny, nx = shape
    cy, cx = _center_index(ny), _center_index(nx)
    mask = np.zeros((ny, nx), dtype=np.float64)

    # Half-diagonal: how far a spoke must run to reach the array corners.
    max_radius = float(np.hypot(ny, nx)) / 2.0

    # Step of 0.5 px along each spoke so that after rounding to the nearest
    # grid point we never leave a gap in the drawn line.
    n_steps = int(np.ceil(2 * max_radius / 0.5)) + 1
    t = np.linspace(-max_radius, max_radius, n_steps)

    for angle in np.linspace(0.0, np.pi, n_spokes, endpoint=False):
        # Parametric line through the center at this angle.
        yy = np.rint(cy + t * np.sin(angle)).astype(int)
        xx = np.rint(cx + t * np.cos(angle)).astype(int)

        # Drop the parts of the spoke that fall outside the array.
        inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        mask[yy[inside], xx[inside]] = 1.0

    return mask


def variable_density_mask(
    shape: tuple[int, int],
    ratio: float,
    poly_order: float = 6.0,
    center_radius: float = 0.05,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Random variable-density undersampling.

    Each k-space point is kept or dropped by an independent coin flip, but
    the coin is biased by distance from the center: points near the center
    are almost certain to be kept, points near the edges are unlikely to be.
    The probability profile is

        p(r)  =  alpha * (1 - r)^poly_order,   clipped to [0, 1]

    where r is the normalized distance from the k-space center and `alpha` is
    solved for numerically so that the *average* probability equals the
    requested sampling `ratio` (i.e. so we sample the right number of points
    on average). A small disk at the very center is forced to p = 1.

    Two reasons this strategy exists:

      * Same as always -- the center carries the image energy, so it must be
        densely sampled while the sparse periphery only costs us fine detail.
      * *Randomness* is the key ingredient for compressed sensing (Stage 2).
        Regular Cartesian skipping makes aliases that are coherent: the
        missing data folds the image onto itself as crisp ghost copies, which
        are indistinguishable from real image structure. Random sampling
        instead spreads the error out as low-level incoherent noise across
        the whole image. Noise-like error can be removed by a sparsity-
        promoting denoiser; coherent ghosts cannot. That is precisely why
        compressed-sensing MRI uses random variable-density masks.

    Parameters
    ----------
    shape : (ny, nx)
    ratio : float in (0, 1]
        Target fraction of k-space to sample.
    poly_order : float
        How aggressively density falls off toward the edges. Larger = more
        concentrated at the center.
    center_radius : float
        Normalized radius of the always-sampled central disk.
    seed : int or None
        Seed for reproducibility. Pass None for a different draw each run.

    Returns
    -------
    Float mask of `shape` containing 0.0 / 1.0.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    radius = _radius_grid(shape)

    # Unscaled density profile: 1.0 at the center, falling to 0 at r = 1.
    # np.clip keeps the corners (r > 1) from going negative.
    profile = np.clip(1.0 - radius, 0.0, 1.0) ** poly_order

    # The always-sampled central disk.
    center_disk = radius <= center_radius

    # Solve for the scale factor `alpha` that makes the mean probability
    # equal `ratio`. We cannot do this in closed form because of the clipping
    # at 1.0, so we bisect. mean_prob(alpha) is monotonically increasing in
    # alpha, which makes bisection reliable.
    def mean_prob(alpha: float) -> float:
        p = np.clip(alpha * profile, 0.0, 1.0)
        p[center_disk] = 1.0
        return float(p.mean())

    lo, hi = 0.0, 1.0
    while mean_prob(hi) < ratio:      # grow the bracket until it contains the answer
        hi *= 2.0
        if hi > 1e9:                  # ratio unreachable (center disk alone exceeds it)
            break

    for _ in range(100):              # ~100 bisections is far more precision than needed
        mid = 0.5 * (lo + hi)
        if mean_prob(mid) < ratio:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)

    prob = np.clip(alpha * profile, 0.0, 1.0)
    prob[center_disk] = 1.0

    rng = np.random.default_rng(seed)
    return (rng.random(shape) < prob).astype(np.float64)


# ---------------------------------------------------------------------------
# 4. Stage 2 teaching masks: centre-only and edges-only
# ---------------------------------------------------------------------------
#
# These two are not realistic acquisition strategies -- no scanner would use
# them. They exist to make one claim visible in a single pair of pictures:
#
#     the CENTRE of k-space carries contrast and overall shape,
#     the EDGES carry fine detail and sharp boundaries.
#
# Keep only the centre and you get the right organ, correctly bright and
# correctly shaped, but blurred -- a low-pass filter. Keep only the edges and
# the organ vanishes, leaving a dark image with bright outlines where the
# boundaries were -- a high-pass filter, i.e. an edge detector.
#
# Both take the same `ratio` argument as the real strategies, meaning "keep
# this fraction of the k-space points", so they slot into the same dispatch
# table and the same UI slider. The radius that achieves a given ratio is
# found numerically rather than by the area formula pi*r^2/4, because the
# corners of a square k-space are outside the inscribed circle and the exact
# pixel count depends on the grid.


def _radius_for_ratio(radius_map: np.ndarray, ratio: float) -> float:
    """
    Smallest radius whose enclosed disc contains `ratio` of all points.

    Found by sorting the radii and reading off a quantile -- exact, and
    cheaper than bisecting on a count.
    """
    flat = np.sort(radius_map, axis=None)
    index = int(np.clip(round(ratio * (flat.size - 1)), 0, flat.size - 1))
    return float(flat[index])


def center_only_mask(shape: tuple[int, int], ratio: float) -> np.ndarray:
    """
    Keep only a central disc of k-space: an ideal circular **low-pass** filter.

    The reconstruction shows what the low spatial frequencies alone encode --
    contrast, brightness and gross shape, with edges smeared out. Because the
    disc has a hard rim, expect faint concentric ringing around sharp
    boundaries: that is Gibbs ringing, the 2-D echo of the same overshoot you
    get when you truncate a 1-D Fourier series at a square edge.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    radius = _radius_grid(shape)
    return (radius <= _radius_for_ratio(radius, ratio)).astype(np.float64)


def edges_only_mask(shape: tuple[int, int], ratio: float) -> np.ndarray:
    """
    Keep only the outer annulus of k-space: an ideal **high-pass** filter.

    The exact complement of :func:`center_only_mask` at ratio `1 - ratio`. The
    reconstruction is mostly black with bright edges, because discarding the
    centre removes the DC term -- the mean brightness of the image -- along
    with everything else that describes slowly-varying tissue.

    This is the picture to put next to the centre-only one. It is also why
    every practical mask in section 3 protects the centre: those samples are
    where essentially all of the energy is (a mean of ~93% inside the central
    10% radius, measured across the sample store).
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    radius = _radius_grid(shape)
    # Threshold at the radius that leaves `ratio` of the points *outside* it.
    return (radius > _radius_for_ratio(radius, 1.0 - ratio)).astype(np.float64)


# ---------------------------------------------------------------------------
# 5. Dispatch table, so main.py can loop over strategies by name
# ---------------------------------------------------------------------------

MASK_BUILDERS = {
    "cartesian": cartesian_mask,
    "radial": radial_mask,
    "variable_density": variable_density_mask,
    # Stage 2 demonstration masks -- deliberately not realistic acquisitions.
    "center_only": center_only_mask,
    "edges_only": edges_only_mask,
}

MASK_LABELS = {
    "cartesian": "Cartesian (every Nth line + center block)",
    "radial": "Radial (spokes through center)",
    "variable_density": "Random variable-density",
    "center_only": "Center only (ideal low-pass)",
    "edges_only": "Edges only (ideal high-pass)",
}

# The three that model a real accelerated scan. main.py sweeps over these;
# the two teaching masks above are opt-in via --center-edges.
ACQUISITION_MASKS = ["cartesian", "radial", "variable_density"]


def build_mask(kind: str, shape: tuple[int, int], ratio: float, **kwargs) -> np.ndarray:
    """
    Build a mask by name. `kind` is one of the keys of MASK_BUILDERS.

    Extra keyword arguments are forwarded to the underlying builder (e.g.
    `seed=` for the variable-density mask).
    """
    if kind not in MASK_BUILDERS:
        raise ValueError(
            f"unknown mask type {kind!r}; expected one of {sorted(MASK_BUILDERS)}"
        )
    return MASK_BUILDERS[kind](shape, ratio, **kwargs)


def _validate_ratio(ratio: float) -> None:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"sampling ratio must be in (0, 1], got {ratio}")
