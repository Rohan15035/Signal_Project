"""Forward/inverse FFT, sampling masks, and zero-filled reconstruction.

Every k-space array in this package is fftshift-centered: DC sits at
[ny//2, nx//2], so masks can just reason about distance from the middle.
from_kspace undoes that before calling ifft2.
"""

from __future__ import annotations

import numpy as np


def to_kspace(image: np.ndarray) -> np.ndarray:
    """Image -> centered k-space. (A real scanner measures this directly.)"""
    return np.fft.fftshift(np.fft.fft2(image))


def from_kspace(kspace_centered: np.ndarray) -> np.ndarray:
    """Centered k-space -> magnitude image."""
    # ifftshift, not fftshift: they differ for odd-sized arrays.
    # Magnitude because masked k-space is no longer the transform of a real
    # image, so the result has a genuine imaginary part. Scanners do the same.
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_centered)))


def apply_mask(kspace_centered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-fill every k-space point the mask does not select."""
    if kspace_centered.shape != mask.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match k-space shape "
            f"{kspace_centered.shape}"
        )
    return kspace_centered * mask


def reconstruct(kspace_centered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-filled reconstruction: mask, then inverse FFT."""
    return from_kspace(apply_mask(kspace_centered, mask))


def sampling_ratio(mask: np.ndarray) -> float:
    """Fraction of k-space actually sampled (1.0 = fully sampled)."""
    return float(np.count_nonzero(mask)) / mask.size


def acceleration_factor(mask: np.ndarray) -> float:
    """Scan-time speed-up, 1 / sampling_ratio. Ratio 0.25 means 4x faster."""
    ratio = sampling_ratio(mask)
    return float("inf") if ratio == 0 else 1.0 / ratio


def _center_index(n: int) -> int:
    """DC index along an axis of length n, matching fftshift."""
    return n // 2


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    """Distance from center, normalized so 1.0 is the edge (corners exceed it)."""
    ny, nx = shape
    cy, cx = _center_index(ny), _center_index(nx)

    y = np.arange(ny) - cy
    x = np.arange(nx) - cx

    # Normalize per axis so non-square images still get circular contours.
    y = y / max(cy, 1)
    x = x / max(cx, 1)

    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.sqrt(yy**2 + xx**2)


def _full_mask(shape: tuple[int, int]) -> np.ndarray:
    """Fully-sampled mask -- the 100% reference case."""
    return np.ones(shape, dtype=np.float64)


def cartesian_mask(
    shape: tuple[int, int],
    ratio: float,
    center_fraction: float | None = None,
) -> np.ndarray:
    """Keep every Nth horizontal line, plus a fully-sampled center block.

    One line = one phase-encoding step = one unit of scan time, so skipping
    lines gives a real speed-up.

    The center block is not optional: the center of k-space holds the low
    frequencies (contrast, overall shape) and nearly all the image energy,
    while the edges hold fine detail. Undersample the center and the image is
    unusable; undersample the edges and it just gets blurrier.

    center_fraction defaults to 0.32 * ratio, the fastMRI convention.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    ny, nx = shape
    if center_fraction is None:
        center_fraction = 0.32 * ratio

    n_target = max(1, int(round(ratio * ny)))

    n_center = max(1, int(round(center_fraction * ny)))
    n_center = min(n_center, n_target)  # never overspend the budget

    cy = _center_index(ny)
    start = cy - n_center // 2
    center_lines = np.arange(start, start + n_center)
    center_lines = center_lines[(center_lines >= 0) & (center_lines < ny)]

    mask = np.zeros((ny, nx), dtype=np.float64)
    mask[center_lines, :] = 1.0

    # Spend whatever budget is left on evenly-spaced outer lines.
    n_outer = n_target - len(center_lines)
    if n_outer > 0:
        outer_lines = np.setdiff1d(np.arange(ny), center_lines)
        stride = len(outer_lines) / n_outer
        picks = np.unique(np.floor(np.arange(n_outer) * stride).astype(int))
        mask[outer_lines[picks], :] = 1.0

    return mask


def radial_mask(
    shape: tuple[int, int],
    ratio: float | None = None,
    n_spokes: int | None = None,
) -> np.ndarray:
    """Sample along spokes through the k-space center.

    Every spoke crosses the center, so the low frequencies are densely covered
    for free. Its artifacts are incoherent streaks rather than the crisp ghosts
    regular Cartesian skipping produces.

    Rasterized approximation: real radial trajectories land off-grid and need a
    NUFFT. Snapping to the nearest grid point lets us keep using ifft2 and still
    reproduces the characteristic artifacts.

    Give exactly one of ratio (spoke count found by bisection) or n_spokes.
    Spokes never reach the far corners, so ratio 1.0 returns a full mask.
    """
    if (ratio is None) == (n_spokes is None):
        raise ValueError("give exactly one of `ratio` or `n_spokes`")

    if n_spokes is not None:
        return _draw_spokes(shape, n_spokes)

    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    # Coverage grows monotonically with spoke count, so bisection is safe.
    lo, hi = 1, 4 * max(shape)
    best = _draw_spokes(shape, hi)
    if sampling_ratio(best) < ratio:
        return best  # unreachable with spokes; return the densest one

    while lo < hi:
        mid = (lo + hi) // 2
        candidate = _draw_spokes(shape, mid)
        if sampling_ratio(candidate) >= ratio:
            hi, best = mid, candidate
        else:
            lo = mid + 1

    return best


def _draw_spokes(shape: tuple[int, int], n_spokes: int) -> np.ndarray:
    """Rasterize n_spokes lines through the center onto the grid."""
    ny, nx = shape
    cy, cx = _center_index(ny), _center_index(nx)
    mask = np.zeros((ny, nx), dtype=np.float64)

    max_radius = float(np.hypot(ny, nx)) / 2.0

    # 0.5 px steps so rounding to the grid never leaves gaps in a line.
    n_steps = int(np.ceil(2 * max_radius / 0.5)) + 1
    t = np.linspace(-max_radius, max_radius, n_steps)

    # [0, pi) not [0, 2pi): a spoke and its 180-degree rotation are identical.
    for angle in np.linspace(0.0, np.pi, n_spokes, endpoint=False):
        yy = np.rint(cy + t * np.sin(angle)).astype(int)
        xx = np.rint(cx + t * np.cos(angle)).astype(int)

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
    """Random sampling biased toward the center, p(r) = alpha * (1 - r)^poly_order.

    alpha is solved numerically so the mean probability equals ratio; a small
    central disk is forced to p = 1.

    Randomness is what compressed sensing needs. Regular skipping produces
    coherent ghosts that look like real structure; random sampling spreads the
    error into incoherent noise, which a sparsity-promoting denoiser can undo.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    radius = _radius_grid(shape)

    # 1.0 at the center falling to 0 at r = 1; clip keeps corners non-negative.
    profile = np.clip(1.0 - radius, 0.0, 1.0) ** poly_order

    center_disk = radius <= center_radius

    # Clipping at 1.0 rules out a closed form, but mean_prob is monotonic in
    # alpha, so bisect.
    def mean_prob(alpha: float) -> float:
        p = np.clip(alpha * profile, 0.0, 1.0)
        p[center_disk] = 1.0
        return float(p.mean())

    lo, hi = 0.0, 1.0
    while mean_prob(hi) < ratio:
        hi *= 2.0
        if hi > 1e9:  # ratio unreachable; center disk alone exceeds it
            break

    for _ in range(100):
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


# The next two are not realistic acquisitions. They exist to show, in one pair
# of pictures, that the center carries contrast and shape while the edges carry
# detail. Radii are found numerically because the corners of a square k-space
# fall outside the inscribed circle.


def _radius_for_ratio(radius_map: np.ndarray, ratio: float) -> float:
    """Smallest radius enclosing `ratio` of all points, read off as a quantile."""
    flat = np.sort(radius_map, axis=None)
    index = int(np.clip(round(ratio * (flat.size - 1)), 0, flat.size - 1))
    return float(flat[index])


def center_only_mask(shape: tuple[int, int], ratio: float) -> np.ndarray:
    """Central disc only: an ideal low-pass filter.

    Right contrast and shape, blurred edges. The hard rim causes Gibbs ringing
    around sharp boundaries.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    radius = _radius_grid(shape)
    return (radius <= _radius_for_ratio(radius, ratio)).astype(np.float64)


def edges_only_mask(shape: tuple[int, int], ratio: float) -> np.ndarray:
    """Outer annulus only: an ideal high-pass filter.

    Mostly black with bright edges -- dropping the center also drops DC, the
    mean brightness. Put this next to center_only_mask.
    """
    _validate_ratio(ratio)
    if ratio >= 1.0:
        return _full_mask(shape)

    radius = _radius_grid(shape)
    return (radius > _radius_for_ratio(radius, 1.0 - ratio)).astype(np.float64)


MASK_BUILDERS = {
    "cartesian": cartesian_mask,
    "radial": radial_mask,
    "variable_density": variable_density_mask,
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

# The three that model a real accelerated scan; main.py sweeps over these.
ACQUISITION_MASKS = ["cartesian", "radial", "variable_density"]


def build_mask(kind: str, shape: tuple[int, int], ratio: float, **kwargs) -> np.ndarray:
    """Build a mask by name; extra kwargs go to the underlying builder."""
    if kind not in MASK_BUILDERS:
        raise ValueError(
            f"unknown mask type {kind!r}; expected one of {sorted(MASK_BUILDERS)}"
        )
    return MASK_BUILDERS[kind](shape, ratio, **kwargs)


def _validate_ratio(ratio: float) -> None:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"sampling ratio must be in (0, 1], got {ratio}")
