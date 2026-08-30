"""Task A: corrupting k-space with patient motion during a scan.

The whole module rests on the Fourier shift theorem:

    f(x - a)  <-->  F(k) * exp(-j*2*pi*k*a/N)

Translating the object leaves the k-space magnitude untouched and only stamps
a linear phase ramp. So a uniform shift is harmless -- it reconstructs to
exactly the shifted image (verify_shift_theorem checks this).

The artifact appears when the shift is not uniform across the scan. A
Cartesian scan fills one row per pulse repetition, so row index i is the
acquisition time for that row. If the patient moves between rows, each row
carries a different ramp and the result is no longer the transform of any one
consistent object -- which shows up as ghosting, blur, or streaking.

Only phase is changed here, never magnitude and never which points count as
measured, so motion composes with any mask in either order.

Every model returns (dy, 0.0): a y-only shift is a single per-row phase
scalar, giving the clean textbook ghosts along the phase-encode axis. A
time-varying x shift would smear each row sideways instead, reading as blur.
apply_motion itself accepts general (dy, dx).
"""

from __future__ import annotations

import numpy as np

from .kspace import from_kspace, to_kspace


def apply_motion(
    kspace_centered: np.ndarray,
    displacements: list[tuple[float, float]],
) -> np.ndarray:
    """Corrupt centered k-space with patient translation during the scan.

    displacements[i] is where the patient was, in image-space pixels, when
    row i was acquired; positive matches np.roll's sign. Needs one entry per
    row. Only phase changes, so abs(out) == abs(input) up to floating point.
    """
    ny, nx = kspace_centered.shape
    if len(displacements) != ny:
        raise ValueError(
            f"need one displacement per row: expected {ny}, got {len(displacements)}"
        )

    cy, cx = ny // 2, nx // 2
    kx = np.arange(nx) - cx  # same for every row
    out = kspace_centered.copy()

    for i, (dy, dx) in enumerate(displacements):
        ky = i - cy  # constant across the row
        ramp = np.exp(-2j * np.pi * (ky * dy / ny + kx * dx / nx))
        out[i, :] = kspace_centered[i, :] * ramp

    return out


def verify_shift_theorem(
    shape: tuple[int, int] = (64, 64), dy: int = 3, dx: int = -2
) -> float:
    """A uniform displacement must reconstruct to exactly np.roll(image, (dy, dx)).

    The load-bearing check for this module: if it does not hold to a couple of
    machine epsilons, the ramp formula has a sign or axis error and nothing
    built on apply_motion means anything. Returns the max absolute pixel
    error, expected around 1e-15.
    """
    ny = shape[0]
    rng = np.random.default_rng(0)
    image = rng.random(shape)

    k = to_kspace(image)
    uniform = [(dy, dx)] * ny
    shifted_via_kspace = from_kspace(apply_motion(k, uniform))
    shifted_via_roll = np.roll(image, (dy, dx), axis=(0, 1))

    return float(np.abs(shifted_via_kspace - shifted_via_roll).max())


# The three models below return one (dy, dx) per row, ready for apply_motion.


def sudden_jerk(ny: int, amp: float, at: int) -> list[tuple[float, float]]:
    """Patient sits still, jumps to a new position at row `at`, stays there.

    Rows before and after are each internally consistent, describing two sharp
    objects offset by amp -- so the result is a discrete ghost, not a smear.
    """
    return [(0.0, 0.0) if i < at else (amp, 0.0) for i in range(ny)]


def slow_drift(ny: int, amp: float) -> list[tuple[float, float]]:
    """Linear drift from 0 to amp over the scan, e.g. sinking into the table.

    Every row disagrees slightly with its neighbours rather than splitting into
    two consistent blocks, so the inconsistency spreads across k-space and
    reconstructs as blur rather than a second copy.
    """
    if ny == 1:
        return [(0.0, 0.0)]
    return [(amp * i / (ny - 1), 0.0) for i in range(ny)]


def periodic(ny: int, amp: float, cycles: float) -> list[tuple[float, float]]:
    """Sinusoidal motion, e.g. breathing: amp * sin(2*pi*cycles*i/ny).

    A sinusoid is a pair of complex exponentials, so to first order this
    convolves the image with two deltas offset along the phase-encode axis --
    a regular train of ghosts spaced by `cycles`.
    """
    i = np.arange(ny)
    return list(zip((amp * np.sin(2 * np.pi * cycles * i / ny)).tolist(),
                     [0.0] * ny))


# Radial scans acquire whole spokes, not rows, so timing radial data by row is
# physically wrong: the center should be an incoherent average over every
# spoke's position -- radial's whole robustness advantage -- but per-row timing
# stamps it with one row's phase error and the advantage vanishes.
# kspace._draw_spokes collapses spokes into a 0/1 mask and discards which point
# came from which spoke, so the indexed version below duplicates its geometry
# rather than editing it.


def n_spokes_for_ratio(shape: tuple[int, int], ratio: float) -> int:
    """Smallest spoke count covering `ratio` of k-space.

    Same search radial_mask runs internally, exposed because callers needing
    per-spoke timing must build the indexed map themselves.
    """
    lo, hi = 1, 4 * max(shape)
    best_n = hi
    while lo < hi:
        mid = (lo + hi) // 2
        coverage = np.count_nonzero(draw_spokes_indexed(shape, mid)) / (shape[0] * shape[1])
        if coverage >= ratio:
            hi, best_n = mid, mid
        else:
            lo = mid + 1
    return best_n


def draw_spokes_indexed(shape: tuple[int, int], n_spokes: int) -> np.ndarray:
    """Like kspace._draw_spokes, but stores spoke_index + 1 instead of 1.

    Keeps track of which spoke -- and so which acquisition time -- each point
    belongs to.

    Tie-break matters near the center: every spoke crosses it, and within about
    n_spokes/pi pixels adjacent spokes are closer than one pixel apart. A
    "first spoke wins" rule would hand that crowded, high-energy region to
    whichever spoke is processed first, re-concentrating the corruption onto a
    few acquisition times and destroying exactly the incoherence that makes
    radial motion-robust. So contested points go to whichever spoke's angle is
    closest to the point's own polar angle, which has no directional bias.
    """
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    index_map = np.zeros((ny, nx), dtype=np.int32)
    best_angle_gap = np.full((ny, nx), np.inf)

    max_radius = float(np.hypot(ny, nx)) / 2.0
    n_steps = int(np.ceil(2 * max_radius / 0.5)) + 1
    t = np.linspace(-max_radius, max_radius, n_steps)

    for s, angle in enumerate(np.linspace(0.0, np.pi, n_spokes, endpoint=False)):
        yy = np.rint(cy + t * np.sin(angle)).astype(int)
        xx = np.rint(cx + t * np.cos(angle)).astype(int)

        inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        yy, xx = yy[inside], xx[inside]

        # Mod pi to match the [0, pi) range spokes are drawn over. The center
        # pixel's angle is undefined (0/0) but it lies on every spoke anyway.
        true_angle = np.arctan2(yy - cy, xx - cx) % np.pi
        gap = np.abs(true_angle - angle)
        gap = np.minimum(gap, np.pi - gap)  # wraparound at the 0/pi seam

        closer = gap < best_angle_gap[yy, xx]
        index_map[yy[closer], xx[closer]] = s + 1
        best_angle_gap[yy[closer], xx[closer]] = gap[closer]

    return index_map


def apply_motion_radial(
    kspace_centered: np.ndarray,
    spoke_index: np.ndarray,
    spoke_displacements: list[tuple[float, float]],
) -> np.ndarray:
    """Motion timed per radial spoke rather than per row.

    spoke_index comes from draw_spokes_indexed; 0 means not on any spoke and is
    left alone (the mask zeroes it downstream anyway). One displacement per
    spoke.

    Unlike apply_motion, this uses each point's own (ky, kx): a spoke is not
    horizontal, so its points do not share a single ky. That is what makes the
    physics right for non-Cartesian trajectories.
    """
    ny, nx = kspace_centered.shape
    if spoke_index.shape != (ny, nx):
        raise ValueError(
            f"spoke_index shape {spoke_index.shape} does not match k-space "
            f"shape {(ny, nx)}"
        )
    n_spokes = int(spoke_index.max())
    if len(spoke_displacements) != n_spokes:
        raise ValueError(
            f"need one displacement per spoke: expected {n_spokes}, "
            f"got {len(spoke_displacements)}"
        )

    cy, cx = ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij")

    out = kspace_centered.copy()
    for s, (dy, dx) in enumerate(spoke_displacements):
        on_spoke = spoke_index == (s + 1)
        if not np.any(on_spoke):
            continue
        ky = yy[on_spoke]
        kx = xx[on_spoke]
        ramp = np.exp(-2j * np.pi * (ky * dy / ny + kx * dx / nx))
        out[on_spoke] = kspace_centered[on_spoke] * ramp

    return out


if __name__ == "__main__":
    # Cheap load-bearing check, run whenever this module is executed directly.
    err = verify_shift_theorem()
    print(f"uniform-shift check max error: {err:.3e}")
    assert err < 1e-9, "apply_motion does not match np.roll for a uniform shift"
    print("OK")
