"""
motion.py -- Task A: corrupting k-space with patient motion during a scan.

THE PHYSICS, IN ONE LINE: THE FOURIER SHIFT THEOREM
-----------------------------------------------------
    f(x - a)   <-->   F(k) * exp(-j*2*pi*k*a/N)

Translating the object in image space does not touch the *magnitude* of its
k-space at all -- it only stamps a linear **phase ramp** across k-space. This
is why a single, uniform patient shift is harmless: `apply_motion` with the
same `(dy, dx)` on every row reconstructs to *exactly* the shifted image
(see `verify_shift_theorem` below), because a uniform ramp is just what a
shifted object's k-space looks like.

The artifact appears only when the shift is **not** uniform across the scan.
`cartesian_mask` fills k-space one row at a time, and each row is one
repetition of the pulse sequence -- i.e. one moment in time. So row index `i`
IS the acquisition time for that row (row 0 earliest, row `ny - 1` latest). If
the patient is in a different place for different rows, each row carries a
*different* phase ramp. The reconstruction is then the inverse FFT of a
k-space that is not the transform of any single, consistent object -- and
that inconsistency between rows is exactly what shows up as ghosting, blur,
or streaking, depending on how the motion evolves over the scan.

This module only ever changes the *phase* of k-space samples, never their
magnitude and never which samples are considered "measured" -- motion and
undersampling are independent, orthogonal corruptions, and `apply_motion`
can be composed with any mask from `kspace.py` in either order relative to
`apply_mask` (motion does not care which points are subsequently zeroed).

WHY WE ONLY SHIFT ALONG y (THE PHASE-ENCODE AXIS)
---------------------------------------------------
Every model below returns displacements of the form `(dy, 0.0)`. This is a
deliberate simplification, not a limitation of `apply_motion` (which accepts
general `(dy, dx)`). A shift confined to y changes only a *per-row phase*
(because `ky` is constant across a row, `exp(-2j*pi*ky*dy/ny)` is a single
complex scalar multiplying the whole line) -- it does not redistribute a
row's energy across columns. That is what produces the clean, textbook
"discrete ghost copies displaced along the phase-encode axis" artifact this
module is built to demonstrate. A time-varying *x* shift would instead smear
each row's content sideways by a different amount, which reads as blur/shear
rather than discrete ghosts. Keeping motion on the y axis keeps the artifact
legible and matches the "ghosts along the vertical axis" check in
TASKS_MOTION_AND_ROI.md.
"""

from __future__ import annotations

import numpy as np

from .kspace import from_kspace, to_kspace

# ---------------------------------------------------------------------------
# 1. The core function: stamp a per-row phase ramp onto k-space
# ---------------------------------------------------------------------------


def apply_motion(
    kspace_centered: np.ndarray,
    displacements: list[tuple[float, float]],
) -> np.ndarray:
    """
    Corrupt centered k-space with patient translation during the scan.

    Parameters
    ----------
    kspace_centered : 2-D complex array
        Centered k-space (DC at [ny//2, nx//2]), same convention as the rest
        of the package -- see kspace.py's module docstring.
    displacements : sequence of (dy, dx), length ny
        displacements[i] = where the patient was, in pixels, when row i was
        acquired. `dy`/`dx` are in image-space pixels, positive meaning the
        same direction as `np.roll`'s shift argument (see verification
        below).

    Returns
    -------
    2-D complex array, same shape as the input. Only phase is changed --
    `np.abs(out) == np.abs(kspace_centered)` up to floating point.
    """
    ny, nx = kspace_centered.shape
    if len(displacements) != ny:
        raise ValueError(
            f"need one displacement per row: expected {ny}, got {len(displacements)}"
        )

    cy, cx = ny // 2, nx // 2
    kx = np.arange(nx) - cx  # frequency index along the row; same for every i
    out = kspace_centered.copy()

    for i, (dy, dx) in enumerate(displacements):
        ky = i - cy  # this row's frequency index -- constant across the row
        ramp = np.exp(-2j * np.pi * (ky * dy / ny + kx * dx / nx))
        out[i, :] = kspace_centered[i, :] * ramp

    return out


def verify_shift_theorem(
    shape: tuple[int, int] = (64, 64), dy: int = 3, dx: int = -2
) -> float:
    """
    Sanity check for `apply_motion`: a UNIFORM displacement (the same (dy,
    dx) on every row) must reconstruct to exactly `np.roll(image, (dy, dx))`.

    This is the load-bearing check for this whole module. If it does not
    hold to within a couple of machine epsilons, the ramp formula above has
    a sign or axis error and nothing built on top of `apply_motion` -- the
    three motion models, the per-spoke variant -- is meaningful. Run this
    before trusting anything else in this file.

    Returns
    -------
    float : max absolute pixel error between the two reconstructions
            (expected to be ~1e-15, machine-precision noise).
    """
    ny = shape[0]
    rng = np.random.default_rng(0)
    image = rng.random(shape)

    k = to_kspace(image)
    uniform = [(dy, dx)] * ny
    shifted_via_kspace = from_kspace(apply_motion(k, uniform))
    shifted_via_roll = np.roll(image, (dy, dx), axis=(0, 1))

    return float(np.abs(shifted_via_kspace - shifted_via_roll).max())


# ---------------------------------------------------------------------------
# 2. Motion models: displacement over time, for a Cartesian (row-per-time) scan
# ---------------------------------------------------------------------------
#
# Each of these returns a list of length `ny` -- one (dy, dx) per row, ready
# to hand straight to `apply_motion`. All three shift along y only; see the
# module docstring for why.


def sudden_jerk(ny: int, amp: float, at: int) -> list[tuple[float, float]]:
    """
    A single abrupt movement partway through the scan: the patient sits
    still, then jumps to a new position and stays there.

    displacement[i] = 0 for i < at, amp for i >= at.

    Rows before `at` and rows after `at` are each internally consistent --
    they describe two different, perfectly sharp objects offset by `amp`
    pixels. The reconstruction is therefore a superposition of two sharp
    copies of the object rather than a smear: a **discrete ghost**, the
    signature artifact of a step-function motion event.
    """
    return [(0.0, 0.0) if i < at else (amp, 0.0) for i in range(ny)]


def slow_drift(ny: int, amp: float) -> list[tuple[float, float]]:
    """
    A continuous, monotonic drift from 0 to `amp` pixels over the course of
    the scan (e.g. the patient gradually sinking into the table).

    displacement[i] = amp * i / (ny - 1) -- linear ramp, 0 at row 0 to
    `amp` at the last row.

    Because the position changes by a little bit between *every* pair of
    adjacent rows rather than at one instant, there is no pair of large,
    internally-consistent blocks the way there is for `sudden_jerk`. Instead
    every row disagrees slightly with its neighbours, which spreads the
    inconsistency continuously across k-space and reconstructs as **blur**
    rather than a distinct second copy.
    """
    if ny == 1:
        return [(0.0, 0.0)]
    return [(amp * i / (ny - 1), 0.0) for i in range(ny)]


def periodic(ny: int, amp: float, cycles: float) -> list[tuple[float, float]]:
    """
    Repetitive motion, e.g. breathing: displacement oscillates sinusoidally
    for the whole scan.

    displacement[i] = amp * sin(2*pi*cycles*i/ny).

    A sinusoid is itself just a pair of complex exponentials, so modulating
    k-space rows by a sinusoidal phase error is (to first order) equivalent
    to convolving the true image with a pair of delta functions offset in
    the phase-encode direction -- i.e. it produces a **regular train of
    ghosts**, spaced according to `cycles` (the number of breathing cycles
    that fit in the scan), rather than the single extra copy from
    `sudden_jerk` or the smear from `slow_drift`.
    """
    i = np.arange(ny)
    return list(zip((amp * np.sin(2 * np.pi * cycles * i / ny)).tolist(),
                     [0.0] * ny))


# ---------------------------------------------------------------------------
# 3. Per-spoke motion for radial sampling
# ---------------------------------------------------------------------------
#
# THE TRAP: applying `apply_motion` above (per-ROW timing) and then masking
# with `radial_mask` is physically wrong. `apply_motion` assumes acquisition
# order follows Cartesian rows; a radial scan instead acquires whole SPOKES,
# one at a time, and every spoke passes through the k-space center. If we
# time radial data by row instead of by spoke, the center -- which should be
# an incoherent average over every spoke's (different) position, and is
# radial's whole robustness advantage -- gets stamped with a single row's
# phase error like everything else, and the advantage disappears (see the
# "per-ROW timing (WRONG for radial)" numbers in TASKS_MOTION_AND_ROI.md).
#
# `kspace._draw_spokes` collapses all spokes into one 0/1 mask and throws
# away which point came from which spoke, so it cannot drive per-spoke
# timing. `draw_spokes_indexed` below is a separate function with the same
# geometry that keeps that information, without touching `_draw_spokes`
# itself (ground rule: existing mask builders are not to be edited).


def n_spokes_for_ratio(shape: tuple[int, int], ratio: float) -> int:
    """
    Smallest spoke count whose `draw_spokes_indexed` footprint covers at
    least `ratio` of k-space -- the same binary search `radial_mask` runs
    internally, exposed here because callers that need per-spoke timing have
    to build the indexed map themselves and so cannot go through
    `radial_mask` (which only returns the flattened 0/1 mask).
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
    """
    Same rasterized-spoke geometry as `kspace._draw_spokes`, but returns an
    int array where the value is `(spoke_index + 1)` and 0 means unsampled,
    instead of a plain 0/1 mask. This lets us later ask "which spoke, and
    therefore which acquisition time, does this k-space point belong to?"

    Deliberately duplicates `_draw_spokes`'s angle/step construction rather
    than reusing it, since that function returns only a flattened mask.

    TIE-BREAK NEAR THE CENTER: every spoke passes through the center, so
    spokes constantly land on grid points another spoke already claimed --
    within roughly `n_spokes / pi` pixels of the center, adjacent spokes are
    closer together than one pixel. A naive "first spoke to reach it wins"
    rule (looping spokes in angle order) is *not* a harmless arbitrary
    tie-break here: it systematically hands almost all of that crowded,
    high-energy central region to whichever spoke happens to be processed
    first, while later spokes get almost none of it. That defeats the whole
    point of per-spoke motion timing -- it re-concentrates the corruption
    onto a handful of spokes' acquisition times instead of spreading it
    incoherently across all of them, which is exactly the property that
    makes radial sampling motion-robust in the first place. So each
    contested point is instead given to whichever spoke's angle is
    geometrically closest to that point's own true polar angle -- a
    tie-break with no directional bias.
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

        # True polar angle of each candidate point (mod pi, matching the
        # [0, pi) range spokes are drawn over -- a spoke and its 180-degree
        # rotation are the same line). The center point itself has an
        # undefined angle (0/0); it is equally "on" every spoke, so leave it
        # at whatever the first spoke assigns -- one pixel, no real effect.
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
    """
    Corrupt centered k-space with motion timed per RADIAL SPOKE rather than
    per row.

    Parameters
    ----------
    kspace_centered : 2-D complex array, centered.
    spoke_index : int array from `draw_spokes_indexed(shape, n_spokes)`.
        `spoke_index == 0` marks points not on any spoke; left untouched
        (they will be zeroed by the mask downstream anyway).
    spoke_displacements : sequence of (dy, dx), length n_spokes.
        spoke_displacements[s] = where the patient was during spoke s.

    Unlike `apply_motion`, which multiplies a whole row by one phase ramp
    computed from that row's `ky` alone, this uses each point's own true
    `(ky, kx)` position -- a spoke is not horizontal in general, so its
    points do not share a single `ky`. Using the true coordinates is what
    makes the physics correct for non-Cartesian trajectories.
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
    # Same pattern as the round-trip assert in main.py: a cheap, load-bearing
    # correctness check that runs every time this module is executed
    # directly. See `verify_shift_theorem`'s docstring for why this matters.
    err = verify_shift_theorem()
    print(f"uniform-shift check max error: {err:.3e}")
    assert err < 1e-9, "apply_motion does not match np.roll for a uniform shift"
    print("OK")
