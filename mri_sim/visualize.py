"""
visualize.py -- all matplotlib output lives here.

Two kinds of figure:

  * `plot_reconstruction_panel`  -- one row of four panels for a single
    (sampling strategy, undersampling ratio) combination:
        original | sampled k-space (log-magnitude) | reconstruction | error
  * `plot_metrics_summary`       -- PSNR and SSIM versus undersampling ratio,
    one line per sampling strategy.

Plus `plot_mask_gallery`, a convenience figure showing the three mask patterns
side by side at one ratio, `plot_center_vs_edges` and `plot_cs_comparison` for
the Stage 2 demonstrations, and three reduced-FOV figures for `mri_sim/roi.py`:
`plot_kspace_nonlocality`, `plot_reduced_fov_panel` and
`plot_compact_reconstruction`.

Style notes (worth knowing for the report):
  * Images and k-space use a grayscale ramp -- achromatic, monotonic in
    lightness, and the conventional way radiological images are displayed.
  * The error maps use a single-hue blue ramp: error is a magnitude, and
    magnitude wants a sequential (one hue, light -> dark) colour scale, never
    a rainbow. All error maps in a run share one colour scale so panels from
    different figures are directly comparable.
  * The summary chart uses three fixed categorical colours, one per strategy,
    plus a distinct marker shape per strategy so the lines are still
    distinguishable in greyscale print or with colour-vision deficiency.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend: we only write files to disk
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from .kspace import MASK_LABELS, sampling_ratio

# --- Colour and chrome constants -------------------------------------------
# One fixed colour per sampling strategy, assigned by identity and never
# re-shuffled, so a strategy keeps the same colour in every figure.
SERIES_COLORS = {
    "cartesian": "#2a78d6",         # blue
    "radial": "#eb6834",            # orange
    "variable_density": "#1baf7a",  # aqua
}
SERIES_MARKERS = {
    "cartesian": "o",
    "radial": "s",
    "variable_density": "^",
}

SURFACE = "#fcfcfb"      # figure background
INK_PRIMARY = "#0b0b0b"  # titles
INK_MUTED = "#898781"    # tick labels, axis text
GRIDLINE = "#e1e0d9"     # hairline grid
BASELINE = "#c3c2b7"     # axis spines

# Any PSNR above this is not a real measurement -- it is a perfect
# reconstruction whose only error is floating-point round-off.
PSNR_EXACT_THRESHOLD = 100.0

# Single-hue sequential ramp for error magnitude: white (no error) -> dark blue.
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_blue",
    ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def log_magnitude(kspace_centered: np.ndarray) -> np.ndarray:
    """
    Display transform for k-space.

    k-space magnitude has an enormous dynamic range -- the DC point can be
    many thousands of times larger than the outer high-frequency samples. On
    a linear scale you would see a single bright dot on a black field and
    nothing else. log(1 + |K|) compresses that range so the structure of the
    sampling pattern and of the data itself both become visible.

    The `1 +` avoids log(0) at the zero-filled (unsampled) points, which map
    to exactly 0 and therefore render as pure black.
    """
    return np.log1p(np.abs(kspace_centered))


def plot_reconstruction_panel(
    original: np.ndarray,
    kspace_masked: np.ndarray,
    reconstruction: np.ndarray,
    mask_kind: str,
    target_ratio: float,
    mask: np.ndarray,
    metrics: dict,
    save_path: str,
    error_vmax: float | None = None,
) -> str:
    """
    Draw the four-panel comparison for one experiment and save it to disk.

    Parameters
    ----------
    original : ground-truth image
    kspace_masked : centered k-space after the mask has been applied
    reconstruction : magnitude image from the inverse FFT
    mask_kind : key into MASK_LABELS ("cartesian" / "radial" / ...)
    target_ratio : the ratio that was requested (the achieved one is read
        back off `mask`, since radial and random masks only approximate it)
    mask : the sampling mask itself, used to report the achieved ratio
    metrics : dict with "psnr" and "ssim"
    save_path : where to write the PNG
    error_vmax : upper limit of the error colour scale. Pass the same value
        for every figure in a run so the error maps are comparable; None
        auto-scales to this panel alone.

    Returns
    -------
    The path written.
    """
    achieved = sampling_ratio(mask)
    error = np.abs(original - reconstruction)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)

    # --- Panel 1: ground truth ---------------------------------------------
    # vmin/vmax pinned to [0, 1] rather than autoscaled, so brightness is
    # identical between the original and the reconstruction and any visible
    # difference is a real difference.
    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original image", color=INK_PRIMARY, fontsize=11)

    # --- Panel 2: the k-space that was actually acquired --------------------
    axes[1].imshow(log_magnitude(kspace_masked), cmap="gray")
    axes[1].set_title(
        f"Sampled k-space, log|K|\n{achieved * 100:.1f}% of points "
        f"({1 / achieved:.1f}x acceleration)",
        color=INK_PRIMARY,
        fontsize=11,
    )

    # --- Panel 3: the zero-filled reconstruction ----------------------------
    axes[2].imshow(reconstruction, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"Reconstruction\nPSNR {metrics['psnr']:.2f} dB   SSIM {metrics['ssim']:.4f}",
        color=INK_PRIMARY,
        fontsize=11,
    )

    # --- Panel 4: where the reconstruction went wrong -----------------------
    im = axes[3].imshow(error, cmap=ERROR_CMAP, vmin=0.0, vmax=error_vmax)
    axes[3].set_title(
        f"|difference|   (max {error.max():.3f})", color=INK_PRIMARY, fontsize=11
    )
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BASELINE)

    fig.suptitle(
        f"{MASK_LABELS[mask_kind]}  --  target sampling {target_ratio * 100:g}%",
        color=INK_PRIMARY,
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return save_path


def plot_mask_gallery(masks: dict, ratio: float, save_path: str) -> str:
    """
    Show the three sampling patterns side by side at one undersampling ratio.

    Purely explanatory -- it makes the difference between "regular lines",
    "spokes" and "random dots, dense in the middle" obvious at a glance.

    Parameters
    ----------
    masks : {mask_kind: mask array}
    ratio : the target ratio these masks were built for (for the title)
    """
    fig, axes = plt.subplots(1, len(masks), figsize=(4.5 * len(masks), 5.0),
                             facecolor=SURFACE)
    if len(masks) == 1:
        axes = [axes]

    for ax, (kind, mask) in zip(axes, masks.items()):
        # White = sampled, black = skipped.
        ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        achieved = sampling_ratio(mask)
        ax.set_title(
            f"{MASK_LABELS[kind]}\n{achieved * 100:.1f}% sampled",
            color=INK_PRIMARY,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BASELINE)

    fig.suptitle(
        f"k-space sampling patterns at a target of {ratio * 100:g}% "
        f"(white = acquired)",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, save_path)
    return save_path


def plot_center_vs_edges(
    original: np.ndarray,
    center_mask: np.ndarray,
    center_recon: np.ndarray,
    edges_mask: np.ndarray,
    edges_recon: np.ndarray,
    save_path: str,
) -> str:
    """
    Stage 2 demo: what the centre of k-space encodes vs what the edges encode.

    Two rows, three columns: mask | reconstruction | the same reconstruction
    with its own display range stretched.

    The stretched column exists because the edges-only reconstruction is
    genuinely almost black -- discarding the centre removes the DC term, i.e.
    the mean brightness of the image. On a shared [0, 1] scale it looks like
    an empty frame, which hides the point; auto-scaled, the edge map it really
    is becomes obvious.
    """
    rows = [
        ("Center only  (low-pass)", center_mask, center_recon,
         "contrast and shape survive; edges are blurred"),
        ("Edges only  (high-pass)", edges_mask, edges_recon,
         "anatomy is gone; only boundaries remain"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0), facecolor=SURFACE)

    for row_index, (label, mask, recon, caption) in enumerate(rows):
        achieved = sampling_ratio(mask)

        axes[row_index, 0].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_title(
            f"{label}\n{achieved * 100:.1f}% of k-space acquired",
            color=INK_PRIMARY, fontsize=11,
        )

        # Same [0, 1] scale as every other reconstruction in the project, so
        # brightness is directly comparable with the original image.
        axes[row_index, 1].imshow(recon, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].set_title(
            f"Reconstruction, true brightness\nmean = {recon.mean():.3f}",
            color=INK_PRIMARY, fontsize=11,
        )

        axes[row_index, 2].imshow(recon, cmap="gray")
        axes[row_index, 2].set_title(
            f"Same image, contrast stretched\n{caption}",
            color=INK_PRIMARY, fontsize=11,
        )

        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(BASELINE)

    fig.suptitle(
        "Where the information lives: k-space centre vs k-space edges",
        color=INK_PRIMARY, fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return save_path


def plot_cs_comparison(
    original: np.ndarray,
    comparison: dict,
    mask: np.ndarray,
    save_path: str,
) -> str:
    """
    Stage 2 demo: zero-filling vs compressed sensing on identical measurements.

    Five panels: ground truth | the mask | zero-fill | CS | PSNR per iteration.

    Both reconstructions see exactly the same samples. The only difference is
    the assumption each makes about the points that were never measured --
    "they were zero" versus "they were whatever makes the image sparsest while
    still matching what we did measure".
    """
    zero_fill = comparison["zero_fill_image"]
    cs_image = comparison["cs_image"]
    zf_metrics = comparison["zero_fill_metrics"]
    cs_metrics = comparison["cs_metrics"]
    history = comparison["history"]

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6), facecolor=SURFACE)

    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground truth", color=INK_PRIMARY, fontsize=11)

    axes[1].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"Random variable-density mask\n{sampling_ratio(mask) * 100:.1f}% sampled "
        f"({1 / sampling_ratio(mask):.1f}x)",
        color=INK_PRIMARY, fontsize=11,
    )

    axes[2].imshow(zero_fill, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"Zero-filled (linear)\nPSNR {zf_metrics['psnr']:.2f} dB   "
        f"SSIM {zf_metrics['ssim']:.4f}",
        color=INK_PRIMARY, fontsize=11,
    )

    axes[3].imshow(cs_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"Compressed sensing (FISTA)\nPSNR {cs_metrics['psnr']:.2f} dB   "
        f"SSIM {cs_metrics['ssim']:.4f}",
        color=INK_PRIMARY, fontsize=11,
    )

    for ax in axes[:4]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BASELINE)

    # --- convergence curve --------------------------------------------------
    convergence = axes[4]
    convergence.set_facecolor(SURFACE)
    if history and "psnr" in history[0]:
        iterations = [h["iteration"] for h in history]
        psnrs = [h["psnr"] for h in history]
        convergence.plot(iterations, psnrs, color=SERIES_COLORS["variable_density"],
                         linewidth=2.0, label="compressed sensing")
    # The zero-fill baseline is a constant: it does not iterate.
    convergence.axhline(zf_metrics["psnr"], color=INK_MUTED, linestyle="--",
                        linewidth=1.4, label="zero-fill baseline")
    convergence.set_xlabel("iteration", color=INK_MUTED, fontsize=10)
    convergence.set_ylabel("PSNR (dB)", color=INK_MUTED, fontsize=10)
    convergence.set_title("Convergence", color=INK_PRIMARY, fontsize=11)
    convergence.grid(color=GRIDLINE, linewidth=0.8)
    convergence.legend(frameon=False, fontsize=9)
    convergence.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in convergence.spines.values():
        spine.set_edgecolor(BASELINE)

    fig.suptitle(
        "Same measurements, two different assumptions about the missing data",
        color=INK_PRIMARY, fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, save_path)
    return save_path


def plot_metrics_summary(results: list[dict], save_path: str) -> str:
    """
    PSNR and SSIM versus undersampling ratio, one line per sampling strategy.

    Two separate axes rather than one axis with two y-scales: PSNR is in dB
    and SSIM is a unitless 0-1 index, and overlaying two incompatible scales
    on one plot invites false comparisons of the slopes.

    The x axis is log-scaled because the ratios halve each step
    (100 -> 50 -> 25 -> 12.5%); on a linear axis the three interesting,
    heavily-undersampled points would be crushed together at the left edge.

    The PSNR axis is deliberately clipped -- see PSNR_EXACT_THRESHOLD below.

    Parameters
    ----------
    results : list of dicts, each with keys
        "mask" (str), "target_ratio" (float), "psnr" (float), "ssim" (float)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)

    # Keep a stable strategy order so colours and legend order never shuffle.
    kinds = [k for k in SERIES_COLORS if any(r["mask"] == k for r in results)]

    for metric_name, unit, ax in [
        ("psnr", "PSNR (dB)", axes[0]),
        ("ssim", "SSIM", axes[1]),
    ]:
        for kind in kinds:
            rows = sorted(
                (r for r in results if r["mask"] == kind),
                key=lambda r: r["target_ratio"],
            )
            xs = [r["target_ratio"] * 100 for r in rows]
            ys = [r[metric_name] for r in rows]

            ax.plot(
                xs,
                ys,
                color=SERIES_COLORS[kind],
                marker=SERIES_MARKERS[kind],   # shape encodes identity too, so
                markersize=8,                  # the chart survives greyscale
                linewidth=2,                   # printing and colour blindness
                label=MASK_LABELS[kind].split(" (")[0],
                zorder=3,
            )

            # Direct label at the leftmost (most heavily undersampled) point,
            # so the reader does not have to bounce between legend and lines.
            # Left rather than right because at 100% sampling all three
            # strategies land on the same value and the labels would overlap.
            ax.annotate(
                MASK_LABELS[kind].split(" (")[0],
                xy=(xs[0], ys[0]),
                xytext=(-8, 0),
                textcoords="offset points",
                color=SERIES_COLORS[kind],
                fontsize=9,
                va="center",
                ha="right",
                zorder=4,
            )

        # A fully-sampled reconstruction is exact, so its "error" is nothing
        # but float64 round-off and its PSNR comes out around 300 dB. That is
        # not a quality measurement -- it is the numerical noise floor -- and
        # leaving it on the axis compresses every meaningful value into a flat
        # line at the bottom. So we scale the axis to the undersampled points
        # and let the 100% point run off the top, with a note saying so.
        if metric_name == "psnr":
            real_values = [y for y in _all_values(results, "psnr")
                           if y < PSNR_EXACT_THRESHOLD]
            if real_values and len(real_values) < len(_all_values(results, "psnr")):
                ax.set_ylim(top=max(real_values) * 1.18)
                ax.text(
                    0.98,
                    0.03,
                    "100% sampling reconstructs exactly\n"
                    "(PSNR limited only by float64 round-off, off scale)",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=INK_MUTED,
                )

        ax.set_xscale("log")
        ax.set_xticks(sorted({r["target_ratio"] * 100 for r in results}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("k-space sampled (%)  --  log scale", color=INK_MUTED, fontsize=10)
        ax.set_ylabel(unit, color=INK_MUTED, fontsize=10)
        ax.set_title(f"{unit} vs undersampling", color=INK_PRIMARY, fontsize=12)

        # Recessive chrome: a hairline horizontal grid, no top/right spines.
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_facecolor(SURFACE)

        # Headroom on the right so the direct labels are not clipped.
        ax.set_xmargin(0.18)

    # Legend on the left-hand panel only -- both panels share the same three
    # series, so repeating it would be noise.
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_PRIMARY)

    fig.suptitle(
        "Reconstruction quality vs k-space undersampling (higher is better)",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# Reduced-FOV (ROI) figures -- see mri_sim/roi.py
# ---------------------------------------------------------------------------


def plot_kspace_nonlocality(image, demo: dict, save_path: str) -> str:
    """
    Figure for the misconception: "the lesion is over there, so keep that
    corner of k-space".

    Four panels: the object | k-space with one quadrant deleted | the
    reconstruction | the error map, annotated with the mean absolute error in
    each of the four **image** quadrants.

    The annotation is the whole figure. If k-space were spatially local, one
    quadrant of the error map would be lit up and the other three would be
    black. Instead all four numbers are within a factor of two of each other:
    deleting a quarter of k-space damages the entire image, roughly evenly,
    because every k-space sample is a measurement of every pixel.

    Parameters
    ----------
    image : ground-truth object
    demo : the dict returned by `roi.kspace_locality_demo`
    """
    image = np.asarray(image, dtype=np.float64)
    error = demo["error"]
    quadrant_errors = demo["quadrant_errors"]
    ny, nx = image.shape

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), facecolor=SURFACE)

    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Object (ground truth)", color=INK_PRIMARY, fontsize=11)

    axes[1].imshow(log_magnitude(demo["kspace_damaged"]), cmap="gray")
    axes[1].set_title(
        f"k-space with the {demo['quadrant']} quadrant deleted\n"
        f"{demo['kept_fraction'] * 100:.0f}% of samples kept",
        color=INK_PRIMARY, fontsize=11,
    )

    axes[2].imshow(demo["reconstruction"], cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"Reconstruction\nPSNR {demo['metrics']['psnr']:.2f} dB   "
        f"SSIM {demo['metrics']['ssim']:.4f}",
        color=INK_PRIMARY, fontsize=11,
    )

    im = axes[3].imshow(error, cmap=ERROR_CMAP, vmin=0.0)
    axes[3].set_title(
        "|error|, mean per image quadrant\nthe damage is spread everywhere",
        color=INK_PRIMARY, fontsize=11,
    )
    # Quarter lines plus the four means, printed where they were measured.
    axes[3].axhline(ny / 2, color=INK_PRIMARY, linewidth=0.8, alpha=0.5)
    axes[3].axvline(nx / 2, color=INK_PRIMARY, linewidth=0.8, alpha=0.5)
    for row in range(2):
        for col in range(2):
            axes[3].text(
                nx * (0.25 + 0.5 * col), ny * (0.25 + 0.5 * row),
                f"{quadrant_errors[row, col]:.4f}",
                color=INK_PRIMARY, fontsize=13, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor=SURFACE, edgecolor=BASELINE,
                          boxstyle="round,pad=0.3", alpha=0.85),
            )
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BASELINE)

    spread = quadrant_errors.max() / max(quadrant_errors.min(), 1e-12)
    fig.suptitle(
        "k-space is not spatially local: every sample carries every pixel "
        f"(worst/best quadrant error = {spread:.2f}x, not infinity)",
        color=INK_PRIMARY, fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, save_path)
    return save_path


def plot_reduced_fov_panel(image, comparison: dict, save_path: str) -> str:
    """
    Figure for the four ways of spending the same 1/R^2 of k-space.

    One row per variant from `roi.compare_roi_strategies`, five columns:

        what was excited | k-space acquired | reconstruction |
        the ROI, reconstructed | the ROI, ground truth

    Every row uses the same number of samples and therefore the same scan
    time. The last two columns are the ones to look at: only the first row
    reproduces the ground-truth crop, and only the first row excited the box.
    The bottom row -- identical samples to the top row, RF excitation removed
    -- is a fold-over catastrophe with a negative PSNR.

    Parameters
    ----------
    image : ground-truth object
    comparison : the dict returned by `roi.compare_roi_strategies`
    """
    image = np.asarray(image, dtype=np.float64)
    box = comparison["box"]
    variants = comparison["variants"]
    truth_crop = image[box.slices]

    rows = len(variants)
    fig, axes = plt.subplots(rows, 5, figsize=(19.5, 4.0 * rows), facecolor=SURFACE)

    for index, variant in enumerate(variants):
        row = axes[index]

        row[0].imshow(variant["object"], cmap="gray", vmin=0.0, vmax=1.0)
        row[0].set_title(
            "Excited object (what makes signal)" if index == 0 else "",
            color=INK_MUTED, fontsize=10,
        )
        # The row label goes on the y-axis of the first panel, so the reader
        # can scan down the left edge and see the four strategies.
        row[0].set_ylabel(
            f"{variant['label']}\n{variant['note']}",
            color=INK_PRIMARY, fontsize=9.5, rotation=0,
            ha="right", va="center", labelpad=14,
        )

        row[1].imshow(log_magnitude(variant["kspace"]), cmap="gray")
        row[1].set_title(
            f"k-space acquired: {variant['ratio'] * 100:.2f}% "
            f"({variant['acceleration']:.1f}x faster)",
            color=INK_MUTED, fontsize=10,
        )

        # Every panel in this project uses a fixed [0, 1] display range so that
        # brightness is comparable. The no-suppression row is the exception: it
        # sums R^2 aliased copies of the head, so its values run several times
        # past 1.0 and a [0, 1] range would render it as a blank white square,
        # hiding the very fold-over that is the point. Those rows get their own
        # scale, and the title says so.
        recon = variant["reconstruction"]
        peak = float(recon.max())
        overflows = peak > 1.5
        display_max = peak if overflows else 1.0
        scale_note = f"\ndisplay range 0-{peak:.1f} (overflows)" if overflows else ""

        row[2].imshow(recon, cmap="gray", vmin=0.0, vmax=display_max)
        row[2].set_title(
            f"Reconstruction (full grid){scale_note}",
            color=INK_MUTED, fontsize=10,
        )

        crop = recon[box.slices]
        row[3].imshow(crop, cmap="gray", vmin=0.0, vmax=display_max)
        row[3].set_title(
            f"Inside the ROI:  PSNR {variant['psnr']:.2f} dB   "
            f"SSIM {variant['ssim']:.4f}",
            color=INK_PRIMARY, fontsize=10.5,
        )

        row[4].imshow(truth_crop, cmap="gray", vmin=0.0, vmax=1.0)
        row[4].set_title("The ROI, ground truth", color=INK_MUTED, fontsize=10)

        # Outline the excited box on both full-size panels, so it is obvious
        # which part of the picture the numbers refer to.
        for ax in (row[0], row[2]):
            _draw_box(ax, box)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(BASELINE)

    fig.suptitle(
        f"Reduced-FOV imaging at R = {comparison['R']}: same scan time, "
        f"four ways to spend it (scored inside the ROI only)",
        color=INK_PRIMARY, fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, save_path)
    return save_path


def plot_compact_reconstruction(
    image, comparison: dict, compact_wrapped, compact_centered, save_path: str
) -> str:
    """
    Figure for what a scanner would actually hand back from an inner-volume scan.

    Four panels: the full-grid reconstruction (with the box outlined) | the
    ground-truth ROI | the compact reconstruction as it comes out of the small
    inverse FFT | the same after undoing the wrap.

    The point of panels 3 and 4: only `(N/R)^2` samples were measured, so the
    natural output is an `(N/R, N/R)` image -- same resolution, smaller field
    of view, 1/R^2 of the data. It comes out circularly shifted, because
    decimating k-space keeps the *full* FOV's origin as the small FOV's
    origin; rolling by `(y0 mod N/R, x0 mod N/R)` puts the anatomy back.
    """
    image = np.asarray(image, dtype=np.float64)
    box = comparison["box"]
    reduced = comparison["variants"][0]      # the reduced-FOV row
    R = comparison["R"]
    full_recon = reduced["reconstruction"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), facecolor=SURFACE)

    axes[0].imshow(full_recon, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(
        f"Full-grid reconstruction\n{full_recon.shape[0]}x{full_recon.shape[1]}, "
        f"zero-filled + x{R * R} density compensation",
        color=INK_PRIMARY, fontsize=11,
    )
    _draw_box(axes[0], box)

    axes[1].imshow(image[box.slices], cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"Ground truth in the ROI\n{box.size}x{box.size}",
        color=INK_PRIMARY, fontsize=11,
    )

    axes[2].imshow(compact_wrapped, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"Compact reconstruction, raw\n{compact_wrapped.shape[0]}x"
        f"{compact_wrapped.shape[1]} -- ROI appears wrapped",
        color=INK_PRIMARY, fontsize=11,
    )

    axes[3].imshow(compact_centered, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"Same data, wrap undone\nrolled by ({box.y0 % box.size}, "
        f"{box.x0 % box.size}) pixels",
        color=INK_PRIMARY, fontsize=11,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BASELINE)

    fig.suptitle(
        f"What the scanner really returns: {box.size}x{box.size} pixels from "
        f"{100.0 / (R * R):.2f}% of k-space, at full resolution",
        color=INK_PRIMARY, fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, save_path)
    return save_path


def _draw_box(ax, box) -> None:
    """Outline an `roi.ROIBox` on an image axis (the excited region)."""
    ax.add_patch(
        Rectangle(
            (box.x0 - 0.5, box.y0 - 0.5), box.size, box.size,
            fill=False, edgecolor="#eb6834", linewidth=1.6, linestyle="--",
        )
    )


def _all_values(results: list[dict], key: str) -> list[float]:
    """Every value of `key` across all result rows."""
    return [r[key] for r in results]


def _save(fig, save_path: str) -> None:
    """Create the parent directory if needed, write the PNG, close the figure."""
    directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)  # closing matters: we make dozens of figures in one run
