"""All matplotlib output.

plot_reconstruction_panel draws the four-panel row for one (strategy, ratio)
pair; plot_metrics_summary draws PSNR/SSIM against ratio; plot_mask_gallery
shows the mask patterns side by side.

Style: grayscale for images and k-space, as radiology uses. Error maps get a
single-hue blue ramp (error is a magnitude, so it wants a sequential scale,
never a rainbow) with one shared colour scale per run. The summary chart gives
each strategy a fixed colour plus its own marker shape, so it survives
greyscale printing and colour-vision deficiency.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive: we only write files to disk
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap

from .kspace import MASK_LABELS, sampling_ratio

# Fixed per strategy so a strategy keeps its colour across every figure.
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

# Above this, PSNR is not a measurement -- just floating-point round-off.
PSNR_EXACT_THRESHOLD = 100.0

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_blue",
    ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def log_magnitude(kspace_centered: np.ndarray) -> np.ndarray:
    """Display transform for k-space: log1p compresses its huge dynamic range.

    DC can be thousands of times larger than the outer samples, so a linear
    scale shows one bright dot on black. The 1+ avoids log(0) at zero-filled
    points, which then render as pure black.
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
    """Four panels for one experiment: original, sampled k-space, recon, error.

    target_ratio is what was requested; the achieved ratio is read back off
    mask, since radial and random masks only approximate it. Pass the same
    error_vmax for every figure in a run to keep the error maps comparable;
    None auto-scales to this panel alone.
    """
    achieved = sampling_ratio(mask)
    error = np.abs(original - reconstruction)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)

    # Pinned to [0, 1] rather than autoscaled, so brightness matches the
    # reconstruction and any visible difference is a real one.
    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original image", color=INK_PRIMARY, fontsize=11)

    axes[1].imshow(log_magnitude(kspace_masked), cmap="gray")
    axes[1].set_title(
        f"Sampled k-space, log|K|\n{achieved * 100:.1f}% of points "
        f"({1 / achieved:.1f}x acceleration)",
        color=INK_PRIMARY,
        fontsize=11,
    )

    axes[2].imshow(reconstruction, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"Reconstruction\nPSNR {metrics['psnr']:.2f} dB   SSIM {metrics['ssim']:.4f}",
        color=INK_PRIMARY,
        fontsize=11,
    )

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
    """The sampling patterns side by side at one ratio, as {kind: mask}.

    Explanatory only -- it makes regular lines vs spokes vs random dots
    obvious at a glance.
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
    """Stage 2 demo: what the centre encodes vs what the edges encode.

    Two rows of mask | reconstruction | contrast-stretched reconstruction. The
    stretched column is needed because the edges-only image really is almost
    black (no DC means no mean brightness), so on a shared [0, 1] scale it
    looks like an empty frame and hides the point.
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
    """Stage 2 demo: zero-filling vs compressed sensing on identical measurements.

    Ground truth | mask | zero-fill | CS | PSNR per iteration. Both see exactly
    the same samples; they differ only in what they assume about the points
    that were never measured.
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

    convergence = axes[4]
    convergence.set_facecolor(SURFACE)
    if history and "psnr" in history[0]:
        iterations = [h["iteration"] for h in history]
        psnrs = [h["psnr"] for h in history]
        convergence.plot(iterations, psnrs, color=SERIES_COLORS["variable_density"],
                         linewidth=2.0, label="compressed sensing")
    # Zero-fill is a constant: it does not iterate.
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
    """PSNR and SSIM vs undersampling ratio, one line per strategy.

    results rows need "mask", "target_ratio", "psnr", "ssim".

    Two axes rather than one with twin scales: dB and a unitless 0-1 index on
    shared axes would invite false slope comparisons. The x axis is log-scaled
    because the ratios halve each step, and on a linear axis the interesting
    heavily-undersampled points would bunch up at the left edge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)

    # Stable order so colours and legend never shuffle.
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

            # Label at the leftmost point so the reader need not bounce between
            # legend and lines. Left, not right: at 100% all three strategies
            # land on the same value and the labels would collide.
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

        # A full reconstruction is exact, so its PSNR is ~300 dB of round-off
        # noise. Leaving it on the axis flattens every meaningful value into a
        # line at the bottom, so scale to the undersampled points and let it
        # run off the top with a note.
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

        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_facecolor(SURFACE)

        # Headroom so the direct labels are not clipped.
        ax.set_xmargin(0.18)

    # Left panel only -- both share the same series.
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_PRIMARY)

    fig.suptitle(
        "Reconstruction quality vs k-space undersampling (higher is better)",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, save_path)
    return save_path


def _all_values(results: list[dict], key: str) -> list[float]:
    """Every value of `key` across all result rows."""
    return [r[key] for r in results]


def _save(fig, save_path: str) -> None:
    """Make the parent directory, write the PNG, close the figure."""
    directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)  # we make dozens of figures per run
