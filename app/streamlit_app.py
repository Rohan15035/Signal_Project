"""
streamlit_app.py -- the interactive MRI k-space simulator.

Run it from the project root:

    streamlit run app/streamlit_app.py

then open http://localhost:8501 in a browser.

WHAT THIS IS
------------
A teaching front end for everything in `mri_sim`, driven by the pre-built
k-space sample store so it starts instantly and never touches the 12 GB raw
dataset. Each tab answers one question:

    1. Acquire      -- what does undersampling do to the image?
    2. Center vs edges -- which frequencies carry what?
    3. Noise        -- what does scanner noise do, and how does it interact
                       with undersampling?
    4. Compressed sensing -- can we do better than assuming the missing data
                       was zero?
    5. Sweep        -- how do PSNR and SSIM fall off as we accelerate?
    6. Reduced FOV  -- can we scan only the region we care about? (and why
                       "keep the part of k-space where the lesion is" is not
                       a thing)
    ... plus an "About this sample" tab with the provenance of the data.

DESIGN NOTES
------------
* The heavy work is cached with `@st.cache_data`, so dragging a slider only
  recomputes what actually changed. A masked reconstruction is ~1 ms, so the
  UI keeps up with the slider.
* Every array that crosses a function boundary is centered k-space, matching
  the convention used everywhere else in the project.
* Reconstructions are displayed with a fixed [0, 1] range, never auto-scaled,
  so the brightness you see is the brightness the reconstruction has.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the project root: the app
# lives one directory down, so the project root has to be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kspace_store.store import KSpaceStore          # noqa: E402
from mri_sim import cs, kspace as ks, metrics, noise, roi  # noqa: E402

STORE_PATH = os.path.join("data", "kspace_store")

# Ratios used by the sweep tab. 1.0 is included as the "no undersampling"
# reference point.
SWEEP_RATIOS = [1.0, 0.5, 0.25, 0.125, 0.0625]


# ---------------------------------------------------------------------------
# Data access, cached
# ---------------------------------------------------------------------------


@st.cache_resource
def get_store() -> KSpaceStore:
    """The store object itself: opened once per server process."""
    return KSpaceStore(STORE_PATH)


@st.cache_data(show_spinner=False)
def load_sample(sample_id: str):
    """
    Arrays for one sample.

    Returned as plain numpy arrays rather than the Sample dataclass because
    Streamlit's cache hashes what it stores, and arrays hash cleanly.

    The k-space is promoted to complex128 here. It is stored as complex64 to
    halve the file size, but the iterative reconstruction runs hundreds of
    FFTs and is better off in double precision.
    """
    sample = get_store().load(sample_id)
    return (
        sample.image.astype(np.float64),
        sample.kspace.astype(np.complex128),
        None if sample.tumor_mask is None else sample.tumor_mask.astype(bool),
        sample.meta,
    )


@st.cache_data(show_spinner=False)
def build_mask(kind: str, shape: tuple[int, int], ratio: float, seed: int) -> np.ndarray:
    """Cached mask construction (the radial mask's binary search is the slow one)."""
    extra = {"seed": seed} if kind == "variable_density" else {}
    return ks.build_mask(kind, shape, ratio, **extra)


@st.cache_data(show_spinner=False)
def acquire(
    sample_id: str,
    kind: str,
    ratio: float,
    snr_db: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate one scan: build the mask, sample k-space, add noise, reconstruct.

    Returns (acquired k-space, reconstruction, metrics).
    """
    image, full_kspace, _, _ = load_sample(sample_id)
    mask = build_mask(kind, image.shape, ratio, seed)

    if snr_db is None:
        acquired = ks.apply_mask(full_kspace, mask)
    else:
        # Mask first, then noise: only measured points carry measurement noise.
        acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=snr_db, seed=seed)

    reconstruction = ks.from_kspace(acquired)
    return acquired, reconstruction, metrics.compute_metrics(image, reconstruction)


@st.cache_data(show_spinner=False)
def run_cs(
    sample_id: str,
    ratio: float,
    lambda_: float,
    n_iter: int,
    snr_db: float | None,
    seed: int,
) -> dict:
    """Cached compressed-sensing run (the only genuinely slow operation, ~1 s)."""
    image, full_kspace, _, _ = load_sample(sample_id)
    mask = build_mask("variable_density", image.shape, ratio, seed)

    if snr_db is None:
        acquired = ks.apply_mask(full_kspace, mask)
    else:
        acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=snr_db, seed=seed)

    result = cs.compare_with_zero_fill(
        acquired, mask, image, lambda_=lambda_, n_iter=n_iter
    )
    result["mask"] = mask
    return result


@st.cache_data(show_spinner=False)
def roi_scan(sample_id: str, center: tuple[int, int], R: int, seed: int) -> dict:
    """
    Cached reduced-FOV comparison: the four ways of spending 1/R^2 of k-space.

    Note this one deliberately ignores the sidebar's strategy/ratio/SNR: a
    reduced-FOV scan is a different acquisition, defined entirely by where the
    box is and how coarse the grid is. Its own controls live in the tab.
    """
    image, _, _, _ = load_sample(sample_id)
    return roi.compare_roi_strategies(image, center, R, seed=seed)


@st.cache_data(show_spinner=False)
def locality_demo(sample_id: str, quadrant: str) -> dict:
    """Cached "k-space is not spatially local" experiment (see roi.py)."""
    image, _, _, _ = load_sample(sample_id)
    return roi.kspace_locality_demo(image, quadrant=quadrant)


@st.cache_data(show_spinner=False)
def sweep(sample_id: str, snr_db: float | None, seed: int) -> pd.DataFrame:
    """PSNR/SSIM for every strategy at every ratio -- the summary chart."""
    rows = []
    for kind in ks.ACQUISITION_MASKS:
        for ratio in SWEEP_RATIOS:
            _, _, scores = acquire(sample_id, kind, ratio, snr_db, seed)
            mask = build_mask(kind, load_sample(sample_id)[0].shape, ratio, seed)
            rows.append({
                "strategy": ks.MASK_LABELS[kind],
                "sampling %": ks.sampling_ratio(mask) * 100.0,
                "acceleration": ks.acceleration_factor(mask),
                "PSNR (dB)": scores["psnr"],
                "SSIM": scores["ssim"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show_image(array: np.ndarray, caption: str, stretch: bool = False) -> None:
    """
    Render a float image as a grayscale panel.

    `stretch=False` pins the display range to [0, 1] so brightness is
    comparable between panels -- important, because "the reconstruction is
    darker than the original" is a real finding, not a display artifact.
    """
    data = np.asarray(array, dtype=np.float64)
    if stretch:
        low, high = float(data.min()), float(data.max())
        data = (data - low) / (high - low) if high > low else np.zeros_like(data)
    else:
        data = np.clip(data, 0.0, 1.0)

    # PNG, not Streamlit's default JPEG. This whole app is about the artifacts
    # that appear when high frequencies are discarded -- rendering the panels
    # through a lossy codec that does exactly that would add a second, fake
    # layer of the very effect being demonstrated.
    st.image(
        data, caption=caption, use_container_width=True,
        clamp=True, output_format="PNG",
    )


def show_kspace(kspace: np.ndarray, caption: str) -> None:
    """Render k-space as log(1 + |K|), the only way its dynamic range is visible."""
    log_magnitude = np.log1p(np.abs(kspace))
    peak = float(log_magnitude.max())
    show_image(log_magnitude / peak if peak > 0 else log_magnitude, caption)


def show_error(original: np.ndarray, reconstruction: np.ndarray, caption: str) -> None:
    """Absolute difference map, auto-scaled (the caption reports the true peak)."""
    error = np.abs(original - reconstruction)
    show_image(error, f"{caption} (peak {error.max():.3f})", stretch=True)


def show_with_box(array: np.ndarray, box, caption: str) -> None:
    """
    Render an image with the excited ROI box outlined in orange.

    Drawn by hand into an RGB copy rather than with matplotlib, because these
    panels are `st.image` calls, not figures.
    """
    data = np.clip(np.asarray(array, dtype=np.float64), 0.0, 1.0)
    overlay = np.stack([data] * 3, axis=-1)

    outline = [1.0, 0.41, 0.20]
    y1, x1 = box.y0 + box.size - 1, box.x0 + box.size - 1
    overlay[box.y0, box.x0:x1 + 1] = outline      # top edge
    overlay[y1, box.x0:x1 + 1] = outline          # bottom edge
    overlay[box.y0:y1 + 1, box.x0] = outline      # left edge
    overlay[box.y0:y1 + 1, x1] = outline          # right edge

    st.image(overlay, caption=caption, use_container_width=True,
             clamp=True, output_format="PNG")


def metric_row(scores: dict, baseline: dict | None = None) -> None:
    """PSNR and SSIM as metric cards, optionally with a delta against a baseline."""
    left, right = st.columns(2)
    left.metric(
        "PSNR", f"{scores['psnr']:.2f} dB",
        None if baseline is None else f"{scores['psnr'] - baseline['psnr']:+.2f} dB",
    )
    right.metric(
        "SSIM", f"{scores['ssim']:.4f}",
        None if baseline is None else f"{scores['ssim'] - baseline['ssim']:+.4f}",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MRI k-Space Simulator",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 MRI k-Space Reconstruction Simulator")
st.caption(
    "How an MRI scanner acquires data in the Fourier domain, and what happens "
    "to the image when you speed the scan up by measuring less of it."
)

try:
    store = get_store()
except FileNotFoundError:
    st.error(
        "No k-space store found. Build it first:\n\n"
        "```\npython -m kspace_store.build\n```"
    )
    st.stop()

# --- Sidebar: the scan setup ------------------------------------------------

with st.sidebar:
    st.header("Scan setup")

    # Group the sample picker by collection so the list is navigable.
    collection = st.selectbox(
        "Collection",
        store.collections(),
        format_func=lambda name: {
            "BrainTumorDataPublic": "Brain tumours (with masks)",
            "NINS_Dataset": "Brain pathologies",
            "MRI_Dataset": "Spine (real DICOM, T1/T2)",
            "3D_volumetric_imaging": "Other anatomy & CT",
        }.get(name, name),
    )
    records = store.records(collection=collection)
    titles = {record["id"]: record["title"] for record in records}
    sample_id = st.selectbox(
        "Subject", list(titles), format_func=lambda key: titles[key]
    )

    st.divider()

    strategy = st.selectbox(
        "Sampling strategy",
        ks.ACQUISITION_MASKS,
        format_func=lambda kind: ks.MASK_LABELS[kind],
    )
    ratio = st.slider(
        "k-space sampled", min_value=0.02, max_value=1.0, value=0.25, step=0.01,
        format="%.2f",
        help="Fraction of k-space the scanner acquires. 0.25 means a 4x faster scan.",
    )
    st.caption(f"→ **{1 / ratio:.1f}× acceleration** (scan takes {ratio * 100:.0f}% of the time)")

    st.divider()

    add_noise = st.checkbox(
        "Simulate scanner noise", value=False,
        help="Complex Gaussian noise added to the k-space samples that were "
             "actually measured.",
    )
    snr_db = st.slider(
        "k-space SNR (dB)", min_value=0.0, max_value=50.0, value=25.0, step=1.0,
        disabled=not add_noise,
        help="40 dB is a clean clinical scan, 20 dB is visibly grainy, 10 dB is bad.",
    ) if add_noise else None

    seed = st.number_input(
        "Random seed", min_value=0, max_value=9999, value=0, step=1,
        help="Controls the random mask draw and the noise realisation.",
    )

    st.divider()
    st.caption(
        f"Store: {len(store)} samples at "
        f"{store.manifest['resolution']}×{store.manifest['resolution']}"
    )

# --- Load the chosen subject ------------------------------------------------

image, full_kspace, tumor_mask, meta = load_sample(sample_id)
mask = build_mask(strategy, image.shape, ratio, int(seed))
acquired, reconstruction, scores = acquire(
    sample_id, strategy, ratio, snr_db, int(seed)
)

tabs = st.tabs([
    "1 · Acquire",
    "2 · Centre vs edges",
    "3 · Noise",
    "4 · Compressed sensing",
    "5 · Sweep",
    "6 · Reduced FOV (ROI)",
    "ℹ️ About this sample",
])

# ---------------------------------------------------------------------------
# Tab 1: the main pipeline
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("The pipeline, end to end")
    st.markdown(
        "The scanner measures **k-space**, not an image. Skipping samples makes "
        "the scan faster; the inverse FFT then has to assume the missing samples "
        "were zero, and that wrong assumption is what you see as artifacts."
    )

    columns = st.columns(5)
    with columns[0]:
        show_image(image, "1. Ground truth")
    with columns[1]:
        show_kspace(full_kspace, "2. Full k-space, log|K|")
    with columns[2]:
        show_image(mask, f"3. Mask — {ks.sampling_ratio(mask) * 100:.1f}% acquired")
    with columns[3]:
        show_kspace(acquired, "4. What the scanner got")
    with columns[4]:
        show_image(reconstruction, "5. Reconstruction (zero-filled)")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Reconstruction quality**")
        metric_row(scores)
        st.caption(
            f"Acquired {ks.sampling_ratio(mask) * 100:.1f}% of k-space → "
            f"{ks.acceleration_factor(mask):.1f}× faster scan."
        )
    with right:
        show_error(image, reconstruction, "Absolute error")

    st.info(
        {
            "cartesian": "**Cartesian**: skipping whole lines folds the image onto "
                         "itself — the ghosts are crisp copies of the anatomy, "
                         "shifted by FOV/acceleration. Coherent artifacts like these "
                         "are the hardest kind to remove, because they look like real "
                         "structure.",
            "radial": "**Radial**: spokes oversample the centre and leave gaps that "
                      "widen outward, so the error appears as streaks radiating from "
                      "bright edges. Radial is also famously robust to motion, since "
                      "every spoke re-measures the centre.",
            "variable_density": "**Random variable-density**: the error is spread out "
                                "as incoherent, noise-like grain instead of structured "
                                "ghosts. That is exactly the property compressed "
                                "sensing needs — see tab 4.",
        }[strategy]
    )

# ---------------------------------------------------------------------------
# Tab 2: centre vs edges
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Which part of k-space carries what?")
    st.markdown(
        "Both reconstructions below use the **same number of samples**. The only "
        "difference is *where* in k-space those samples were taken."
    )

    demo_ratio = st.slider(
        "Fraction of k-space kept (in both cases)",
        min_value=0.01, max_value=0.5, value=0.10, step=0.01, format="%.2f",
    )

    center_mask = ks.build_mask("center_only", image.shape, demo_ratio)
    edges_mask = ks.build_mask("edges_only", image.shape, demo_ratio)
    center_recon = ks.from_kspace(ks.apply_mask(full_kspace, center_mask))
    edges_recon = ks.from_kspace(ks.apply_mask(full_kspace, edges_mask))

    left, right = st.columns(2)

    with left:
        st.markdown("#### Centre only — a low-pass filter")
        inner = st.columns(3)
        with inner[0]:
            show_image(center_mask, "Mask")
        with inner[1]:
            show_image(center_recon, "As reconstructed")
        with inner[2]:
            show_image(center_recon, "Contrast stretched", stretch=True)
        metric_row(metrics.compute_metrics(image, center_recon))
        st.success(
            f"Mean brightness **{center_recon.mean():.3f}** vs {image.mean():.3f} "
            "for the original — contrast and shape are intact, only fine detail is "
            "lost. The faint rings around sharp edges are **Gibbs ringing**, from "
            "truncating the Fourier series at the rim of the disc."
        )

    with right:
        st.markdown("#### Edges only — a high-pass filter")
        inner = st.columns(3)
        with inner[0]:
            show_image(edges_mask, "Mask")
        with inner[1]:
            show_image(edges_recon, "As reconstructed")
        with inner[2]:
            show_image(edges_recon, "Contrast stretched", stretch=True)
        metric_row(metrics.compute_metrics(image, edges_recon))
        st.error(
            f"Mean brightness **{edges_recon.mean():.4f}** — essentially black. "
            "Throwing away the centre throws away the DC term, i.e. the average "
            "brightness of the whole image, along with every slowly-varying "
            "structure. Stretched, it is an edge map."
        )

    energy = meta["stats"]["energy_within_r0.1"]
    st.info(
        f"For this sample, **{energy * 100:.1f}%** of all k-space energy sits inside "
        "the central 10% radius — about 1% of the samples. That is why every "
        "realistic mask in tab 1 protects the centre."
    )

# ---------------------------------------------------------------------------
# Tab 3: noise
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Scanner noise lives in k-space")
    st.markdown(
        "Real noise is added to the **measured samples**, not to the finished "
        "image. It is complex (the receiver has an I and a Q channel) and white "
        "— the same power at every frequency. Since the outer samples are tiny "
        "and the centre is huge, the same noise destroys fine detail long before "
        "it touches overall contrast."
    )

    demo_snr = st.slider(
        "k-space SNR (dB)", min_value=0.0, max_value=50.0, value=20.0, step=1.0,
        key="noise_tab_snr",
    )

    noisy_full = noise.add_kspace_noise(full_kspace, snr_db=demo_snr, seed=int(seed))
    noisy_recon = ks.from_kspace(noisy_full)
    noisy_acquired = noise.simulate_acquisition(
        full_kspace, mask, snr_db=demo_snr, seed=int(seed)
    )
    noisy_under = ks.from_kspace(noisy_acquired)

    columns = st.columns(4)
    with columns[0]:
        show_image(image, "Ground truth")
        st.caption("noiseless, fully sampled")
    with columns[1]:
        show_image(noisy_recon, "Noisy, fully sampled")
        metric_row(metrics.compute_metrics(image, noisy_recon))
    with columns[2]:
        show_image(reconstruction, f"Undersampled {ratio * 100:.0f}%, noiseless")
        metric_row(scores)
    with columns[3]:
        show_image(noisy_under, f"Undersampled {ratio * 100:.0f}% + noise")
        metric_row(metrics.compute_metrics(image, noisy_under))

    st.caption(
        f"Verification: requested {demo_snr:.0f} dB, measured "
        f"{noise.measured_snr_db(full_kspace, noisy_full):.2f} dB on the full k-space."
    )

    with st.expander("Why does undersampling let in *less* total noise?"):
        st.markdown(
            "Noise enters once per measurement, so a mask that keeps 25% of "
            "k-space also admits about 25% of the noise energy — the familiar "
            "`SNR ∝ √N` of MRI. That does **not** make fast scans cleaner: you "
            "lose signal and gain artifacts at the same time. The honest "
            "statement is that a faster scan is noisier *per unit of signal*.\n\n"
            "Notice also that the magnitude operation turns zero-mean complex "
            "noise into strictly positive **Rician** noise, which is why the "
            "background of a noisy MRI image is a faint grey haze rather than "
            "true black."
        )

# ---------------------------------------------------------------------------
# Tab 4: compressed sensing
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Compressed sensing: a better guess at the missing data")
    st.markdown(
        "Zero-filling assumes every unmeasured point was zero. Compressed "
        "sensing instead asks: *of all the images consistent with what we "
        "measured, which one is the sparsest in a wavelet basis?* It needs the "
        "**random variable-density** mask, because its artifacts are incoherent "
        "— Cartesian ghosts are just as sparse as real anatomy, so no sparsity "
        "prior can tell them apart."
    )

    controls = st.columns(3)
    cs_ratio = controls[0].slider(
        "k-space sampled", min_value=0.03, max_value=0.5, value=0.125, step=0.005,
        format="%.3f", key="cs_ratio",
    )
    cs_lambda = controls[1].slider(
        "λ — sparsity strength", min_value=0.002, max_value=0.10, value=0.01,
        step=0.002, format="%.3f",
        help="Larger = sparser = smoother. Too large and real anatomy is "
             "thresholded away.",
    )
    cs_iters = controls[2].slider(
        "Iterations", min_value=10, max_value=150, value=80, step=10,
    )

    with st.spinner("Running FISTA..."):
        result = run_cs(sample_id, cs_ratio, cs_lambda, cs_iters, snr_db, int(seed))

    columns = st.columns(3)
    with columns[0]:
        show_image(result["mask"], f"Mask — {ks.sampling_ratio(result['mask']) * 100:.1f}%")
        st.caption(f"{1 / ks.sampling_ratio(result['mask']):.1f}× acceleration")
    with columns[1]:
        show_image(result["zero_fill_image"], "Zero-filled (linear, instant)")
        metric_row(result["zero_fill_metrics"])
    with columns[2]:
        show_image(result["cs_image"], "Compressed sensing (FISTA)")
        metric_row(result["cs_metrics"], baseline=result["zero_fill_metrics"])

    history = pd.DataFrame(result["history"])
    if "psnr" in history:
        chart = history[["iteration", "psnr"]].rename(columns={"psnr": "compressed sensing"})
        chart["zero-fill baseline"] = result["zero_fill_metrics"]["psnr"]
        st.line_chart(chart, x="iteration", y=["compressed sensing", "zero-fill baseline"])

    delta_psnr = result["cs_metrics"]["psnr"] - result["zero_fill_metrics"]["psnr"]
    delta_ssim = result["cs_metrics"]["ssim"] - result["zero_fill_metrics"]["ssim"]
    if delta_psnr > 0:
        st.success(
            f"CS wins by **{delta_psnr:+.2f} dB** PSNR and **{delta_ssim:+.4f}** SSIM "
            "from exactly the same measurements."
        )
    else:
        st.warning(
            f"At this ratio CS trades **{delta_psnr:.2f} dB** of PSNR for "
            f"**{delta_ssim:+.4f}** SSIM. That is the expected behaviour when "
            "undersampling is mild: zero-filling is already close to perfect, so "
            "the sparsity prior costs more than it gains. Push the ratio below "
            "~0.15 and CS pulls ahead on both."
        )

# ---------------------------------------------------------------------------
# Tab 5: the sweep
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Quality versus acceleration")
    st.markdown(
        "Every strategy, every ratio, scored against the ground truth — the "
        "quantitative version of tab 1."
    )

    with st.spinner("Sweeping..."):
        table = sweep(sample_id, snr_db, int(seed))

    left, right = st.columns(2)
    with left:
        st.markdown("**PSNR (dB) vs sampling %**")
        st.line_chart(
            table.pivot(index="sampling %", columns="strategy", values="PSNR (dB)")
        )
    with right:
        st.markdown("**SSIM vs sampling %**")
        st.line_chart(
            table.pivot(index="sampling %", columns="strategy", values="SSIM")
        )

    st.dataframe(
        table.style.format({
            "sampling %": "{:.1f}",
            "acceleration": "{:.1f}×",
            "PSNR (dB)": "{:.2f}",
            "SSIM": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_id}_metrics.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 6: reduced field of view -- scanning only the part that matters
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Can we scan only the bit we care about?")
    st.markdown(
        "Yes — but **not** by keeping the part of k-space \"where the lesion "
        "is\". There is no such part: every k-space sample is a measurement of "
        "every pixel. The way to do it is to stop the rest of the anatomy "
        "producing signal at all, with a spatially selective **RF excitation**, "
        "and then take advantage of the smaller field of view that leaves."
    )

    with st.expander("The two Fourier relationships this rests on", expanded=False):
        st.markdown(
            "| Quantity | Controls | Relationship |\n"
            "|---|---|---|\n"
            "| `dk` — spacing between samples | field of view | `FOV = 1 / dk` |\n"
            "| `k_max` — how far out you go | resolution | `dx = 1 / (2·k_max)` |\n\n"
            "Tab 2's centre-only mask turns the **second** knob: it shrinks "
            "`k_max`, so it costs resolution. Reduced-FOV imaging turns the "
            "**first**: with only a small box excited, the samples can be "
            "spaced `R` times further apart without the object folding onto "
            "itself, and `k_max` — so resolution — is untouched. Sampling "
            "every `R`-th point in both directions measures `1/R²` of "
            "k-space: at R = 4 that is 6.25%, a **16× faster scan**, at full "
            "resolution inside the box."
        )

    # --- Controls: where the box goes and how coarse the grid is ------------
    factors = roi.reduction_factors(image.shape)
    control_left, control_mid, control_right = st.columns([1, 1, 1])

    with control_left:
        R = st.select_slider(
            "Reduction factor R",
            options=factors,
            value=roi.DEFAULT_REDUCTION if roi.DEFAULT_REDUCTION in factors else factors[0],
            help="R must divide the image size exactly, otherwise the aliasing "
                 "period N/R is not a whole number of pixels and the "
                 "reconstruction silently degrades. Only legal values are offered.",
        )
        st.caption(
            f"→ **{R * R}× faster** · {100.0 / (R * R):.2f}% of k-space · "
            f"box {image.shape[0] // R}×{image.shape[1] // R} px"
        )

    # Default the box to the expert tumour segmentation when the sample has
    # one -- that is the realistic case: a radiologist points at the lesion.
    if tumor_mask is not None and tumor_mask.any():
        default_center = roi.roi_center_from_mask(tumor_mask)
        center_note = "defaults to the centroid of the expert tumour mask"
    else:
        default_center = (image.shape[0] // 2, image.shape[1] // 2)
        center_note = "this sample has no segmentation, so the default is the image centre"

    with control_mid:
        center_y = st.slider("ROI centre — row (y)", 0, image.shape[0] - 1,
                             int(default_center[0]))
    with control_right:
        center_x = st.slider("ROI centre — column (x)", 0, image.shape[1] - 1,
                             int(default_center[1]))
    st.caption(f"ROI centre ({center_y}, {center_x}) — {center_note}. "
               "The box is clamped to stay inside the image.")

    comparison = roi_scan(sample_id, (int(center_y), int(center_x)), int(R), int(seed))
    box = comparison["box"]
    reduced = comparison["variants"][0]

    # --- The scan itself ----------------------------------------------------
    st.markdown("**The inner-volume scan, step by step**")
    columns = st.columns(5)
    with columns[0]:
        show_with_box(image, box, "1. Target, with the box to excite")
    with columns[1]:
        show_image(reduced["object"], "2. After the RF pulse: only the box has signal")
    with columns[2]:
        show_image(reduced["mask"], f"3. Every {R}th sample, full k_max")
    with columns[3]:
        show_kspace(reduced["kspace"], "4. What the scanner measured")
    with columns[4]:
        show_image(reduced["reconstruction"],
                   f"5. Reconstruction (×{R * R} density compensation)")

    st.caption(
        "Panel 5 tiles: outside the box the reconstruction is just periodic "
        f"replicas of the ROI, {image.shape[0] // R} pixels apart. That is "
        "harmless — nothing out there was excited, so there is no signal to "
        "get wrong. Score the box, not the picture."
    )

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Inside the ROI**")
        metric_row({"psnr": reduced["psnr"], "ssim": reduced["ssim"]})
        st.caption(
            "Scored with `metrics.compute_metrics_in_roi`. A PSNR in the "
            "hundreds means the only error left is floating-point round-off: "
            "inside the box this is not an approximation, it is a complete, "
            "critically-sampled measurement."
        )
    with right:
        crop_left, crop_right = st.columns(2)
        with crop_left:
            show_image(reduced["reconstruction"][box.slices], "ROI, reconstructed")
        with crop_right:
            show_image(image[box.slices], "ROI, ground truth")

    # --- The same samples spent four different ways -------------------------
    st.divider()
    st.markdown("**Same scan time, four ways to spend it**")
    st.markdown(
        "Every row below measures the same fraction of k-space. Only the first "
        "one excites the box; the last one uses *identical* samples to the "
        "first with the RF excitation removed."
    )

    table = pd.DataFrame([
        {
            "strategy": variant["label"],
            "k-space %": variant["ratio"] * 100.0,
            "acceleration": variant["acceleration"],
            "PSNR in ROI (dB)": variant["psnr"],
            "SSIM in ROI": variant["ssim"],
        }
        for variant in comparison["variants"]
    ])
    st.dataframe(
        table.style.format({
            "k-space %": "{:.2f}",
            "acceleration": "{:.1f}×",
            "PSNR in ROI (dB)": "{:.2f}",
            "SSIM in ROI": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    crops = st.columns(len(comparison["variants"]))
    for column, variant in zip(crops, comparison["variants"]):
        with column:
            # The no-suppression row sums R^2 folded copies, so its values run
            # past 1.0; stretch it or it renders as a white square.
            stretch = float(variant["reconstruction"].max()) > 1.5
            show_image(
                variant["reconstruction"][box.slices],
                f"{variant['key']} — {variant['psnr']:.1f} dB"
                + (" (display stretched)" if stretch else ""),
                stretch=stretch,
            )

    st.error(
        "**The last column is the point.** Drop the RF excitation and the "
        "whole head is still producing signal, so an R-fold coarse grid folds "
        f"{R * R} pieces of anatomy directly on top of the ROI. PSNR goes "
        "*negative* — the error is larger than the signal. The excitation is "
        "not an optimisation on top of the method; it **is** the method."
    )

    # --- What the scanner would really hand back ---------------------------
    with st.expander("What the scanner actually returns (compact reconstruction)"):
        _, kspace_roi, _ = roi.reduced_fov_acquire(
            image, (int(center_y), int(center_x)), int(R)
        )
        wrapped = roi.compact_reconstruct(kspace_roi, int(R))
        centered = roi.compact_reconstruct(kspace_roi, int(R), box)
        compact_scores = metrics.compute_metrics(image[box.slices], centered)

        st.markdown(
            f"Only `(N/R)² = {box.size}×{box.size}` samples were measured, so "
            f"the natural output is a {box.size}×{box.size} image — same "
            "resolution, smaller field of view — rather than a zero-filled "
            "full-size one. It comes out **circularly shifted**, because "
            "decimating k-space keeps the full FOV's origin as the small FOV's "
            f"origin; rolling by ({box.y0 % box.size}, {box.x0 % box.size}) "
            "pixels puts the anatomy back (the shift theorem, applied after "
            "the transform instead of as a phase ramp before it)."
        )
        small_columns = st.columns(3)
        with small_columns[0]:
            show_image(wrapped, f"Raw {box.size}×{box.size} output — wrapped")
        with small_columns[1]:
            show_image(centered, "Wrap undone")
        with small_columns[2]:
            show_image(image[box.slices], "Ground truth")
        metric_row(compact_scores)

    # --- The misconception, measured ---------------------------------------
    st.divider()
    st.markdown("**Why \"just keep that corner of k-space\" cannot work**")

    locality = locality_demo(sample_id, "top-left")
    errors = locality["quadrant_errors"]

    locality_columns = st.columns(4)
    with locality_columns[0]:
        show_kspace(locality["kspace_damaged"], "k-space, top-left quadrant deleted")
    with locality_columns[1]:
        show_image(locality["reconstruction"], "Reconstruction")
    with locality_columns[2]:
        show_error(image, locality["reconstruction"], "Where the error landed")
    with locality_columns[3]:
        st.markdown("Mean `|error|` per **image** quadrant:")
        st.dataframe(
            pd.DataFrame(
                errors,
                index=["top", "bottom"],
                columns=["left", "right"],
            ).style.format("{:.4f}"),
            use_container_width=True,
        )
        st.caption(
            f"Worst / best = {errors.max() / errors.min():.2f}×, not infinity."
        )

    st.info(
        "One quarter of k-space was deleted, and the damage appears across the "
        "**whole** image rather than in one quadrant. Every k-space sample is "
        "an inner product of the entire object with one global sinusoid, so "
        "every sample carries every pixel. Position in the image lives in the "
        "*phase* relationships between samples, not in where the samples sit "
        "in k-space — which is why knowing where the lesion is tells you "
        "nothing about which samples to keep, and why the answer had to come "
        "from the excitation instead."
    )

# ---------------------------------------------------------------------------
# Tab 7: provenance
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader(meta["title"])

    left, right = st.columns([1, 2])
    with left:
        show_image(image, "Ground truth")
        if tumor_mask is not None:
            # Red overlay on the expert tumour mask.
            overlay = np.stack([image] * 3, axis=-1)
            overlay[tumor_mask] = [1.0, 0.25, 0.25]
            st.image(
                overlay, caption="Expert tumour mask", use_container_width=True,
                clamp=True, output_format="PNG",
            )

    with right:
        st.markdown("**Where this came from**")
        st.markdown(
            f"- Collection: `{meta['collection']}`\n"
            f"- Source file: `{meta['source_file']}`\n"
            f"- Stored shape: {meta['shape'][0]}×{meta['shape'][1]}\n"
            f"- Tags: {', '.join(meta['tags'])}"
        )
        st.caption(meta["collection_note"])

        st.markdown("**Acquisition**")
        acquisition = {k: v for k, v in meta["acquisition"].items() if v not in (None, "")}
        st.json(acquisition, expanded=True)

        st.markdown("**Derived statistics**")
        stats = meta["stats"]
        st.markdown(
            f"- Energy inside the central 10% radius: **{stats['energy_within_r0.1'] * 100:.1f}%**\n"
            f"- Energy inside the central 25% radius: **{stats['energy_within_r0.25'] * 100:.1f}%**\n"
            f"- k-space dynamic range: **{stats['kspace_dynamic_range_db']:.0f} dB** "
            "(peak / median magnitude — why k-space is always shown on a log scale)\n"
            f"- Hermitian asymmetry: **{stats['hermitian_asymmetry']:.2f}** "
            "(≈0 would mean a real-valued image, where half of k-space is redundant)"
        )

    with st.expander("How this k-space was made, and what is simulated"):
        st.markdown(
            "This project starts from reconstructed images and runs a **forward "
            "FFT** to manufacture k-space. A real scanner measures k-space "
            "directly; everything downstream of that point behaves identically.\n\n"
            "The source files are *magnitude* images, so the original scanner "
            "phase no longer exists. A smooth synthetic phase map (a B₀-like "
            "quadratic bowl, a gradient ramp, and coil-like ripples) is applied "
            "before the FFT. Without it the k-space would be perfectly "
            "Hermitian-symmetric, half the data would be a free copy of the "
            "other half, and any partial-Fourier demonstration would be "
            "unrealistically perfect.\n\n"
            f"Stored as: `kspace = fftshift(fft2(image · exp(i·phase)))`, "
            f"complex64, {meta['shape'][0]}×{meta['shape'][1]}."
        )
