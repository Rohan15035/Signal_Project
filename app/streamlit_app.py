"""Interactive front end for mri_sim.

    streamlit run app/streamlit_app.py      # from the project root

Reads the pre-built k-space store, so it starts instantly and never touches
the raw dataset. One tab per question: what undersampling does, which
frequencies carry what, what noise does, whether CS beats zero-filling, and
how PSNR/SSIM fall off with acceleration.

Heavy work is cached so dragging a slider only recomputes what changed.
Reconstructions display on a fixed [0, 1] range, never auto-scaled, so the
brightness shown is the brightness they have.
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
from mri_sim import cs, kspace as ks, metrics, noise  # noqa: E402

STORE_PATH = os.path.join("data", "kspace_store")

# Ratios used by the sweep tab. 1.0 is included as the "no undersampling"
# reference point.
SWEEP_RATIOS = [1.0, 0.5, 0.25, 0.125, 0.0625]


# Data access, cached


@st.cache_resource
def get_store() -> KSpaceStore:
    """The store object itself: opened once per server process."""
    return KSpaceStore(STORE_PATH)


@st.cache_data(show_spinner=False)
def load_sample(sample_id: str):
    """Arrays for one sample, as plain numpy so Streamlit's cache can hash them.

    k-space is stored complex64 to halve the file size but promoted to
    complex128 here, since the iterative reconstruction runs hundreds of FFTs.
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
    """One scan: mask, sample, noise, reconstruct -> (k-space, image, metrics)."""
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


# Display helpers


def show_image(array: np.ndarray, caption: str, stretch: bool = False) -> None:
    """Render a float image as a grayscale panel.

    stretch=False pins the range to [0, 1] so brightness is comparable between
    panels -- "the reconstruction is darker" is a real finding, not a display
    artifact.
    """
    data = np.asarray(array, dtype=np.float64)
    if stretch:
        low, high = float(data.min()), float(data.max())
        data = (data - low) / (high - low) if high > low else np.zeros_like(data)
    else:
        data = np.clip(data, 0.0, 1.0)

    # PNG, not Streamlit's default JPEG: a lossy codec discards high
    # frequencies, which is the very effect this app is demonstrating.
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


# Page

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

# Sidebar: the scan setup

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

# Load the chosen subject

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
    "ℹ️ About this sample",
])

# Tab 1: the main pipeline

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

# Tab 2: centre vs edges

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

# Tab 3: noise

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

# Tab 4: compressed sensing

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

# Tab 5: the sweep

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

# Tab 6: provenance

with tabs[5]:
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
