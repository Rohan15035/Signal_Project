# MRI k-Space Reconstruction Simulator — Full Project Description

*Course project, CSE 220 (Signals and Systems). This document is a complete
factual description of the project, written to be used as source context for
generating a slide deck. Every number quoted here was produced by running the
code in this repository.*

---

## 1. Identity

| Field | Value |
|---|---|
| Title | MRI k-Space Reconstruction Simulator |
| Course | CSE 220 — Signals and Systems |
| Domain | 2-D Fourier analysis applied to medical imaging |
| Stack | Python, NumPy, SciPy, scikit-image, PyWavelets, Matplotlib, Streamlit, pandas |
| Build-time only | h5py, pydicom, nibabel, Pillow (used once, to convert the raw dataset) |
| Size | ~3,900 lines of commented Python across 3 packages + a 40-sample data store |
| Deliverables | A Python library, a batch CLI pipeline, a 6-tab interactive web app, a curated k-space dataset, and generated figures |

**One-sentence pitch:** An MRI scanner does not photograph the body — it
measures the 2-D Fourier transform of a slice, point by point, and the image
is only recovered afterwards by an inverse FFT; this project rebuilds that
entire pipeline in software so you can delete parts of the measurement, watch
the image break in a specific and predictable way, and measure exactly how
much quality a faster scan costs.

**The question the project answers:** MRI scans are slow (minutes per
sequence), and scan time is directly proportional to how many k-space samples
you acquire. So: *if you acquire only a fraction of k-space, how bad is the
image, and does it depend on which fraction you skip?* The answer is that it
depends enormously — the same 12.5% of k-space produces very different images
depending on the sampling pattern, and a smarter reconstruction can recover
part of what was lost.

---

## 2. The physics and signal theory (background)

### What a scanner actually measures

A gradient echo / spin echo sequence applies spatial-encoding magnetic field
gradients while the receiver coil digitises an RF voltage. Because of those
gradients, sample number *n* of that signal corresponds **exactly** to one
point in k-space — the 2-D Fourier transform of the slice being imaged:

    S(kx, ky) = ∫∫ ρ(x, y) · exp(−j2π(kx·x + ky·y)) dx dy

The image is recovered by an inverse 2-D Fourier transform. So the raw
measurement is a frequency-domain signal, and the picture on the radiologist's
screen is a reconstruction, not a capture.

### Key properties of k-space used throughout

- **The centre is low frequency.** After fftshift, DC sits at the middle of
  the array. The centre carries contrast, brightness and overall shape; the
  edges carry fine detail and sharp boundaries.
- **Energy is extremely concentrated.** Measured across all 40 samples in this
  project's store: **92.6% of all k-space energy (mean; range 73.6–98.9%)
  sits inside the central 10% radius**, which is roughly 1% of the samples.
  This single fact justifies why every realistic sampling mask in the project
  protects the centre.
- **The dynamic range is huge.** Peak-to-median magnitude is **62–85 dB**
  (mean 73 dB) across the store, which is why k-space is always displayed as
  log(1 + |K|) — on a linear scale you see one bright dot and nothing else.
- **One Cartesian line = one unit of scan time.** A Cartesian acquisition
  fills one k-space row per pulse repetition (TR). Skipping rows is therefore
  a *real* time saving, not a bookkeeping trick. Sampling 25% of k-space = a
  4× faster scan.
- **Undersampling below Nyquist causes aliasing.** The inverse FFT has no way
  to know a sample was skipped — it assumes the unmeasured value was zero.
  That wrong assumption is precisely what appears as an artifact.

### Why this project starts from an image, not raw scanner data

Real raw k-space is not publicly distributed for most datasets. So the
pipeline runs *forward* first: take a real clinical image, apply a synthetic
phase map, forward-FFT it to manufacture k-space, and from that point onward
everything — masking, noise, reconstruction — is identical to what a scanner
faces. This is stated openly as a limitation rather than hidden (see §9).

The synthetic phase matters: source images are *magnitude* images, so their
k-space would be perfectly Hermitian-symmetric and half the data would be a
free copy of the other half. A smooth synthetic phase map (a B0-like quadratic
bowl + a linear gradient ramp + coil-like ripples) is applied before the FFT
to break that symmetry, exactly as real scanner physics does. Stored as
kspace = fftshift(fft2(image · exp(j·phase))).

---

## 3. Architecture — three layers

    Layer 3   app/streamlit_app.py      interactive 6-tab web demo
                     |
    Layer 2   kspace_store/             40 curated clinical samples, pre-FFT'd
                     |
    Layer 1   mri_sim/                  the simulator library (pure NumPy)
                     |
              main.py                   batch CLI: runs the full sweep, saves figures

### Layer 1 — mri_sim/, the simulator library

| Module | Lines | Responsibility |
|---|---|---|
| kspace.py | 322 | Forward/inverse FFT, all five sampling masks, zero-filled reconstruction, sampling-ratio and acceleration helpers |
| io_utils.py | 71 | Image loading — Shepp-Logan phantom by default, or any grayscale file; normalises to float64 in [0, 1] |
| metrics.py | 57 | PSNR and SSIM wrappers with data_range handled correctly |
| noise.py | 100 | Complex Gaussian scanner noise added *in k-space*, at a target SNR |
| cs.py | 224 | Compressed sensing by ISTA/FISTA with a wavelet sparsity prior |
| motion.py | 238 | Patient motion via the Fourier shift theorem — three motion models, plus a radial-aware variant |
| visualize.py | 424 | All Matplotlib figures: 4-panel comparisons, mask gallery, metric summaries |

Design invariant: **every k-space array in the package is fftshift-centred**,
so DC is at [ny//2, nx//2] and masks can simply reason about distance from
the middle. from_kspace undoes that with ifftshift (not fftshift — they
differ for odd-sized arrays) before calling ifft2.

### Layer 2 — kspace_store/, the dataset pipeline

A separate 1,563-line package that converts a **12 GB, 43,622-file raw Kaggle
dataset** into a clean, self-describing store of 40 samples at 256×256
(43.5 MB total). Four different raw formats are read: HDF5 .mat, JPEG,
DICOM .ima, and NIfTI .nii.

Each sample .npz holds kspace (complex64, centred), image (float32 ground
truth), phase (float32), and tumor_mask (uint8, where available).
manifest.json records, per sample: title, collection, the exact raw source
file it came from, tags, acquisition metadata (TE/TR/field strength/pixel
spacing for the DICOM samples; diagnosis for the pathology samples), and
derived statistics (energy within radii 0.02/0.05/0.1/0.25/0.5, Hermitian
asymmetry, k-space dynamic range in dB).

**Store contents (40 samples, 4 collections):**

| Collection | n | What it is |
|---|---|---|
| BrainTumorDataPublic | 12 | Contrast-enhanced T1 brain slices **with expert-drawn tumour masks** (meningioma, glioma, pituitary) |
| NINS_Dataset | 14 | Clinical brain MRI across pathologies: normal, glioma, meningioma, stroke (infarct & haemorrhage), hydrocephalus, atrophy, white-matter disease, Chiari I malformation, cerebral abscess |
| MRI_Dataset | 8 | Real anonymised Siemens 1.5 T lumbar-spine studies, DICOM, T1/T2 × sagittal/axial |
| 3D_volumetric_imaging | 6 | Knee MRI, abdominal MRI, jaw CT, walnut micro-CT, Utah teapot phantom |

Important separation of concerns: **reading** the finished store needs NumPy
only; h5py/pydicom/nibabel are build-time dependencies that a deployed app
never touches.

### Layer 3 — app/streamlit_app.py, the interactive demo

615 lines. Six tabs, sidebar controls shared across all of them (subject
picker grouped by collection, sampling strategy, sampling-ratio slider with
live acceleration readout, optional noise with an SNR slider, random seed).
Heavy operations are cached with st.cache_data, so dragging a slider
recomputes only what changed. Reconstructions are displayed on a fixed [0, 1]
range, never auto-scaled, so the brightness shown is the brightness the image
actually has — this matters for the centre-vs-edges demo.

---

## 4. The five sampling strategies

All take a target sampling ratio and return a binary mask the same shape as
k-space. The first three model real accelerated acquisitions; the last two
exist purely to prove a point.

### (a) Cartesian — cartesian_mask()

Keeps every Nth horizontal line **plus a fully-sampled block around the
centre** (default centre fraction 0.32 × ratio, the fastMRI convention).
The centre block is not optional: undersample the centre and the image is
unusable; undersample the edges and it merely gets blurrier.

*Artifact signature:* **coherent ghosting** — sharp, recognisable copies of
the anatomy displaced by FOV/acceleration, because regular undersampling in
k-space is multiplication by an impulse train, which is convolution with an
impulse train in the image domain. These are the hardest artifacts to remove,
because they look exactly like real structure.

### (b) Radial — radial_mask()

Samples along spokes through the k-space centre, at angles evenly spaced over
[0, π) (a spoke and its 180° rotation are identical). Spoke count for a target
ratio is found by bisection. Every spoke crosses the centre, so the low
frequencies are densely covered for free — and each spoke re-measures the
centre, which is why radial is used clinically for motion-tolerant scans.

*Artifact signature:* **incoherent streaks** radiating from bright edges,
because the gaps between spokes widen with radius.

*Honest caveat implemented as a comment in the code:* this is a rasterised
approximation. Real radial trajectories land off-grid and require a NUFFT;
snapping to the nearest grid point lets the project keep using ifft2 and
still reproduces the characteristic artifacts.

### (c) Random variable-density — variable_density_mask()

Random sampling with probability p(r) = α·(1 − r)^6, clipped at 1, with a
small central disc forced to p = 1. α is solved by bisection so the mean
probability equals the requested ratio.

*Artifact signature:* **incoherent, noise-like grain** instead of structured
ghosts. This is the whole point — randomness is what makes compressed sensing
possible (see §5b).

### (d) Centre-only — center_only_mask() — an ideal low-pass filter

### (e) Edges-only — edges_only_mask() — an ideal high-pass filter

These two are a controlled experiment, not a realistic acquisition: keep the
**same number of samples** both ways and change only *where* they are taken.
The centre-only image has correct contrast and shape with blurred detail and
visible **Gibbs ringing** around sharp boundaries (from truncating the Fourier
series at the hard rim of the disc). The edges-only image is nearly black —
dropping the centre drops the DC term, i.e. the average brightness of the
entire image — and when contrast-stretched it is recognisably an edge map.

---

## 5. Stage 2 features

### (a) Scanner noise in k-space — noise.py

Noise is added to the **measured samples**, not to the finished image, because
that is where it physically enters. Consequences that image-domain noise
cannot reproduce:

- The noise is **complex** (the receiver has independent I and Q channels).
- It is **white in k-space** — the same absolute noise swamps the tiny outer
  samples while barely touching the enormous DC sample, so *fine detail
  degrades long before contrast does*.
- Taking the magnitude turns zero-mean complex noise into strictly positive
  **Rician** noise — which is why the background of a real MRI image is a
  faint grey haze rather than true black. This falls out of the model for free.

Noise is applied **only where the mask actually measures**, so a 25% mask
admits ~25% of the noise energy (the SNR ∝ √N rule of real MRI). Order
matters: mask first, then noise. Noising first and masking after would model a
scanner that acquires all of k-space, corrupts it, then throws most of it away
— which would wrongly make the noise level independent of acceleration.

SNR is specified in dB relative to mean k-space power:
σ = sqrt(mean|K|² / (2·10^(SNR/10))). Roughly: 40 dB = clean clinical scan,
20 dB = visibly grainy, 10 dB = bad. measured_snr_db() reads the SNR back off
the result to verify the noise level is what was requested.

### (b) Compressed sensing — cs.py

Zero-filling assumes every unmeasured k-space point is zero. It is not zero —
we just did not look, and that wrong guess *is* the artifact. Compressed
sensing instead asks: **of all images consistent with what we measured, which
is the simplest?**

    minimise  ‖ M·F·x − y ‖²  +  λ·‖ W·x ‖₁

Solved by ISTA with FISTA (Nesterov) momentum, alternating two steps per
iteration:

1. **Data consistency** — fold the true measurements back into the current
   estimate. F is unitary, so the gradient step size is exactly 1.
2. **Sparsity** — soft-threshold the wavelet coefficients
   (soft(v,t) = sign(v)·max(|v|−t, 0), the exact proximal operator of the
   L1 norm, which is what makes ISTA provably convergent).

Transform: Daubechies-4 wavelets, 3 levels, periodization mode.

Three implementation decisions worth defending in a viva:

- **The approximation band is exempt from thresholding.** It is a dense,
  high-energy summary of the anatomy holding the largest coefficients in the
  transform; shrinking it every iteration drains energy, contrast drops, and
  the result ends up *worse* than zero-filling. Only the detail bands are
  sparse, so only they are thresholded.
- **Phase is preserved inside the loop.** The internal inverse FFT keeps the
  complex estimate; FFT(|x|) ≠ FFT(x), so taking the magnitude inside the
  loop would make data consistency fight itself. The magnitude is taken
  exactly once, at the very end.
- **CS requires the random mask.** Cartesian ghosts are just as sparse as real
  anatomy, so no sparsity prior can tell them apart. Incoherent grain, it can.

### (c) Patient motion — motion.py

Built entirely on the **Fourier shift theorem**:
f(x − a) ↔ F(k)·exp(−j2πka/N). Translating the object leaves the k-space
*magnitude* untouched and only stamps a linear phase ramp — so a uniform shift
is completely harmless and reconstructs to exactly the shifted image.

The artifact appears when the shift is **not uniform across the scan**. A
Cartesian scan fills one row per TR, so the row index *is* the acquisition
time. If the patient moves between rows, each row carries a different phase
ramp and the data is no longer the transform of any single consistent object.

Three motion models, each producing one displacement per row:

| Model | Physical analogue | Artifact |
|---|---|---|
| sudden_jerk | Patient flinches once mid-scan | A **discrete ghost** — rows before and after are each internally consistent, describing two sharp objects |
| slow_drift | Sinking into the table | **Blur** — every row disagrees slightly with its neighbours, so inconsistency spreads |
| periodic | Breathing | A **regular train of ghosts** — a sinusoid is a pair of complex exponentials, so to first order this convolves the image with two deltas |

A radial-aware variant (draw_spokes_indexed + apply_motion_radial) times
motion **per spoke** rather than per row, because timing radial data by row is
physically wrong: the centre should be an incoherent average over every
spoke's position — which is exactly radial's motion-robustness advantage — and
per-row timing would destroy that. Its centre tie-break assigns contested
pixels to whichever spoke's angle is closest to the pixel's own polar angle,
avoiding a directional bias that would re-concentrate corruption onto a few
acquisition times.

*Status note:* motion.py is a complete, tested library module used for
analysis and its own self-test; it is not wired into a tab of the Streamlit
app.

---

## 6. Results (all verified by running the code)

### 6a. Stage 1 sweep — Shepp-Logan phantom, 256×256, no noise

Source: outputs/metrics.csv, produced by `python main.py`.

| Strategy | 100% (1×) | 50% (2×) | 25% (4×) | 12.5% (8×) |
|---|---|---|---|---|
| **Cartesian** PSNR | 321.7 dB | 26.81 dB | 20.54 dB | 17.43 dB |
| **Cartesian** SSIM | 1.0000 | 0.6832 | 0.5491 | 0.5348 |
| **Radial** PSNR | 321.7 dB | 29.64 dB | 23.29 dB | 19.65 dB |
| **Radial** SSIM | 1.0000 | 0.4760 | 0.3115 | 0.2763 |
| **Variable-density** PSNR | 321.7 dB | 31.58 dB | 28.86 dB | 25.81 dB |
| **Variable-density** SSIM | 1.0000 | 0.8227 | 0.7223 | 0.4777 |

Readings from this table:

1. **321.7 dB at 100% is the correctness proof**, not a result — it is numerical
   round-trip error at float64 precision. It confirms the forward/inverse FFT
   and the fftshift/ifftshift pairing are exactly right.
2. **Variable-density wins on both metrics at every ratio.** Random sampling
   beats structured sampling even before compressed sensing is applied.
3. **PSNR and SSIM disagree about radial vs Cartesian** — radial scores ~3 dB
   *higher* PSNR but much *lower* SSIM. This is a genuinely valuable result,
   not an inconsistency: PSNR is a per-pixel error measure and is blind to
   structure, so it cannot distinguish diffuse streaks from a coherent ghost
   of the same energy. SSIM compares local means, variances and covariance, so
   it responds to lost structure and tracks human judgement better. Radial
   spreads its error thinly (good PSNR) while destroying local structure
   everywhere (bad SSIM). **Quoting one metric alone would have hidden this.**

### 6b. Compressed sensing vs zero-filling — sample brain-glioma-778, random mask

| Acceleration | Zero-fill | Compressed sensing (FISTA, λ=0.01, 80 iters) | Gain |
|---|---|---|---|
| 8× (12.5% sampled) | 30.37 dB / 0.8069 SSIM | 31.54 dB / 0.9009 SSIM | **+1.17 dB, +0.094 SSIM** |
| 16× (6.25% sampled) | 26.10 dB / 0.6425 SSIM | 28.10 dB / 0.7609 SSIM | **+2.00 dB, +0.118 SSIM** |

From **exactly the same measurements** — the only thing that changed is the
answer to "what was not measured?". The gain grows as undersampling gets more
aggressive, which is the expected and correct behaviour.

**The project also reports where CS loses.** At mild undersampling (above
~15% sampling) zero-filling is already close to perfect, so the sparsity prior
costs more PSNR than it gains; the app says so explicitly in that regime
rather than hiding it. Reporting the failure case is a strength, not a defect.

### 6c. Motion cost — sample brain-glioma-778, fully sampled k-space

Even with 100% of k-space acquired, motion alone degrades the image:

| Motion | PSNR | SSIM |
|---|---|---|
| 5 px sudden jerk at mid-scan | 22.75 dB | 0.7932 |
| 5 px linear drift over the scan | 21.07 dB | 0.7535 |
| 5 px periodic, 4 cycles (breathing) | 21.37 dB | 0.6464 |

For scale: a 5-pixel jerk on a *fully sampled* scan costs more quality than
8× undersampling with a variable-density mask. Motion is not a small effect.

### 6d. Correctness checks built into the project

| Check | Expected | Actual |
|---|---|---|
| FFT round trip, phantom float64 | ~0 | max error 3.6e-7 (asserted < 1e-9, or < 1e-5 for complex64 store data) |
| FFT round trip, store sample complex64 | limited by float32 | 4.12e-8 |
| Fourier shift theorem: uniform shift must equal np.roll | machine epsilon | **6.66e-16** |
| Requested vs measured k-space SNR | equal | verified live in the app |
| Reconstruction speed | fast enough for live sliders | **1.7 ms** per inverse FFT |
| CS run time | interactive | ~1.5 s for 80 FISTA iterations at 256×256 |

These are runnable live during a presentation, which is the strongest possible
answer to "how do you know your FFT is right?"

---

## 7. The interactive app, tab by tab

| Tab | What it shows |
|---|---|
| **1. Acquire** | The full pipeline in five panels: ground truth → full k-space (log) → mask → what the scanner got → zero-filled reconstruction, plus a PSNR/SSIM readout, an absolute-error map, and a strategy-specific explanation of *why* this artifact looks like this |
| **2. Centre vs edges** | Side-by-side low-pass vs high-pass with the **same sample budget**; reports mean brightness for each (the centre-only image matches the original's mean; the edges-only image is near zero) and states what fraction of this sample's energy is inside the central 10% radius |
| **3. Noise** | 2×2 comparison: clean/full, noisy/full, undersampled/clean, undersampled+noisy — isolating noise from undersampling as independent effects; includes a live verification that the measured SNR equals the requested SNR |
| **4. Compressed sensing** | Zero-fill vs FISTA on identical measurements, with sliders for ratio, λ and iteration count, and a per-iteration PSNR convergence curve plotted against the zero-fill baseline |
| **5. Sweep** | PSNR and SSIM vs sampling % for all three strategies at 100/50/25/12.5/6.25%, as charts plus a downloadable CSV table |
| **6. About this sample** | Provenance: exact source file, collection, acquisition metadata, expert tumour mask overlay where available, derived statistics, and a plain statement of what is simulated and what is real |

---

## 8. Mapping to Signals and Systems course concepts

| Concept | Where it appears |
|---|---|
| 2-D Discrete Fourier Transform | The entire forward/inverse pipeline |
| FFT / IFFT and fftshift conventions | kspace.py; the 321 dB round trip is the proof it is right |
| Frequency-domain interpretation of images | Centre = low frequency = contrast; edges = high frequency = detail |
| Sampling theorem and aliasing | Undersampling below Nyquist → ghosting; this is Nyquist violated on purpose |
| Impulse-train sampling ↔ convolution | Why regular Cartesian skipping produces evenly-spaced coherent ghosts |
| Ideal low-pass / high-pass filtering | Centre-only and edges-only masks |
| Gibbs phenomenon | Ringing at the hard rim of the centre-only disc |
| Fourier shift theorem | The complete basis of the motion module |
| DC component | Removing the centre removes mean image brightness |
| Linearity and unitarity of the DFT (Parseval) | Why the CS gradient step size is exactly 1 |
| Complex signals, I/Q channels | Complex Gaussian receiver noise; Rician magnitude statistics |
| SNR in dB, white noise | noise.py |
| Sparsity, L1 minimisation, proximal operators | cs.py — the same principle as JPEG-2000 compression, run backwards |
| Hermitian symmetry of real-signal transforms | Why a synthetic phase map is applied; the hermitian_asymmetry statistic |
| Quality metrics: PSNR vs SSIM | metrics.py, and the radial/Cartesian disagreement in §6a |

---

## 9. Limitations, stated openly

1. **The k-space is simulated.** A real scanner measures k-space directly;
   this project starts from reconstructed images and runs a forward FFT.
   Everything downstream of that point behaves identically, but the
   acquisition itself is not modelled at the level of RF pulses and gradients.
2. **The phase is synthetic.** Source files are magnitude images, so the
   original scanner phase no longer exists and a plausible smooth phase map is
   substituted.
3. **Radial sampling is rasterised.** Real radial trajectories are off-grid
   and need a NUFFT; nearest-grid-point snapping is an approximation that
   reproduces the artifacts but not the exact sample positions.
4. **Single-coil only.** Real accelerated MRI also uses parallel imaging
   (SENSE/GRAPPA) with multi-coil sensitivity maps; that is not modelled.
5. **The NINS samples are JPEG-compressed at source**, so some high-frequency
   content was lost before this pipeline ever saw them.
6. **Compressed sensing here is a basic ISTA/FISTA with a wavelet prior** —
   not a state-of-the-art learned reconstruction.

Every one of these is documented in the code and in the store's own README,
and every sample records the exact raw file it came from.

---

## 10. Headline numbers for slides

| Quantity | Value |
|---|---|
| Raw dataset processed | 12 GB, 43,622 files, 4 file formats |
| Curated sample store | 40 samples, 256×256, 43.5 MB |
| Energy inside central 10% radius | **92.6%** mean (73.6–98.9%) |
| k-space dynamic range | 62–85 dB (mean 73 dB) |
| Sampling strategies implemented | 5 |
| FFT round-trip PSNR at 100% sampling | 321.7 dB (numerically exact) |
| Shift-theorem verification error | 6.66e-16 (machine precision) |
| Reconstruction speed | 1.7 ms — fast enough for live sliders |
| Best variable-density result at 8× | 25.81 dB / 0.478 SSIM (phantom) |
| CS gain at 16× acceleration | **+2.00 dB, +0.118 SSIM** over zero-fill |
| Motion cost, 5 px jerk, fully sampled | 22.75 dB / 0.793 SSIM |
| Code size | ~3,900 lines across mri_sim/, kspace_store/, app/ |

---

## 11. How to run

    # Interactive demo (from the project root)
    streamlit run app/streamlit_app.py          # -> http://localhost:8501

    # Batch pipeline: full sweep, figures + CSV into outputs/
    python main.py
    python main.py --image brain.png            # bring your own grayscale image
    python main.py --size 512                   # work at 512x512
    python main.py --ratios 1.0 0.5 0.25        # choose sampling ratios
    python main.py --sample brain-glioma-778    # use a real clinical sample
    python main.py --noise-snr 20               # Stage 2: add scanner noise
    python main.py --center-edges               # Stage 2: low-pass vs high-pass figure
    python main.py --cs --cs-iters 80           # Stage 2: compressed sensing comparison

    # Rebuild the k-space store from the raw dataset (build-time deps needed)
    python -m kspace_store.build

    # Self-test of the motion module
    python -m mri_sim.motion                    # prints the shift-theorem error

---

## 12. Figures already generated (available to embed in slides)

In outputs/:

- cartesian_ratio1000/0500/0250/0125.png — 4-panel comparisons (original |
  sampled k-space | reconstruction | difference map) at 100/50/25/12.5%
- radial_ratio*.png, variable_density_ratio*.png — same, per strategy
- mask_gallery.png — all three masks side by side at the most aggressive ratio
- summary_metrics.png — PSNR and SSIM vs sampling ratio, one line per strategy
- metrics.csv — the full numeric table
- kspace_store_gallery.png — the 40-sample store at a glance
- outputs/stage2_demo/ — includes center_vs_edges.png and
  compressed_sensing.png
- outputs/store_demo/ — the same sweep run on the real clinical sample
  brain-glioma-778
- data/kspace_store/previews/ — image + log-k-space PNG pair for all 40 samples

---

## 13. Suggested slide structure

A 14–16 slide deck, tuned for a course presentation with a live demo:

1. **Title** — MRI k-Space Reconstruction Simulator, CSE 220
2. **The hook** — An MRI scanner never photographs anything. It measures the
   2-D Fourier transform. Show one image next to its log-k-space.
3. **Why this matters** — MRI is slow; scan time ∝ number of k-space samples;
   patients must hold still, children are sedated. Faster scans = less data.
4. **The core question** — If you skip most of k-space, how bad is the image,
   and does *which* part you skip matter?
5. **The pipeline** — image → FFT → k-space → mask → IFFT → reconstruction, as
   a diagram. Note: the IFFT assumes unmeasured = zero, and that assumption is
   the artifact.
6. **What k-space holds** — the 92.6% energy statistic and the 73 dB dynamic
   range; centre = contrast, edges = detail. Use center_vs_edges.png.
7. **Three sampling strategies** — mask_gallery.png, with the one-line
   artifact signature for each (coherent ghosts / streaks / incoherent grain).
8. **Results, visually** — the 4-panel figure at 12.5% for each strategy.
9. **Results, numerically** — the §6a table + summary_metrics.png.
10. **The PSNR/SSIM disagreement** — radial wins on PSNR, loses on SSIM; what
    each metric is actually measuring. (This is the best "we understood our
    own results" slide in the deck.)
11. **Scanner noise** — noise lives in k-space, is complex and white, becomes
    Rician after the magnitude; the 2×2 comparison.
12. **Compressed sensing** — the minimisation problem, the two alternating
    steps, and the §6b table showing +2.00 dB at 16×. Emphasise: same
    measurements, better answer to "what was not measured?"
13. **Patient motion** — the Fourier shift theorem, and the three artifact
    types; the §6c table.
14. **The dataset** — 12 GB → 40 curated samples with full provenance; the
    store gallery.
15. **Correctness** — the §6d verification table (321.7 dB round trip, 6.7e-16
    shift-theorem error). All runnable live.
16. **Limitations & closing** — §9 stated plainly, then what was learned.

Live demo, if time allows: drag the ratio slider in tab 1 from 100% to 5% and
watch the ghosts appear; switch strategies at a fixed ratio to show the three
artifact signatures; then tab 4, push to 16×, and show CS recovering detail
the zero-fill lost.
