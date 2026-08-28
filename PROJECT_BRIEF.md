# MRI k-Space Reconstruction Simulator — Project Brief

*A plain-language explanation of what this project is, why it matters, and how to
present it. Written for you, not for the instructor — the technical
documentation is in [README.md](README.md).*

---

## 1. The one-sentence version

> An MRI scanner does not measure an image — it measures the image's 2-D Fourier
> transform, one sample at a time, and scan time is proportional to how many
> samples it collects. This project simulates that process on real clinical MRI
> data, and shows exactly what breaks when you speed the scan up by measuring
> less.

If your supervisor only remembers one thing, make it that. Everything else in
the project is a consequence of it.

---

## 2. What is actually going on (the physics, briefly)

A patient in an MRI scanner sits in a strong magnetic field. Radio pulses knock
hydrogen nuclei out of alignment, and as they relax they emit a radio signal
that a coil picks up as a voltage over time.

Here is the part that makes this a *signals* project rather than a physics one.
While that signal is being received, the scanner applies **magnetic field
gradients** — deliberately making the field strength vary across space. Because
precession frequency depends on field strength, position gets encoded into
frequency and phase. The consequence is remarkable:

**Each digitised sample of the received voltage corresponds directly to one
point in the 2-D Fourier transform of the slice being imaged.**

That Fourier domain is called **k-space**. The scanner fills it up sample by
sample; the image is recovered with a 2-D inverse FFT. No transformation is
"applied" to the data to get there — the Fourier relationship falls out of the
physics of gradient encoding.

Two facts follow immediately, and the whole project lives in the gap between
them:

1. **Scan time ∝ number of k-space samples acquired.** A patient lying still in
   a loud tube for 40 minutes is a real clinical problem — for children, for
   trauma cases, for anyone who cannot hold still.
2. **The inverse FFT needs *all* of k-space** to reconstruct the image exactly.

So: measure less, finish sooner, and get a worse image. **How much worse, and in
what way, depends entirely on *which* samples you skip.** That question is the
project.

### Why we start from an image instead of raw scanner data

Real scanner k-space (`.dat` / ISMRMRD files) involves multi-coil arrays, coil
sensitivity maps, ramp sampling and vendor-specific corrections — a semester of
work before you can produce a single picture. So we invert the pipeline:

```
real image  ──FFT──▶  synthetic k-space  ──mask──▶  undersampled k-space  ──iFFT──▶  reconstruction
             (we fake this step)          (this is the experiment)
```

The forward FFT manufactures k-space that a scanner *would* have measured.
**Everything downstream of that point is identical to the real pipeline** — the
masks, the zero-filling, the inverse FFT, the artifacts, the metrics. And we get
something a real scan never gives you: perfect ground truth to measure error
against.

Be upfront about this. It is a legitimate, standard simulation methodology, and
saying so plainly reads as understanding, not as apology.

---

## 3. A second axis of corruption: patient motion

Everything above assumes the patient held still. They often can't — children
especially, which is the actual reason paediatric scans have to be fast. This
is a second, independent way a scan can go wrong, and it is worth pulling
apart from undersampling because the physics is different and, per point
below, it finally gives radial sampling a real justification.

**The physics, in one line — the Fourier shift theorem:**

```
f(x − a)   ⟷   F(k) · exp(−j·2π·k·a/N)
```

Moving the patient does **not** change k-space *magnitude* at all — it only
stamps a linear **phase ramp** onto it. Cartesian k-space is filled one row
per repetition of the pulse sequence, so row index *is* acquisition time. If
the patient is in a different place for different rows, each row carries a
different phase ramp, and the reconstruction is the inverse FFT of data that
is no longer consistent with any single object. That inconsistency is what
shows up as ghosting, blur, or streaking, depending on *how* the motion
evolves over the scan.

Three motion profiles, measured fully sampled (no undersampling in the mix,
so this is motion's cost in isolation) on `brain-glioma-778`:

| Motion | PSNR | SSIM | Artifact |
|---|---|---|---|
| none | (exact) | 1.000 | — |
| sudden jerk, 5 px at mid-scan | 22.75 dB | 0.793 | discrete ghost copy |
| slow drift, 0 → 6 px | 20.05 dB | 0.703 | smearing / blur |
| breathing, 3 px, 8 cycles | 24.36 dB | 0.738 | regular ghost train |

A 5-pixel movement costs on the order of 100+ dB — motion is a far bigger
threat to image quality than any undersampling ratio in this project.

**Why this should finally make radial sampling look good, and the honest
result:** every radial spoke passes through the k-space centre, so in theory
no single spoke's mistiming can corrupt the centre the way one bad Cartesian
row can — radial *should* be the motion-robust choice. Getting that to show
up requires timing motion **per spoke**, not per row (naively reusing
row-based timing and just masking with the radial pattern washes out almost
all of the difference between strategies). We built the per-spoke machinery
(`draw_spokes_indexed` + `apply_motion_radial` in `mri_sim/motion.py`) and
verified it is mathematically exact — a uniform shift applied per spoke
reconstructs identically to shifting the image first, to machine precision.

But with the straightforward implementation — spokes timed in simple
sequential angle order — radial did **not** come out ahead of Cartesian under
identical motion on our reference sample, at several ratios and with every
motion model we tried. Our working theory: real motion-robust radial
acquisition relies on *interleaving* spoke order away from angle order (e.g.
golden-angle sampling), so that any short time window of motion samples
widely-separated angles instead of one contiguous wedge; a plain sequential
sweep doesn't get that property for free. This is flagged as an open
question, not asserted either way — see [§9](#9-be-honest-about-these).

**Status:** implemented as a self-contained module, `mri_sim/motion.py` —
not yet wired into the CLI sweep or the Streamlit app, so it is a verified
library result rather than a demoable feature today.

---

## 4. What we built

Three layers, each usable on its own.

### Layer 1 — the simulator (`mri_sim/`)

The core signal processing. Forward FFT, five sampling masks, zero-filled
reconstruction, PSNR/SSIM scoring, publication-quality figures. Plus the Stage 2
additions: k-space noise simulation and a compressed-sensing reconstruction.
Plus, most recently, a patient-motion simulator (§3) — currently a verified
module, not yet surfaced in the CLI or the app.

### Layer 2 — the k-space sample store (`kspace_store/`, `data/kspace_store/`)

We were given a **12 GB, 43,622-file** clinical dataset. Most projects would
load one JPEG from it and move on. Instead we built a curated library: 40
carefully-chosen slices converted into ready-to-use k-space, each with full
metadata and provenance back to the exact source file.

| Source | n | What it contributes |
|---|---|---|
| Brain tumour MATLAB files | 12 | Meningioma / glioma / pituitary, each with an **expert-drawn tumour mask** |
| Clinical pathology JPEGs | 14 | Normal, glioma, infarct, haemorrhage, hydrocephalus, atrophy, Chiari I, abscess… |
| **Real DICOM spine studies** | 8 | The *same anatomy* under T1 and T2, with the scanner's own TE/TR/field-strength readings |
| 3-D volumes (NIfTI) | 6 | Knee MRI, abdominal MRI, jaw CT, walnut micro-CT, synthetic control |

Total: 42 MB, loads in milliseconds, needs only NumPy to read.

### Layer 3 — the interactive web app (`app/streamlit_app.py`)

Six tabs, driven by sliders, running at ~1 ms per reconstruction so it updates
live as you drag. This is what you demo.

---

## 5. The signals & systems content

This matters most for grading. The project is not "an MRI thing" — it is a
direct application of the course, and you should name the concepts explicitly.

| Concept | Where it appears |
|---|---|
| **2-D DFT / FFT** | The entire forward and inverse pipeline |
| **Sampling & Nyquist** | Undersampling k-space below Nyquist is precisely what causes the artifacts |
| **Aliasing** | Cartesian undersampling folds the image onto itself as ghost copies, spaced by FOV / acceleration factor |
| **Low-pass / high-pass filtering** | Centre-only vs edges-only masking — an ideal circular filter of each type |
| **Gibbs phenomenon** | The ringing around sharp edges when k-space is truncated at a hard rim — the 2-D version of truncating a Fourier series |
| **Convolution theorem** | Multiplying k-space by a mask = convolving the image with the mask's PSF. This *is* the artifact: each mask's artifact pattern is literally the Fourier transform of that mask |
| **Spectral energy distribution** | 92.6% of k-space energy sits in the central 10% radius — measured across all 40 samples |
| **fftshift conventions** | Centered k-space throughout; `ifftshift` (not `fftshift`) to undo it, which differs for odd sizes |
| **Additive white Gaussian noise** | Complex AWGN in k-space, with the correct I/Q channel model |
| **Sparsity & L1 minimisation** | Compressed sensing via iterative soft-thresholding (FISTA) |
| **Fourier shift theorem** | Patient motion during acquisition — a spatial shift is a phase ramp in k-space, never a magnitude change |

The convolution-theorem line is the strongest single point in that table. If
asked "why does Cartesian give ghosts and radial give streaks?", the answer is
one sentence: *because masking in k-space is convolution in image space, and the
artifact is the point-spread function — the Fourier transform of the mask
itself. A comb of lines transforms to a comb of shifted copies; spokes transform
to streaks.*

---

## 6. The five-minute live demo

Do it in this order. Each step sets up the next.

**① Open tab 1, pick a brain tumour case, leave sampling at 100%.**
Point out the five panels: truth → k-space → mask → acquired → reconstruction.
Say: *"The scanner measures the middle panel, not the first one."* Note that
k-space is displayed on a log scale because its dynamic range is ~72 dB — on a
linear scale it is one bright dot.

**② Drag the ratio down to 25%, strategy = Cartesian.**
Ghosts appear. *"We just made the scan 4× faster. Those ghost copies are
aliasing — skipping every 4th line in k-space folds the image onto itself."*

**③ Switch strategy to Radial, then to Random variable-density. Same 25%.**
The artifact *changes character* — streaks, then noise-like grain — while the
sample count stays identical. PSNR climbs 24.1 → 27.5 → 35.8 dB. *"Same number
of measurements, same scan time. All that changed is which points we chose."*
This is the moment the project stops looking like a homework exercise.

**④ Jump to tab 2 (Centre vs edges), keep 10% in both.**
Centre-only: recognisable, correctly bright, blurred. Edges-only: **a black
rectangle**. Let that sit for a second, then hit the contrast-stretched panel —
it is an edge map. *"Same 10% budget. The centre carries contrast and shape; the
edges carry only boundaries. Discarding the centre discards the DC term, which
is the average brightness of the whole image."*

**⑤ Tab 4 (Compressed sensing), ratio 0.0625.**
Zero-fill vs FISTA on identical measurements: +2.06 dB, +0.124 SSIM. *"Same
data. The only difference is what we assume about the samples we never measured
— zero, versus whatever makes the image sparsest while still matching what we
did measure."*

**⑥ Finish on the About tab.**
Show the source file path, the scanner's real TE/TR, the tumour mask overlay.
*"Every sample traces back to a specific file in the clinical dataset."*

---

## 7. What makes this stand out

Three things, in order of how much they'll actually count.

### It uses real clinical data properly, with provenance

Not one demo image — 40 curated slices across four different file formats
(HDF5/MATLAB, JPEG, DICOM, NIfTI), each recording the exact source file it came
from. The DICOM spine samples carry the scanner's own acquisition parameters, so
you can put T1 (TE 9.2 ms, TR 620 ms) next to T2 (TE 94 ms, TR 3370 ms) of **the
same patient's spine** and show that MRI contrast is a property of the
acquisition, not of the anatomy. That is a genuinely instructive pairing, and it
came out of the dataset rather than being asserted from a textbook.

### It reports where the method *loses*

Compressed sensing is the sexy part, and the tempting move is to show only the
case where it wins. Ours reports all three:

| Sampling | Zero-fill | CS (FISTA) | Change |
|---|---|---|---|
| 25% (4×) | 35.84 dB / 0.933 | 34.65 / 0.939 | **−1.19 dB**, +0.006 |
| 12.5% (8×) | 30.37 / 0.807 | 31.54 / 0.901 | +1.17 dB, +0.094 |
| 6.25% (16×) | 26.10 / 0.643 | 28.16 / 0.767 | +2.06 dB, +0.124 |

And explains the loss: at 4× the variable-density mask already captures nearly
all the signal energy, so plain zero-filling is close to perfect and the
sparsity prior costs more than it gains. Being able to explain *why your method
loses* is a stronger demonstration of understanding than any winning number.

### It is a working instrument, not a static report

Sliders, live reconstruction, 40 subjects, downloadable metrics. Your supervisor
can grab the mouse and try to break it. Very few class projects survive that.

---

## 8. Questions you should be ready for

**"Is this real MRI data or simulated?"**
The *images* are real clinical MRI and CT. The *k-space* is simulated — we run a
forward FFT on the images to manufacture the measurements a scanner would have
made. Real raw scanner data involves multi-coil arrays and vendor corrections
that are out of scope; everything downstream of the forward FFT is identical to
the real pipeline, and this way we have exact ground truth to measure against.

**"Where does the phase come from, if your sources are magnitude images?"**
We add a synthetic smooth phase map — a B₀-like quadratic bowl, a gradient ramp,
and coil-sensitivity-like ripples — before the FFT. Without it the image would
be purely real, which makes k-space perfectly Hermitian-symmetric
(`K(−k) = conj(K(k))`), meaning half the data would be a free copy of the other
half and any partial-Fourier demonstration would be unrealistically perfect. The
manifest reports a Hermitian-asymmetry figure of 0.83–1.70 per sample, versus ~0
for a real-valued image, so the effect is measured rather than assumed.

**"Why does everything keep the centre of k-space?"**
Because that is where the signal is. Measured across all 40 samples: the central
10% radius — about 1% of the samples — holds a mean of **92.6%** of total
k-space energy (range 73.6–98.9%). Low frequencies encode contrast and gross
shape; the periphery encodes fine detail. Drop the centre and you lose the
image; drop the periphery and you lose sharpness.

**"Why is radial better than Cartesian on PSNR but worse on SSIM?"**
(At 25% on the glioma case: radial 27.5 dB / 0.651, Cartesian 24.1 dB / 0.722.)
The two metrics measure different things. PSNR is mean squared error — radial
oversamples the centre, so its total energy error is lower. SSIM measures local
structural similarity, and radial's streaks smear across tissue boundaries,
which SSIM punishes hard. This is a good illustration of why one number is never
enough to judge an image reconstruction.

**"Does patient motion matter more than undersampling?"**
Yes, by a wide margin. A 5-pixel jerk mid-scan costs on the order of 100+ dB,
fully sampled — motion is a phase corruption (Fourier shift theorem), and it
does not care whether every k-space point was measured. This is why fast
imaging protocols exist for children even when image quality could otherwise
be near-perfect: the real threat isn't Nyquist, it's the patient moving.

**"What is compressed sensing actually doing?"**
Zero-filling answers "what was the unmeasured data?" with "zero", which is
false. CS instead asks: of all images consistent with what we measured, which is
sparsest in a wavelet basis? It alternates two steps — enforce the measured
samples, then shrink small wavelet coefficients to zero — and needs *random*
sampling, because Cartesian ghosts are just as sparse as real anatomy, so no
sparsity prior can separate them.

**"How do you know your FFT round trip is correct?"**
Every sample is verified on build: reconstructing untouched k-space reproduces
the stored image to a maximum absolute error of 3.6 × 10⁻⁷ across all 40, which
is float32 storage precision. The noise simulator is self-checking too — request
20 dB SNR and it measures back 19.99 dB.

**"What would you do next?"**
Multi-coil parallel imaging (SENSE/GRAPPA), which is how real scanners actually
accelerate; total-variation regularisation alongside wavelets; and validating
against genuine raw scanner data from the fastMRI dataset.

---

## 9. Be honest about these

Volunteering limitations before you're asked is worth more than hoping they go
unnoticed. The first four are documented in the README; the fifth is a fresh
finding from building the motion module (§3):

- **The k-space is simulated**, via forward FFT from reconstructed images.
- **The phase map is synthetic** — the source files are magnitude images.
- **The pathology JPEGs are lossy-compressed at source**, so some high-frequency
  content was gone before we saw it. (Worth noting that JPEG discards high
  frequencies for much the same reason undersampling does.)
- **Slice selection is automated, not clinical.** The pathology folders mix
  imaging planes with no labels, so slices are chosen by a rule on the pixels:
  left–right mirror correlation ≥ 0.90 identifies axial views (the head is
  nearly symmetric), then pick the one showing the most brain. Four diagnoses
  have no axial candidate and fall back to sagittal — which happens to be the
  conventional view for those anyway.
- **Radial's motion advantage didn't show up under simple spoke ordering.**
  We expected — and the physics argument in §3 predicts — that timing motion
  per spoke would let radial beat Cartesian under identical motion. With
  spokes ordered by simple sequential angle, it didn't, on every ratio and
  motion model we tried. We think this is because real motion-robust radial
  acquisition depends on interleaving spoke order (golden-angle sampling),
  not just per-spoke timing on its own — but that's a hypothesis, not a
  verified result. Say this plainly if asked rather than claiming a win the
  numbers don't support.

---

## 10. Numbers worth memorising

| Quantity | Value |
|---|---|
| Raw dataset | 12 GB, 43,622 files |
| Sample store | 40 samples, 256×256, 42 MB |
| Energy in central 10% radius | **92.6%** mean (73.6–98.9%) |
| k-space dynamic range | 60–85 dB (why log display is mandatory) |
| FFT round-trip error | 3.6 × 10⁻⁷ |
| Reconstruction speed | ~1 ms (so sliders are live) |
| Best CS gain | +2.06 dB, +0.124 SSIM at 16× acceleration |
| Sampling strategies | 5 (Cartesian, radial, variable-density, centre-only, edges-only) |
| Motion cost, fully sampled | 5 px jerk: 22.75 dB / 0.793 SSIM (vs. exact with no motion) |
| Uniform-shift verification error | 6.7 × 10⁻¹⁶ (machine precision) |

---

## 11. Running it

```bash
streamlit run app/streamlit_app.py     # then http://localhost:8501
```

From the project root. Full instructions, CLI flags and technical detail are in
[README.md](README.md) sections 7 and 8.
