# MRI k-Space Reconstruction Simulator: Project Brief

Plain-language notes on what this project does and how to present it. For how
to run it, see [README.md](README.md).

---

## 1. The one-sentence version

> An MRI scanner does not measure an image. It measures the image's 2-D Fourier
> transform, one sample at a time, and scan time is proportional to how many
> samples it collects. This project simulates that process on real clinical MRI
> data and shows what goes wrong when the scan is sped up by measuring less.

Everything else in the project follows from this.

---

## 2. What is going on (the physics, briefly)

A patient in an MRI scanner sits in a strong magnetic field. Radio pulses knock
hydrogen nuclei out of alignment, and as they relax they emit a radio signal
that a coil picks up as a voltage over time.

This is where signals and systems comes in. While that signal is being
received, the scanner applies **magnetic field gradients**, which make the field
strength vary across space. Because precession frequency depends on field
strength, position gets encoded into frequency and phase. As a result:

**Each digitised sample of the received voltage corresponds to one point in the
2-D Fourier transform of the slice being imaged.**

That Fourier domain is called **k-space**. The scanner fills it sample by
sample, and the image is recovered with a 2-D inverse FFT. No transform is
applied to the data to get there; the Fourier relationship comes from the
physics of gradient encoding.

Two facts follow, and the project is about the conflict between them:

1. **Scan time is proportional to the number of k-space samples acquired.**
   Lying still in the scanner for 40 minutes is a real clinical problem for
   children, trauma cases, and anyone who cannot hold still.
2. **The inverse FFT needs *all* of k-space** to reconstruct the image exactly.

So measuring less finishes sooner and gives a worse image. **How much worse, and
in what way, depends on *which* samples are skipped.** That is the question the
project investigates.

### Why we start from an image instead of raw scanner data

Real scanner k-space (`.dat` / ISMRMRD files) involves multi-coil arrays, coil
sensitivity maps, ramp sampling and vendor-specific corrections, which is far
more work than this project allows. So the pipeline starts from the image:

```
real image --FFT--> synthetic k-space --mask--> undersampled k-space --iFFT--> reconstruction
           (simulated step)           (the experiment)
```

The forward FFT produces the k-space a scanner *would* have measured.
**Everything after that point is the same as the real pipeline**: the masks, the
zero-filling, the inverse FFT, the artifacts, and the metrics. It also gives
something a real scan cannot: an exact ground truth to measure error against.

This is a standard way to simulate MRI reconstruction, and it should be stated
clearly in the presentation.

---

## 3. A second source of corruption: patient motion

Everything above assumes the patient held still. Often they can't, especially
children, which is one reason paediatric scans need to be fast. Motion is a
separate way a scan can go wrong. It is treated separately from undersampling
because the physics is different, and it is the case where radial sampling is
expected to help.

**The physics in one line, from the Fourier shift theorem:**

```
f(x - a)   <-->   F(k) * exp(-j*2*pi*k*a/N)
```

Moving the patient does **not** change the k-space *magnitude*. It only adds a
linear **phase ramp**. Cartesian k-space is filled one row per repetition of the
pulse sequence, so the row index is the acquisition time. If the patient is in a
different place for different rows, each row carries a different phase ramp,
and the reconstruction is the inverse FFT of data that no longer matches any
single object. That mismatch shows up as ghosting, blur, or streaking, depending
on *how* the motion changes over the scan.

Three motion profiles, fully sampled (no undersampling, so this is the cost of
motion alone), on `brain-glioma-778`:

| Motion | PSNR | SSIM | Artifact |
|---|---|---|---|
| none | (exact) | 1.000 | none |
| sudden jerk, 5 px at mid-scan | 22.75 dB | 0.793 | single ghost copy |
| slow drift, 0 to 6 px | 20.05 dB | 0.703 | smearing / blur |
| breathing, 3 px, 8 cycles | 24.36 dB | 0.738 | regular ghost train |

A 5-pixel movement costs on the order of 100+ dB, so motion is a far bigger
threat to image quality than any undersampling ratio in this project.

**Radial sampling and motion.** Every radial spoke passes through the k-space
centre, so in theory a single mistimed spoke cannot corrupt the centre the way
one bad Cartesian row can. That would make radial the more motion-tolerant
choice. Showing this requires timing motion **per spoke**, not per row (reusing
row-based timing and only masking with the radial pattern removes almost all of
the difference between strategies). The per-spoke version
(`draw_spokes_indexed` + `apply_motion_radial` in `mri_sim/motion.py`) is
implemented and verified: a uniform shift applied per spoke reconstructs
identically to shifting the image first, to machine precision.

However, with spokes timed in simple sequential angle order, radial did **not**
come out ahead of Cartesian under identical motion on the reference sample, at
several ratios and with every motion model tried. A likely explanation is that
motion-tolerant radial scans on real scanners *interleave* the spoke order (for
example golden-angle sampling), so that any short window of motion is spread
over widely separated angles instead of one contiguous wedge. A plain sequential
sweep does not have that property. This is still an open question; see
[section 9](#9-limitations).

**Status:** implemented as a self-contained module, `mri_sim/motion.py`. It is
not yet connected to the CLI sweep or the Streamlit app, so it is a verified
library result rather than something shown in the demo.

---

## 4. What we built

Three layers, each usable on its own.

### Layer 1: the simulator (`mri_sim/`)

The core signal processing: forward FFT, five sampling masks, zero-filled
reconstruction, PSNR/SSIM scoring, and saved figures. The Stage 2 additions are
k-space noise simulation and a compressed-sensing reconstruction. The patient
motion simulator (section 3) is also here, but is not yet used by the CLI or the
app.

### Layer 2: the k-space sample store (`kspace_store/`, `data/kspace_store/`)

The raw dataset is **12 GB, 43,622 files**. From it we built a library of 40
selected slices converted into k-space, each with metadata and the path of the
exact source file.

| Source | n | What it contributes |
|---|---|---|
| Brain tumour MATLAB files | 12 | Meningioma / glioma / pituitary, each with an **expert-drawn tumour mask** |
| Clinical pathology JPEGs | 14 | Normal, glioma, infarct, haemorrhage, hydrocephalus, atrophy, Chiari I, abscess, and others |
| **Real DICOM spine studies** | 8 | The *same anatomy* under T1 and T2, with the scanner's own TE/TR/field-strength values |
| 3-D volumes (NIfTI) | 6 | Knee MRI, abdominal MRI, jaw CT, walnut micro-CT, synthetic control |

Total: 42 MB, loads in milliseconds, and needs only NumPy to read.

### Layer 3: the interactive web app (`app/streamlit_app.py`)

Six tabs controlled by sliders. A reconstruction takes about 1 ms, so the
display updates while a slider is dragged. This is what is shown in the demo.

---

## 5. Signals and systems concepts used

| Concept | Where it appears |
|---|---|
| **2-D DFT / FFT** | The entire forward and inverse pipeline |
| **Sampling and Nyquist** | Undersampling k-space below Nyquist is what causes the artifacts |
| **Aliasing** | Cartesian undersampling folds the image onto itself as ghost copies, spaced by FOV / acceleration factor |
| **Low-pass / high-pass filtering** | Centre-only vs edges-only masking, an ideal circular filter of each type |
| **Gibbs phenomenon** | Ringing around sharp edges when k-space is cut off at a hard edge, the 2-D version of truncating a Fourier series |
| **Convolution theorem** | Multiplying k-space by a mask is the same as convolving the image with the mask's PSF, so each mask's artifact pattern is the Fourier transform of that mask |
| **Spectral energy distribution** | 92.6% of k-space energy sits in the central 10% radius, measured across all 40 samples |
| **fftshift conventions** | Centred k-space throughout; `ifftshift` (not `fftshift`) to undo it, which differs for odd sizes |
| **Additive white Gaussian noise** | Complex AWGN in k-space, with the I/Q channel model |
| **Sparsity and L1 minimisation** | Compressed sensing via iterative soft-thresholding (FISTA) |
| **Fourier shift theorem** | Patient motion during acquisition: a spatial shift is a phase ramp in k-space, not a magnitude change |

The convolution theorem also answers the question "why does Cartesian give
ghosts and radial give streaks?": *masking in k-space is convolution in image
space, and the artifact is the point-spread function, which is the Fourier
transform of the mask. A comb of lines transforms to a comb of shifted copies;
spokes transform to streaks.*

---

## 6. The five-minute live demo

In this order, since each step leads into the next.

**1. Open tab 1, pick a brain tumour case, leave sampling at 100%.**
Point out the five panels: truth, k-space, mask, acquired, reconstruction.
Say: *"The scanner measures the middle panel, not the first one."* Mention that
k-space is shown on a log scale because its dynamic range is about 72 dB; on a
linear scale it is a single bright dot.

**2. Drag the ratio down to 25%, strategy = Cartesian.**
Ghosts appear. *"The scan is now 4x faster. Those ghost copies are aliasing:
skipping lines in k-space folds the image onto itself."*

**3. Switch strategy to Radial, then to Random variable-density, still at 25%.**
The artifact changes type (streaks, then noise-like grain) while the sample
count stays the same. PSNR goes 24.1, 27.5, 35.8 dB. *"Same number of
measurements, same scan time. The only change is which points were chosen."*

**4. Go to tab 2 (Centre vs edges), keep 10% in both.**
Centre-only: recognisable, correct brightness, blurred. Edges-only: **nearly
black**. Then point to the contrast-stretched panel, which shows an edge map.
*"Same 10% budget. The centre carries contrast and shape; the edges carry only
boundaries. Removing the centre removes the DC term, which is the average
brightness of the whole image."*

**5. Tab 4 (Compressed sensing), ratio 0.0625.**
Zero-fill vs FISTA on the same measurements: +2.06 dB, +0.124 SSIM. *"Same data.
The only difference is what we assume about the samples we never measured: zero,
or whatever makes the image sparsest while still matching what we did measure."*

**6. Finish on the About tab.**
Show the source file path, the scanner's TE/TR, and the tumour mask overlay.
*"Every sample traces back to a specific file in the clinical dataset."*

---

## 7. Strengths of the project

### Real clinical data, with provenance

40 selected slices across four file formats (HDF5/MATLAB, JPEG, DICOM, NIfTI),
each recording the source file it came from. The DICOM spine samples include the
scanner's acquisition parameters, so T1 (TE 9.2 ms, TR 620 ms) can be shown next
to T2 (TE 94 ms, TR 3370 ms) of **the same patient's spine**. This shows, using
the dataset itself, that MRI contrast depends on the acquisition settings and
not only on the anatomy.

### Results include where compressed sensing does worse

| Sampling | Zero-fill | CS (FISTA) | Change |
|---|---|---|---|
| 25% (4x) | 35.84 dB / 0.933 | 34.65 / 0.939 | **-1.19 dB**, +0.006 |
| 12.5% (8x) | 30.37 / 0.807 | 31.54 / 0.901 | +1.17 dB, +0.094 |
| 6.25% (16x) | 26.10 / 0.643 | 28.16 / 0.767 | +2.06 dB, +0.124 |

At 4x the variable-density mask already captures nearly all the signal energy,
so zero-filling is close to perfect and the sparsity prior costs more than it
gains.

### It is interactive

Sliders, live reconstruction, 40 subjects, and downloadable metrics. Any setting
can be tried during the demo.

---

## 8. Likely questions

**"Is this real MRI data or simulated?"**
The *images* are real clinical MRI and CT. The *k-space* is simulated: a forward
FFT on the images produces the measurements a scanner would have made. Raw
scanner data involves multi-coil arrays and vendor corrections that are out of
scope. Everything after the forward FFT is the same as the real pipeline, and
this approach gives an exact ground truth to measure against.

**"Where does the phase come from, if your sources are magnitude images?"**
A synthetic smooth phase map (a B0-like quadratic bowl, a gradient ramp, and
coil-sensitivity-like ripples) is added before the FFT. Without it the image
would be purely real, which makes k-space exactly Hermitian-symmetric
(`K(-k) = conj(K(k))`). Half the data would then be a copy of the other half,
and any partial-Fourier demonstration would be unrealistically perfect. The
manifest reports a Hermitian-asymmetry value of 0.83-1.70 per sample, versus
about 0 for a real-valued image, so the effect is measured.

**"Why does everything keep the centre of k-space?"**
Because that is where the signal is. Across all 40 samples, the central 10%
radius (about 1% of the samples) holds a mean of **92.6%** of the total k-space
energy (range 73.6-98.9%). Low frequencies encode contrast and overall shape;
the outer region encodes fine detail. Dropping the centre loses the image;
dropping the outer region loses sharpness.

**"Why is radial better than Cartesian on PSNR but worse on SSIM?"**
(At 25% on the glioma case: radial 27.5 dB / 0.651, Cartesian 24.1 dB / 0.722.)
The two metrics measure different things. PSNR is based on mean squared error,
and radial oversamples the centre, so its total error energy is lower. SSIM
measures local structural similarity, and radial's streaks cross tissue
boundaries, which SSIM penalises heavily. This is why one number is not enough
to judge a reconstruction.

**"Does patient motion matter more than undersampling?"**
Yes. A 5-pixel jerk mid-scan costs on the order of 100+ dB, even fully sampled.
Motion is a phase corruption (Fourier shift theorem), and it affects the data
whether or not every k-space point was measured. This is one reason fast
protocols are used for children: the patient moving is a larger problem than
undersampling.

**"What is compressed sensing doing?"**
Zero-filling answers "what was the unmeasured data?" with "zero", which is
wrong. CS instead asks: of all images consistent with the measurements, which is
sparsest in a wavelet basis? It alternates two steps (enforce the measured
samples, then shrink small wavelet coefficients to zero) and needs *random*
sampling, because Cartesian ghosts are just as sparse as real anatomy and a
sparsity prior cannot separate them.

**"How do you know your FFT round trip is correct?"**
Every sample is checked when the store is built: reconstructing the untouched
k-space reproduces the stored image with a maximum absolute error of 3.6e-7
across all 40, which is float32 storage precision. The noise simulator also
checks itself: requesting 20 dB SNR measures back as 19.99 dB.

**"What would you do next?"**
Multi-coil parallel imaging (SENSE/GRAPPA), which is how real scanners
accelerate; total-variation regularisation alongside wavelets; and testing on
raw scanner data from the fastMRI dataset.

---

## 9. Limitations

These should be stated in the presentation:

- **The k-space is simulated**, via forward FFT from reconstructed images.
- **The phase map is synthetic**, because the source files are magnitude images.
- **The pathology JPEGs are lossy-compressed at source**, so some high-frequency
  content was already gone. (JPEG discards high frequencies for a similar reason
  to undersampling.)
- **Slice selection is automated, not clinical.** The pathology folders mix
  imaging planes with no labels, so slices are chosen by a rule on the pixels:
  a left-right mirror correlation of at least 0.90 identifies axial views (the
  head is nearly symmetric), then the slice showing the most brain is picked.
  Four diagnoses have no axial candidate and fall back to sagittal, which is the
  usual view for those conditions anyway.
- **Radial's motion advantage did not appear with simple spoke ordering.**
  The argument in section 3 predicts that timing motion per spoke would let
  radial beat Cartesian under identical motion. With spokes in sequential angle
  order it did not, at every ratio and motion model tried. The likely reason is
  that motion-tolerant radial scans depend on interleaved spoke order
  (golden-angle sampling), not just per-spoke timing. This is a hypothesis, not
  a verified result, and should be described that way if asked.

---

## 10. Key numbers

| Quantity | Value |
|---|---|
| Raw dataset | 12 GB, 43,622 files |
| Sample store | 40 samples, 256x256, 42 MB |
| Energy in central 10% radius | **92.6%** mean (73.6-98.9%) |
| k-space dynamic range | 60-85 dB (why a log display is needed) |
| FFT round-trip error | 3.6e-7 |
| Reconstruction speed | about 1 ms (so sliders update live) |
| Best CS gain | +2.06 dB, +0.124 SSIM at 16x acceleration |
| Sampling strategies | 5 (Cartesian, radial, variable-density, centre-only, edges-only) |
| Motion cost, fully sampled | 5 px jerk: 22.75 dB / 0.793 SSIM (vs. exact with no motion) |
| Uniform-shift verification error | 6.7e-16 (machine precision) |

---

## 11. Running it

```bash
streamlit run app/streamlit_app.py     # then http://localhost:8501
```

Run from the project root. `python main.py --help` lists the command-line
options for the batch pipeline.
