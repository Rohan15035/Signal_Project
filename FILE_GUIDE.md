# File Guide

What every file in this project does. Read this before touching anything.

## The one idea

```
image --FFT--> k-space --mask--> fewer samples --inverse FFT--> damaged image
```

A real scanner measures k-space directly. We fake it by FFT-ing a normal image,
so we have a ground truth to score against. The **only** thing that changes
between experiments is which k-space points the mask keeps.

**Centre of k-space = brightness, contrast, overall shape. Edges = fine detail.**
That is why every mask protects the centre.

---

## `main.py` (487 lines) — the runner

Runs every (strategy x ratio) combination and saves figures.

| Function | What it does |
|---|---|
| `parse_args()` | CLI flags. Stage 2 (`--noise-snr`, `--cs`, `--center-edges`) and reduced-FOV (`--roi`, `--roi-factor`, `--roi-center`) are all off by default |
| `run(args)` | The experiment grid. Loads image, FFTs it **once**, then loops strategies and ratios |
| `write_table(runs, outdir)` | Prints the results table, writes `outputs/metrics.csv` |
| `run_reduced_fov(args, image, tumor_mask)` | Only under `--roi`. The non-locality proof, then the four-way ROI comparison, then the compact reconstruction. Writes three `roi_*.png` figures |

The core loop inside `run()` is only four lines per experiment:

```python
mask   = ks.build_mask(kind, image.shape, ratio, **extra)
masked = ks.apply_mask(full_kspace, mask)      # zero-fill what we skipped
recon  = ks.from_kspace(masked)                # inverse FFT + magnitude
scores = metrics.compute_metrics(image, recon) # PSNR / SSIM
```

There is an `assert` after the forward FFT checking the round trip is under
`1e-9`. If that ever fires, an `fftshift`/`ifftshift` is wrong. Don't delete it.

---

## `mri_sim/` — the simulator

### `kspace.py` (592 lines) — **the important file**

Everything that matters lives here. Convention: **every k-space array is
centred** (`fftshift` applied, DC in the middle), so masks can just reason about
distance from the centre.

**Transforms**

| Function | Code | Meaning |
|---|---|---|
| `to_kspace(image)` | `fftshift(fft2(image))` | Forward. The step a real scanner skips |
| `from_kspace(k)` | `abs(ifft2(ifftshift(k)))` | Inverse. `ifftshift` not `fftshift` — they differ for odd sizes |
| `apply_mask(k, mask)` | `k * mask` | Zero-fill the unmeasured points |
| `reconstruct(k, mask)` | both of the above | The whole zero-fill reconstruction |

`from_kspace` takes `np.abs` because masked k-space is no longer the transform
of a real image, so the result has a genuine imaginary part. Real scanners do
exactly this and display the magnitude.

**The three sampling masks** — each takes `(shape, ratio)` and returns a 0/1 array:

- **`cartesian_mask`** — keeps whole horizontal rows. One row = one
  phase-encode step = one unit of scan time, so skipping rows is a real
  speed-up. Two parts: a fully-sampled centre block (default `0.32 * ratio`
  of rows), then every Nth row outside it. Produces **coherent ghosts**.
- **`radial_mask`** — spokes through the centre, angles spread over `[0, pi)`.
  Binary-searches the spoke count to hit the target ratio. `_draw_spokes()`
  rasterises each line by rounding to the nearest grid point. Produces
  **incoherent streaks**, and oversamples the centre for free.
- **`variable_density_mask`** — random coin flip per point, biased by radius:
  `p(r) = alpha * (1 - r)**poly_order`, with `alpha` bisected so the mean
  probability equals `ratio`. A small central disc is forced to `p = 1`.
  Randomness is what makes compressed sensing work.

**Two teaching masks** (not real acquisitions):
`center_only_mask` (ideal low-pass -> blurry but correct contrast) and
`edges_only_mask` (ideal high-pass -> black with bright outlines).

**Registry:** `MASK_BUILDERS` dict + `build_mask(kind, shape, ratio)` so callers
loop over strategies by name. Add new masks here.

### `metrics.py` (223 lines)
`compute_psnr`, `compute_ssim`, `compute_metrics`. Thin wrappers over
scikit-image. `data_range` is taken from the **original**, never the
reconstruction — otherwise a few bright artifact pixels would inflate the score.
Raises if handed a complex array, to catch a missing `np.abs`.

Plus the ROI-restricted versions used by `roi.py`:
`compute_psnr_in_roi`, `compute_ssim_in_roi`, `compute_metrics_in_roi(original,
recon, roi_mask)` — score **only inside a mask**. A targeted scan that nails the
lesion and leaves the skull a smear is a good scan; whole-image metrics call
that a failure. PSNR just averages over fewer pixels. SSIM cannot: it is a
windowed statistic, so the map is computed over the whole image and averaged
over the ROI **eroded by half a window**, keeping only pixels whose entire 7×7
neighbourhood is inside the region. Without that erosion an exact ROI scores
0.93 instead of 1.00, purely from windows straddling the boundary.

### `io_utils.py` (110 lines)
`load_phantom()`, `load_image_file()`, `load_image()`. Everything ends up
2-D float64 rescaled to `[0, 1]`, so `data_range=1.0` works everywhere.

### `noise.py` (181 lines) — Stage 2
Noise belongs **in k-space, not the image**: it is complex (I/Q receiver
channels) and white, so the same absolute noise destroys the tiny outer samples
while barely denting the huge centre. Detail dies before contrast.

`simulate_acquisition(k, mask, snr_db, seed)` is the one to call — it masks
**first**, then adds noise only to measured points, because an unmeasured point
is absent, not a noisy zero. That ordering is what makes noise scale as `sqrt(N)`.

### `cs.py` (355 lines) — Stage 2
Compressed sensing. Zero-filling assumes unmeasured points are zero; CS instead
asks which image is *simplest* while still matching the measurements:

```
minimise  ||M F x - y||^2  +  lambda * ||W x||_1
```

`ista_reconstruct(...)` runs FISTA, alternating **data consistency** (put back
what we measured) and **soft-thresholding wavelet coefficients** (shrink small
ones to zero). `compare_with_zero_fill(...)` runs both and scores them.

Two traps, both commented in the file: the iteration must stay **complex**
(`FFT(|x|) != FFT(x)`), and the wavelet **approximation band must not be
thresholded** (it is dense and high-energy, not sparse).

CS only beats zero-fill at high acceleration — roughly break-even at 4x,
clearly ahead at 8x and 16x.

### `roi.py` (597 lines) — reduced-FOV / inner-volume imaging
"Can we scan only the lesion?" Yes, 16x faster at full resolution — but **not**
by keeping "the part of k-space where the lesion is". There is no such part.

Two things this file proves, in order:

1. **k-space is not spatially local.** `kspace_locality_demo(image, quadrant)`
   zeros one quadrant of k-space and measures the mean error per *image*
   quadrant: `0.0388 0.0223 0.0273 0.0334` — spread everywhere, not localised.
   Position lives in the **phase relationships between** samples, not in where
   samples sit. So knowing where the tumour is tells you nothing about which
   samples to keep.
2. **The real lever is `dk`, not `k_max`.** `FOV = 1/dk` and
   `dx = 1/(2*k_max)` are independent. Centre-only masking shrinks `k_max` and
   costs resolution. Reduced-FOV shrinks the *object* instead — a spatially
   selective RF pulse excites only a box of size `N/R` — which lets you space
   samples `R` times further apart with `k_max`, and therefore resolution,
   untouched.

| Function | What it does |
|---|---|
| `roi_box(shape, center, R)` | Places the `N/R` box, clamped inside the image. Returns an `ROIBox` with `.slices`, `.mask(shape)` |
| `roi_center_from_mask(m)` | Centroid of an expert tumour mask — where to aim |
| `excitation_profile(shape, box)` | The RF pulse, as an ideal 0/1 box |
| `coarse_grid_mask(shape, R)` | Every Rth sample in both axes, anchored on DC. Full `k_max` — this is *not* a Stage 1 mask |
| `reduced_fov_acquire(image, center, R)` | Excite the box, forward FFT, decimate. `-> (excited, kspace, mask)` |
| `reduced_fov_reconstruct(k, R)` | `from_kspace(k) * R*R` — the `R^2` is density compensation for keeping `1/R^2` of the points |
| `compact_reconstruct(k, R, box)` | The honest `(N/R)x(N/R)` output a scanner returns. Comes out wrapped; `box` rolls the wrap out |
| `compare_roi_strategies(...)` | The four-way table below |
| `reduction_factors(shape)` | Which `R` are legal — the app only offers these |

Measured on `brain-pituitary-1111`, `R = 4`, all rows using ~6.25% of k-space,
all scored **inside the ROI only**:

```
reduced_fov       6.25%   16.0x   PSNR  329.48   SSIM 1.0000
undersampled      6.36%   15.7x   PSNR   24.14   SSIM 0.7148
low_pass          6.27%   15.9x   PSNR   24.69   SSIM 0.7657
no_suppression    6.25%   16.0x   PSNR  -12.89   SSIM 0.0285
```

The last row uses *identical* samples to the first, with the RF excitation
removed: the whole head still makes signal, so 16 pieces of anatomy fold onto
the ROI and the error exceeds the signal. **The suppression is the method.**

**Two traps, both guarded.** `R` must divide `N` exactly, or the aliasing
period `N/R` is fractional and you get a silent ~40 dB instead of 330 —
`_validate_reduction` raises. And a reduced-FOV reconstruction is *tiled*
outside the box, so it must be scored with `metrics.compute_metrics_in_roi`,
never with whole-image metrics.

Note it re-derives k-space from the magnitude image with `to_kspace` rather than
using the store's stored k-space — it has to, because the RF excitation is a
multiplication in image space and must happen before the forward transform.

### `visualize.py` (767 lines)
All matplotlib. `plot_reconstruction_panel` (original | mask | recon | error),
`plot_mask_gallery`, `plot_center_vs_edges`, `plot_cs_comparison`,
`plot_metrics_summary`. `log_magnitude()` does `log(1 + |k|)` because raw
k-space has ~70 dB of dynamic range and looks like a single white dot otherwise.

ROI figures: `plot_kspace_nonlocality` (the misconception, with the four
quadrant errors printed on the error map), `plot_reduced_fov_panel` (the
four-way table as a 4x5 panel) and `plot_compact_reconstruction` (full-grid vs
the small image a scanner returns). Everything is displayed on a fixed `[0, 1]`
range with one exception, flagged in its own title: the no-suppression row sums
`R^2` folded copies and overflows past 1.0, so it gets its own scale rather than
rendering as a white square.

---

## `kspace_store/` — building the real-data store

Turns raw medical datasets into ready-to-use k-space samples. **Run once.**
Reading the finished store needs numpy only.

| File | Role |
|---|---|
| `sources.py` | Format readers: `.mat` (HDF5), NINS JPEGs, DICOM, NIfTI |
| `prepare.py` | Crop, normalise, resize to 256. `synthetic_phase()` adds a realistic B0 bowl + linear ramp + coil ripples; `build_kspace()` FFTs it |
| `catalog.py` | Which files to pull and what to label them |
| `build.py` | Driver — writes `.npz` samples, previews, `manifest.json` |
| `store.py` | **The read API.** `KSpaceStore(root).load(id)` -> `Sample` |
| `demo.py` | `python -m kspace_store.demo --list` |

`Sample` carries `.kspace` (complex64, centred), `.image` (float32 ground
truth), `.phase`, and `.tumor_mask` (uint8, expert segmentation, on 12 brain
cases). Also `.reconstruct(mask)` and `.log_kspace(mask)`.

**The synthetic phase matters.** It makes k-space non-Hermitian
(`hermitian_asymmetry` ~0.8-1.3 in the manifest), which is realistic and means
tricks that assume a real-valued image will not work for free.

---

## `data/kspace_store/` — the built store (40 samples)

`manifest.json` (metadata + stats), `samples/*.npz` (arrays),
`previews/*.png`. 12 brain tumour cases with masks, 8 spine DICOM, plus knee,
abdomen, jaw CT, and a synthetic control.

---

## `app/streamlit_app.py` (939 lines)

`streamlit run app/streamlit_app.py`. Seven tabs: **Acquire**, **Centre vs
edges**, **Noise**, **Compressed sensing**, **Sweep**, **Reduced FOV (ROI)**,
**About this sample**. Sidebar picks subject, strategy, ratio, SNR. Uses
`@st.cache_data` so dragging a slider does not re-run everything.

The ROI tab has its own controls rather than using the sidebar's, because a
reduced-FOV scan is a different acquisition: it is defined by where the box is
and how coarse the grid is, not by a strategy and a ratio. `R` is a
`select_slider` over `roi.reduction_factors()`, so an illegal factor cannot be
chosen at all.

---

## Everything else

| Path | What |
|---|---|
| `outputs/` | Generated figures + `metrics.csv`. Safe to delete and regenerate |
| `README.md` | Full write-up |
| `PROJECT_BRIEF.md` | Presentation notes, demo script, likely questions |
| `requirements.txt` | numpy, scipy, matplotlib, scikit-image + PyWavelets, streamlit, pandas; build-only: h5py, pillow, pydicom, nibabel |
| `Dataset/` | Raw source data. Only needed to rebuild the store |

## Running it

```bash
python main.py                              # phantom, defaults
python main.py --sample brain-glioma-778    # a real brain scan
python main.py --cs --center-edges --noise-snr 20   # Stage 2 extras
python main.py --sample brain-pituitary-1111 --roi  # reduced-FOV ROI scan
streamlit run app/streamlit_app.py          # interactive
```

## Still to do

**Task A, motion simulation** (`TASKS_MOTION_AND_ROI.md`) is not started. Task
B, reduced-FOV ROI imaging, is done — see `mri_sim/roi.py` above.
