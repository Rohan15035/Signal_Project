# App Walkthrough — what happens, and which code does it

Companion to [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md). That one is the
script for tomorrow; this one is the reference for **"sir, what exactly is
happening when you move that slider?"**

Every tab is covered, but Tabs 1 and 2 are done in full detail because those are
the two you'll spend the most time on.

---

## 0. First: how the app is wired

You need this to explain anything else, and it's the single most common
misunderstanding about Streamlit.

**Streamlit has no event handlers.** There is no "on slider change" callback.
Instead, when you touch *any* widget, Streamlit **re-executes the entire script
from line 1 to line 616**. Widgets return their current value on the way past,
and the page is rebuilt from scratch.

So "moving the slider" really means: *the whole program runs again with a
different number in one variable.*

Two consequences worth knowing:

**1. Every tab's code runs on every interaction.** `st.tabs` is not lazy — the
tab bodies all execute, and the tabs you can't see are just hidden with CSS.
Move the sidebar ratio slider and Tab 2, Tab 3 and Tab 5 all recompute too, even
though you're looking at Tab 1.

**2. That would be unusably slow, so the expensive work is cached.**
`@st.cache_data` hashes the function's arguments; identical arguments return the
stored result instead of recomputing. This is what makes the sliders feel
instant.

| Function | Line | Cached | Cost on a miss |
|---|---|---|---|
| `get_store` | [41](app/streamlit_app.py#L41) | `@st.cache_resource` | opens the manifest, once per server |
| `load_sample` | [48](app/streamlit_app.py#L48) | yes | ~10 ms disk read of the `.npz` |
| `build_mask` | [64](app/streamlit_app.py#L64) | yes | <1 ms, except radial (see below) |
| `acquire` | [71](app/streamlit_app.py#L71) | yes | ~2 ms (one FFT + metrics) |
| `run_cs` | [93](app/streamlit_app.py#L93) | yes | **~1 s** (80 FFT pairs) |
| `sweep` | [118](app/streamlit_app.py#L118) | yes | 15 reconstructions |

If asked *"why did you cache it?"* — without the cache, dragging the ratio
slider would re-run FISTA in Tab 4 on every single pixel of slider movement.

---

## 1. The sidebar — the controls shared by every tab

Defined at [lines 210–266](app/streamlit_app.py#L210-L266).

| Control | Variable | Feeds |
|---|---|---|
| Collection / Subject | `sample_id` | everything |
| Sampling strategy | `strategy` | Tabs 1, 3, 5 |
| **k-space sampled** (0.02–1.0) | `ratio` | Tabs 1, 3 |
| Simulate scanner noise | `add_noise`, `snr_db` | Tabs 1, 4, 5 |
| Random seed | `seed` | random mask draw + noise realisation |

After the sidebar, three lines at [278–282](app/streamlit_app.py#L278-L282) do
the work that Tab 1 displays:

```python
image, full_kspace, tumor_mask, meta = load_sample(sample_id)
mask = build_mask(strategy, image.shape, ratio, int(seed))
acquired, reconstruction, scores = acquire(sample_id, strategy, ratio, snr_db, int(seed))
```

---

## 2. Tab 1 — Acquire

### What is on screen

Five panels left to right, the pipeline in order:

| # | Panel | Source |
|---|---|---|
| 1 | Ground truth | `image`, straight from the store |
| 2 | Full k-space, log\|K\| | `full_kspace`, from the store |
| 3 | The mask | `mask` |
| 4 | What the scanner got | `acquired` |
| 5 | Reconstruction | `reconstruction` |

Below: PSNR/SSIM cards, an absolute-error map, and a strategy-specific
explanation box.

### What happens when you drag the ratio slider

Say you move it from 0.25 to 0.24. In order:

**Step 1 — the script restarts.** `ratio` is now `0.24`.

**Step 2 — `load_sample` is a cache hit.** `sample_id` didn't change, so nothing
is read from disk. Panels 1 and 2 will redraw with *identical* data.

> **This is worth saying out loud:** "Notice the first two panels don't change.
> The patient is the same and the full k-space is the same — I'm only changing
> which part of it the scanner is allowed to keep."

**Step 3 — `build_mask` is a cache miss** → [`ks.build_mask`](mri_sim/kspace.py#L311)
dispatches on the strategy name to one of three builders:

- **Cartesian** → [`cartesian_mask`](mri_sim/kspace.py#L78). Computes a line
  budget `n_target = round(ratio * ny)`, reserves `0.32 * ratio` of the lines for
  a fully-sampled centre block, and spreads what's left evenly over the outer
  lines. **A whole row is kept or dropped** — never individual points — because
  one row is one phase-encoding step, which is the real unit of scan time.
- **Radial** → [`radial_mask`](mri_sim/kspace.py#L127). You asked for a *fraction*,
  but the mask is built from *spokes*, and there's no formula linking the two
  (spokes overlap near the centre). So it **binary-searches** the spoke count,
  rasterising a trial mask at each step via
  [`_draw_spokes`](mri_sim/kspace.py#L172) until coverage first exceeds your
  target. This is the slow one, ~10 rasterisations per new ratio — and exactly
  why the achieved ratio is always a little *above* what you asked for.
- **Variable density** → [`variable_density_mask`](mri_sim/kspace.py#L195). Builds
  a probability map `p(r) = α(1−r)^6`, then **bisects on α** until the mean
  probability equals your ratio, forces the central 5% disc to `p = 1`, and
  flips a biased coin per point. The `seed` makes the draw reproducible.

**Step 4 — `acquire` is a cache miss.** Three lines of real signal processing:

```python
acquired       = ks.apply_mask(full_kspace, mask)   # zero-fill the unmeasured points
reconstruction = ks.from_kspace(acquired)           # ifftshift → ifft2 → abs
scores         = metrics.compute_metrics(image, reconstruction)
```

[`apply_mask`](mri_sim/kspace.py#L26) is one multiply. **It does not delete rows
— it zeroes them**, keeping the array the same size, because the inverse FFT
needs a full-size grid. That zero-filling *is* the wrong assumption, and every
artifact you see downstream is its consequence.

[`from_kspace`](mri_sim/kspace.py#L18) is the whole reconstruction in one line.

**Step 5 — panels 3, 4, 5 and the metrics redraw.**

Total: ~2 ms for Cartesian and variable-density, noticeably more the first time
you hit a new radial ratio.

### Why each strategy fails the way it does

This is the physics your instructor is actually testing.

**Cartesian → crisp evenly-spaced ghosts.**
Keeping every Nth row is uniform sampling in k-space. Uniform undersampling in
one domain is *aliasing* in the other — so the image folds onto itself, producing
sharp copies displaced by FOV/acceleration. Same Nyquist violation as 1-D
aliasing, with the two domains swapped from the usual textbook picture. These
ghosts are **coherent**: they look exactly like real anatomy, which is what makes
them the hardest artifact to remove.

**Radial → streaks radiating outward.**
Every spoke passes through the centre, so low frequencies are massively
oversampled for free. But the gaps between spokes widen with radius, so the outer
k-space is sampled unevenly in angle. Missing high-frequency wedges show up as
streaks off sharp edges. Incoherent, so less destructive-looking than ghosts —
but note in Tab 5 that radial's **SSIM is the worst of the three**, because
streaks are still structure that wasn't in the original.

**Variable density → noise-like grain.**
Random sampling means the aliasing isn't coherent at all. The error is spread
thinly across the whole image as grain instead of concentrated into replicas.
This is the property compressed sensing needs, and Tab 4 exploits it.

### Things you can do live that look good

- Slide to **1.0** → PSNR jumps to ~320 dB. Say: *"that's not a quality
  measurement, that's float64 round-off. The reconstruction is exact."*
- Slide to the **0.02 floor** → the image collapses. Even here, Cartesian keeps
  its centre block, so you still get a blurry blob rather than nothing.
- Hold the ratio fixed and **switch strategy** in the sidebar. Same sample count,
  three completely different failure modes. This is the money shot.

---

## 3. Tab 2 — Centre vs edges

### The claim being tested

Both sides use the **same number of samples**. The only difference is *where*
those samples sit. If the two reconstructions look wildly different, position
matters more than count — which justifies every design choice in Tab 1.

I verified the sample counts are genuinely equal:

| Slider | Centre keeps | Edges keeps |
|---|---|---|
| 0.05 | 3,281 px (5.01%) | 3,273 px (4.99%) |
| 0.10 | 6,557 px (10.01%) | 6,549 px (9.99%) |
| 0.25 | 16,389 px (25.01%) | 16,371 px (24.98%) |

(The few-pixel difference is quantile rounding on the radius map, not a bug.)

### What happens when you drag this slider

This slider is `demo_ratio` at [line 346](app/streamlit_app.py#L346), range
0.01–0.50, default 0.10. It is **local to this tab** — it does not touch the
sidebar ratio.

Note these four lines call `ks.build_mask` **directly, not through the cached
wrapper**, so they genuinely recompute on every rerun. They're cheap enough that
it doesn't matter:

```python
center_mask  = ks.build_mask("center_only", image.shape, demo_ratio)
edges_mask   = ks.build_mask("edges_only",  image.shape, demo_ratio)
center_recon = ks.from_kspace(ks.apply_mask(full_kspace, center_mask))
edges_recon  = ks.from_kspace(ks.apply_mask(full_kspace, edges_mask))
```

**How the two masks are built** — the trick is in
[`_radius_for_ratio`](mri_sim/kspace.py#L256). You can't use the area formula
`πr²/4`, because the corners of a square k-space fall outside the inscribed
circle. So it sorts every pixel's radius and reads off a quantile — exact, by
construction:

- [`center_only_mask`](mri_sim/kspace.py#L263) keeps `radius <= r(ratio)` → a disc
- [`edges_only_mask`](mri_sim/kspace.py#L277) keeps `radius > r(1 − ratio)` → the complementary annulus

**As you drag right:** the centre disc grows, so its reconstruction gets
progressively sharper (more high frequencies admitted). The edges annulus grows
*inward*, so more structure appears — but it stays dark, because it still never
contains DC.

**As you drag left toward 0.01:** the centre becomes a severely blurred blob with
strong ring artifacts, and the edges side becomes almost pure black.

### The number that makes the point

At the default 0.10, on the phantom:

| | Samples kept | Share of k-space energy |
|---|---|---|
| Centre only | 10.0% | **95.08%** |
| Edges only | 10.0% | **0.13%** |

**Same sample count, ~700× difference in energy captured.** Mean brightness of
the reconstructions: 0.132 for centre, 0.004 for edges.

The app pulls the per-sample version of this from the manifest at
[line 393](app/streamlit_app.py#L393) (`meta["stats"]["energy_within_r0.1"]`), a
statistic computed at build time by
[`center_energy_fraction`](kspace_store/prepare.py). Across all 40 samples the
mean is 92.6%.

### The three sub-panels on each side

Each side shows **Mask | As reconstructed | Contrast stretched**. That third
panel exists for an honest reason:

The edges-only reconstruction is genuinely almost black, so on the shared [0, 1]
scale used everywhere in the project it looks like an empty frame — which hides
the point. The stretched version auto-scales to its own min/max and reveals it is
a clean **edge map**. `show_image(..., stretch=True)` at
[line 138](app/streamlit_app.py#L138) handles this.

> Say: *"I'm showing both because the true-brightness version proves the DC term
> is gone, and the stretched version proves the edge information is still there.
> Either one alone would be misleading."*

### Two things to name explicitly

**Gibbs ringing.** The faint concentric rings around sharp boundaries in the
centre-only image. A hard-edged disc in k-space is an ideal low-pass filter;
truncating a Fourier series at a sharp cutoff always overshoots at
discontinuities. It's the 2-D version of the 1-D Gibbs phenomenon from lectures —
a direct callback to course material.

**Why edges-only is black.** Discarding the centre discards the **DC term**,
which *is* the mean brightness of the image. Not "makes it darker" — the average
value is now literally zero, and the magnitude operation is the only reason it
isn't negative.

---

## 4. Tab 3 — Noise

**Control:** `demo_snr` slider, 0–50 dB ([line 411](app/streamlit_app.py#L411)),
independent of the sidebar noise checkbox.

**Four panels:** ground truth · noisy but fully sampled · undersampled but clean
· undersampled **and** noisy. The 2×2 design lets you separate the two effects.

**Code path:**
```python
noisy_full     = noise.add_kspace_noise(full_kspace, snr_db=demo_snr, seed=seed)
noisy_acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=demo_snr, seed=seed)
```

The key function is
[`simulate_acquisition`](mri_sim/noise.py#L77), and the **order of its two lines
is the whole physics**:

```python
acquired = kspace_centered * mask                    # measure first
return add_kspace_noise(acquired, snr_db, mask=mask) # then each measurement picks up noise
```

Noise *then* mask would model a scanner that acquires all of k-space, corrupts
it, and throws most away — which would make the noise level independent of
acceleration. That's wrong, and it's a good thing to be asked about.

`sigma` is derived from the target SNR in
[`noise_sigma_for_snr`](mri_sim/noise.py#L24): complex noise has power `2σ²`, so
`σ = sqrt(mean(|K|²) / (2·10^(SNR/10)))`.

**Self-check on screen:** the caption reports the *measured* SNR via
[`measured_snr_db`](mri_sim/noise.py#L93) next to what you requested. They match.
Point at it — it's free evidence the simulation is calibrated.

**Rician noise.** The expander explains why a noisy MRI background is grey haze
rather than black: `abs()` of zero-mean complex Gaussian noise is strictly
positive and Rician-distributed. This falls out automatically; it isn't modelled
separately.

---

## 5. Tab 4 — Compressed sensing

**Three controls** ([lines 469–481](app/streamlit_app.py#L469-L481)): `cs_ratio`,
`cs_lambda` (λ), and `cs_iters`. All three feed `run_cs`, which is cached — so a
repeat of the same combination is instant, but a new one costs ~1 s and shows a
spinner.

**What each slider does to the maths:**

- **`cs_ratio`** — how much data FISTA has to work with. Below ~0.15 CS pulls
  clearly ahead; above ~0.3 zero-filling is already good and CS can *lose* on
  PSNR. The app detects this and prints a warning instead of a success box
  ([line 508](app/streamlit_app.py#L508)) — don't hide it, explain it.
- **`λ`** — sparsity strength. It's scaled by
  [`_detail_scale`](mri_sim/cs.py#L79) to a fraction of the largest wavelet
  *detail* coefficient, so the same λ means the same thing across images. Push it
  to 0.10 live and watch real anatomy get thresholded away into a waxy, smoothed
  image. Good demonstration of over-regularisation.
- **`cs_iters`** — the convergence chart under the panels comes from the
  per-iteration `history` and flattens well before 80.

**The algorithm** is [`ista_reconstruct`](mri_sim/cs.py#L108). Per iteration:

1. **Data consistency** — FFT the guess, overwrite the measured points with the
   real measurements, inverse FFT. Because the FFT is unitary the gradient step
   size is exactly 1, so there is no learning rate to tune.
2. **Sparsify** — wavelet transform (db4, 3 levels), soft-threshold, invert.
3. **FISTA momentum** — extrapolate past the previous iterate.

**Two implementation details worth defending if asked:**

- [`_ifft_complex`](mri_sim/cs.py#L31) keeps the phase. Using `abs()` inside the
  loop would be a real bug, since `FFT(|x|) ≠ FFT(x)` — the iteration would fight
  itself and converge *worse* than the zero-filled image it started from.
  Magnitude is taken once, at the very end.
- [`_threshold_details`](mri_sim/cs.py#L60) exempts the wavelet **approximation**
  band. That band is a dense, high-energy summary of the anatomy; shrinking it
  every iteration bleeds contrast out of the image and drives PSNR *below*
  baseline. Every practical CS-MRI implementation exempts it.

---

## 6. Tab 5 — Sweep

No sliders of its own. [`sweep`](app/streamlit_app.py#L118) loops all 3
strategies × 5 ratios (100% → 6.25%), calling `acquire` for each — 15
reconstructions, all individually cached, so it's near-instant on a revisit.

Two separate charts, not one with twin axes: PSNR is in dB and SSIM is a unitless
0–1 index, and sharing an axis would invite false slope comparisons. Plus a
sortable table and a CSV download.

**This is where the PSNR/SSIM disagreement is visible** — radial's PSNR line sits
above Cartesian's while its SSIM line sits below. See §3 of
[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for how to present that.

---

## 7. Tab 6 — About this sample

Provenance. Source file path back into the raw dataset, acquisition parameters
straight from the DICOM header (TE, TR, field strength, pixel spacing), tags, the
expert tumour mask as a red overlay where one exists, and the three derived
statistics.

The expander at the bottom is your honesty disclosure: the images are real
clinical scans, the **k-space is simulated** by forward FFT, and the phase is
synthetic because the source files are magnitude-only. Open this tab
*before* you're asked, not after.

---

## 8. One-page code map

| Question | File · function |
|---|---|
| Image → k-space | [kspace.py `to_kspace`](mri_sim/kspace.py#L13) |
| k-space → image | [kspace.py `from_kspace`](mri_sim/kspace.py#L18) |
| Zero-fill the unmeasured points | [kspace.py `apply_mask`](mri_sim/kspace.py#L26) |
| Every Nth line + centre block | [kspace.py `cartesian_mask`](mri_sim/kspace.py#L78) |
| Spokes through the centre | [kspace.py `radial_mask`](mri_sim/kspace.py#L127) |
| Random, dense in the middle | [kspace.py `variable_density_mask`](mri_sim/kspace.py#L195) |
| Centre-only / edges-only | [kspace.py L263](mri_sim/kspace.py#L263), [L277](mri_sim/kspace.py#L277) |
| Noise in k-space | [noise.py `simulate_acquisition`](mri_sim/noise.py#L77) |
| Compressed sensing | [cs.py `ista_reconstruct`](mri_sim/cs.py#L108) |
| PSNR / SSIM | [metrics.py `compute_metrics`](mri_sim/metrics.py#L39) |
| Reading the sample store | [kspace_store/store.py](kspace_store/store.py) |
| The app itself | [app/streamlit_app.py](app/streamlit_app.py) |
| Batch version of all of it | [main.py](main.py) |

**If you only memorise three:** `to_kspace`, `from_kspace`, `cartesian_mask`.
Everything else is a variation on those.
