# App Walkthrough: what happens, and which code does it

Companion to [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md). That one covers
what to show and say; this one explains what the code does when a control in
the app changes.

Every tab is covered, with Tabs 1 and 2 in the most detail because they take the
most time in the demo.

---

## 0. How the app is wired

This is needed to explain everything else.

**Streamlit has no event handlers.** There is no "on slider change" callback.
Instead, when *any* widget changes, Streamlit **re-runs the entire script from
line 1 to line 615**. Widgets return their current value as the script passes
them, and the page is rebuilt from scratch.

So "moving the slider" means: *the whole program runs again with a different
number in one variable.*

Two consequences:

**1. Every tab's code runs on every interaction.** `st.tabs` is not lazy: all
tab bodies execute, and the tabs you can't see are hidden with CSS. Moving the
sidebar ratio slider recomputes Tab 2, Tab 3 and Tab 5 too, even while Tab 1 is
showing.

**2. That would be too slow, so the expensive work is cached.**
`@st.cache_data` hashes the function's arguments; identical arguments return the
stored result instead of recomputing. This is what makes the sliders respond
immediately.

| Function | Line | Cached | Cost on a miss |
|---|---|---|---|
| `get_store` | [42](app/streamlit_app.py#L42) | `@st.cache_resource` | opens the manifest, once per server |
| `load_sample` | [48](app/streamlit_app.py#L48) | yes | ~10 ms disk read of the `.npz` |
| `build_mask` | [64](app/streamlit_app.py#L64) | yes | <1 ms, except radial (see below) |
| `acquire` | [71](app/streamlit_app.py#L71) | yes | ~2 ms (one FFT + metrics) |
| `run_cs` | [93](app/streamlit_app.py#L93) | yes | **~1 s** (80 FFT pairs) |
| `sweep` | [118](app/streamlit_app.py#L118) | yes | 15 reconstructions |

Without the cache, dragging the ratio slider would re-run FISTA in Tab 4 on
every small movement of the slider.

---

## 1. The sidebar: controls shared by every tab

Defined at [lines 210-266](app/streamlit_app.py#L210-L266).

| Control | Variable | Feeds |
|---|---|---|
| Collection / Subject | `sample_id` | everything |
| Sampling strategy | `strategy` | Tabs 1, 3, 5 |
| **k-space sampled** (0.02-1.0) | `ratio` | Tabs 1, 3 |
| Simulate scanner noise | `add_noise`, `snr_db` | Tabs 1, 4, 5 |
| Random seed | `seed` | random mask draw + noise realisation |

After the sidebar, the lines at [270-274](app/streamlit_app.py#L270-L274) do the
work that Tab 1 displays:

```python
image, full_kspace, tumor_mask, meta = load_sample(sample_id)
mask = build_mask(strategy, image.shape, ratio, int(seed))
acquired, reconstruction, scores = acquire(sample_id, strategy, ratio, snr_db, int(seed))
```

---

## 2. Tab 1: Acquire

### What is on screen

Five panels, left to right, in pipeline order:

| # | Panel | Source |
|---|---|---|
| 1 | Ground truth | `image`, straight from the store |
| 2 | Full k-space, log\|K\| | `full_kspace`, from the store |
| 3 | The mask | `mask` |
| 4 | What the scanner got | `acquired` |
| 5 | Reconstruction | `reconstruction` |

Below them: PSNR/SSIM cards, an absolute-error map, and an explanation box for
the selected strategy.

### What happens when you drag the ratio slider

Say it moves from 0.25 to 0.24. In order:

**Step 1: the script restarts.** `ratio` is now `0.24`.

**Step 2: `load_sample` is a cache hit.** `sample_id` didn't change, so nothing
is read from disk. Panels 1 and 2 redraw with *identical* data.

> Point out: "Notice the first two panels don't change. The patient is the same
> and the full k-space is the same. I'm only changing which part of it the
> scanner is allowed to keep."

**Step 3: `build_mask` is a cache miss**, so
[`ks.build_mask`](mri_sim/kspace.py#L311) picks one of three builders based on
the strategy name:

- **Cartesian**: [`cartesian_mask`](mri_sim/kspace.py#L78). Computes a line
  budget `n_target = round(ratio * ny)`, reserves `0.32 * ratio` of the lines for
  a fully-sampled centre block, and spreads the rest evenly over the outer
  lines. **A whole row is kept or dropped**, never individual points, because
  one row is one phase-encoding step, which is the real unit of scan time.
- **Radial**: [`radial_mask`](mri_sim/kspace.py#L127). The slider gives a
  *fraction*, but the mask is built from *spokes*, and there's no formula
  linking the two (spokes overlap near the centre). So it **binary-searches**
  the spoke count, drawing a trial mask at each step with
  [`_draw_spokes`](mri_sim/kspace.py#L172) until coverage first reaches the
  target. This is the slow one, about 10 trial masks per new ratio, and it is
  why the achieved ratio is always slightly *above* the requested one.
- **Variable density**: [`variable_density_mask`](mri_sim/kspace.py#L195). Builds
  a probability map `p(r) = alpha * (1 - r)^6`, then **bisects on alpha** until
  the mean probability equals the ratio, forces the central 5% disc to `p = 1`,
  and flips a biased coin per point. The `seed` makes the draw reproducible.

**Step 4: `acquire` is a cache miss.** Three lines of signal processing:

```python
acquired       = ks.apply_mask(full_kspace, mask)   # zero-fill the unmeasured points
reconstruction = ks.from_kspace(acquired)           # ifftshift -> ifft2 -> abs
scores         = metrics.compute_metrics(image, reconstruction)
```

[`apply_mask`](mri_sim/kspace.py#L26) is one multiplication. **It does not
delete rows; it zeroes them**, keeping the array the same size, because the
inverse FFT needs a full-size grid. That zero-filling is the wrong assumption,
and every artifact downstream comes from it.

[`from_kspace`](mri_sim/kspace.py#L18) is the whole reconstruction in one line.

**Step 5: panels 3, 4, 5 and the metrics redraw.**

Total: about 2 ms for Cartesian and variable-density, noticeably more the first
time a new radial ratio is used.

### Why each strategy fails the way it does

**Cartesian: sharp, evenly spaced ghosts.**
Keeping every Nth row is uniform sampling in k-space. Uniform undersampling in
one domain causes *aliasing* in the other, so the image folds onto itself,
producing sharp copies displaced by FOV/acceleration. It is the same Nyquist
violation as 1-D aliasing, with the two domains swapped from the usual textbook
case. These ghosts are **coherent**: they look like real anatomy, which makes
them the hardest artifact to remove.

**Radial: streaks radiating outward.**
Every spoke passes through the centre, so low frequencies are heavily
oversampled. But the gaps between spokes widen with radius, so the outer
k-space is sampled unevenly in angle. Missing high-frequency wedges show up as
streaks off sharp edges. The streaks are less coherent than ghosts, so they look
less destructive, but in Tab 5 radial's **SSIM is the worst of the three**,
because streaks are still structure that wasn't in the original.

**Variable density: noise-like grain.**
Random sampling means the aliasing isn't coherent. The error is spread thinly
across the whole image as grain instead of concentrated into copies. Compressed
sensing relies on this property, and Tab 4 uses it.

### Things to try live

- Slide to **1.0**: PSNR jumps to about 320 dB. Say: *"that's not a quality
  measurement, that's float64 round-off. The reconstruction is exact."*
- Slide to the **0.02 minimum**: the image collapses. Even here, Cartesian keeps
  its centre block, so the result is a blurry blob rather than nothing.
- Keep the ratio fixed and **switch strategy** in the sidebar. Same sample count,
  three different kinds of failure. This is the most useful comparison to show.

---

## 3. Tab 2: Centre vs edges

### What it tests

Both sides use the **same number of samples**. The only difference is *where*
those samples are. If the two reconstructions look very different, position
matters more than count, which justifies the design choices in Tab 1.

The sample counts are equal:

| Slider | Centre keeps | Edges keeps |
|---|---|---|
| 0.05 | 3,281 px (5.01%) | 3,273 px (4.99%) |
| 0.10 | 6,557 px (10.01%) | 6,549 px (9.99%) |
| 0.25 | 16,389 px (25.01%) | 16,371 px (24.98%) |

(The few-pixel difference comes from quantile rounding on the radius map, not a
bug.)

### What happens when you drag this slider

This slider is `demo_ratio` at [line 345](app/streamlit_app.py#L345), range
0.01-0.50, default 0.10. It is **local to this tab** and does not affect the
sidebar ratio.

These four lines call `ks.build_mask` **directly, not through the cached
wrapper**, so they recompute on every rerun. They're cheap enough that it
doesn't matter:

```python
center_mask  = ks.build_mask("center_only", image.shape, demo_ratio)
edges_mask   = ks.build_mask("edges_only",  image.shape, demo_ratio)
center_recon = ks.from_kspace(ks.apply_mask(full_kspace, center_mask))
edges_recon  = ks.from_kspace(ks.apply_mask(full_kspace, edges_mask))
```

**How the two masks are built.** The key function is
[`_radius_for_ratio`](mri_sim/kspace.py#L256). The area formula `pi * r^2 / 4`
can't be used, because the corners of a square k-space fall outside the
inscribed circle. So it sorts every pixel's radius and reads off a quantile,
which is exact by construction:

- [`center_only_mask`](mri_sim/kspace.py#L263) keeps `radius <= r(ratio)`, a disc
- [`edges_only_mask`](mri_sim/kspace.py#L277) keeps `radius > r(1 - ratio)`, the complementary ring

**Dragging right:** the centre disc grows, so its reconstruction gets sharper
(more high frequencies included). The edge ring grows *inward*, so more
structure appears, but it stays dark because it still never contains DC.

**Dragging left toward 0.01:** the centre becomes a heavily blurred blob with
strong ring artifacts, and the edges side becomes almost pure black.

### Energy in each region

At the default 0.10, on the phantom:

| | Samples kept | Share of k-space energy |
|---|---|---|
| Centre only | 10.0% | **95.08%** |
| Edges only | 10.0% | **0.13%** |

**Same sample count, about 700x difference in energy captured.** Mean brightness
of the reconstructions: 0.132 for centre, 0.004 for edges.

The app reads the per-sample version of this from the manifest at
[line 391](app/streamlit_app.py#L391) (`meta["stats"]["energy_within_r0.1"]`), a
statistic computed at build time by
[`center_energy_fraction`](kspace_store/prepare.py). Across all 40 samples the
mean is 92.6%.

### The three sub-panels on each side

Each side shows **Mask | As reconstructed | Contrast stretched**. The third panel
is needed because the edges-only reconstruction is almost black: on the shared
[0, 1] scale used everywhere in the project it looks like an empty frame. The
stretched version scales to its own min/max and shows that it is a clean
**edge map**. `show_image(..., stretch=True)` at
[line 138](app/streamlit_app.py#L138) handles this.

> Say: *"I'm showing both because the true-brightness version shows the DC term
> is gone, and the stretched version shows the edge information is still there.
> Either one alone would be misleading."*

### Two things to name

**Gibbs ringing.** The faint rings around sharp boundaries in the centre-only
image. A hard-edged disc in k-space is an ideal low-pass filter, and truncating
a Fourier series at a sharp cutoff always overshoots at discontinuities. It is
the 2-D version of the 1-D Gibbs phenomenon from lectures.

**Why edges-only is black.** Discarding the centre discards the **DC term**,
which *is* the mean brightness of the image. The average value becomes zero,
not just lower, and the magnitude operation is the only reason the image isn't
negative.

---

## 4. Tab 3: Noise

**Control:** `demo_snr` slider, 0-50 dB ([line 410](app/streamlit_app.py#L410)),
independent of the sidebar noise checkbox.

**Four panels:** ground truth, noisy but fully sampled, undersampled but clean,
and undersampled **and** noisy. This 2x2 layout separates the two effects.

**Code path:**
```python
noisy_full     = noise.add_kspace_noise(full_kspace, snr_db=demo_snr, seed=seed)
noisy_acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=demo_snr, seed=seed)
```

The key function is
[`simulate_acquisition`](mri_sim/noise.py#L77), and the **order of its two
lines** is what models the physics:

```python
acquired = kspace_centered * mask                    # measure first
return add_kspace_noise(acquired, snr_db, mask=mask) # then each measurement picks up noise
```

Adding noise first and masking after would model a scanner that acquires all of
k-space, corrupts it, and throws most of it away, which would make the noise
level independent of acceleration. That would be the wrong model.

`sigma` is derived from the target SNR in
[`noise_sigma_for_snr`](mri_sim/noise.py#L24): complex noise has power
`2 * sigma^2`, so `sigma = sqrt(mean(|K|^2) / (2 * 10^(SNR/10)))`.

**Check on screen:** the caption shows the *measured* SNR from
[`measured_snr_db`](mri_sim/noise.py#L93) next to the requested value. They
match, which confirms the noise level is calibrated.

**Rician noise.** The expander explains why a noisy MRI background is a grey
haze rather than black: `abs()` of zero-mean complex Gaussian noise is always
positive and Rician-distributed. This happens automatically; it isn't modelled
separately.

---

## 5. Tab 4: Compressed sensing

**Three controls** ([lines 468-480](app/streamlit_app.py#L468-L480)): `cs_ratio`,
`cs_lambda` (lambda), and `cs_iters`. All three feed `run_cs`, which is cached,
so repeating a combination is instant, but a new one takes about 1 s and shows a
spinner.

**What each slider changes:**

- **`cs_ratio`**: how much data FISTA has to work with. Below about 0.15, CS
  clearly does better; above about 0.3, zero-filling is already good and CS can
  *lose* on PSNR. The app detects this and shows a warning instead of a success
  box ([line 510](app/streamlit_app.py#L510)), and the warning text explains why.
- **`lambda`**: sparsity strength. It is scaled by
  [`_detail_scale`](mri_sim/cs.py#L79) to a fraction of the largest wavelet
  *detail* coefficient, so the same lambda means the same thing across images.
  Setting it to 0.10 thresholds away real anatomy and gives an over-smoothed
  image, which shows over-regularisation.
- **`cs_iters`**: the convergence chart under the panels comes from the
  per-iteration `history` and flattens well before 80.

**The algorithm** is [`ista_reconstruct`](mri_sim/cs.py#L108). Each iteration:

1. **Data consistency**: FFT the current guess, overwrite the measured points
   with the real measurements, inverse FFT. Because the FFT is unitary, the
   gradient step size is exactly 1, so there is no learning rate to tune.
2. **Sparsify**: wavelet transform (db4, 3 levels), soft-threshold, invert.
3. **FISTA momentum**: extrapolate past the previous iterate.

**Two implementation details:**

- [`_ifft_complex`](mri_sim/cs.py#L31) keeps the phase. Using `abs()` inside the
  loop would be a bug, since `FFT(|x|) != FFT(x)`: the iteration would work
  against itself and converge to something *worse* than the zero-filled image it
  started from. The magnitude is taken once, at the end.
- [`_threshold_details`](mri_sim/cs.py#L60) leaves the wavelet **approximation**
  band unchanged. That band is a dense, high-energy summary of the anatomy;
  shrinking it every iteration drains contrast from the image and pushes PSNR
  *below* the baseline. CS-MRI implementations normally leave it out of the
  thresholding.

---

## 6. Tab 5: Sweep

No sliders of its own. [`sweep`](app/streamlit_app.py#L118) loops over all 3
strategies and 5 ratios (100% down to 6.25%), calling `acquire` for each. That is
15 reconstructions, each cached, so revisiting the tab is almost instant.

Two separate charts rather than one with twin axes: PSNR is in dB and SSIM is a
unitless 0-1 index, and a shared axis would suggest misleading slope
comparisons. There is also a sortable table and a CSV download.

**The PSNR/SSIM disagreement is visible here**: radial's PSNR line sits above
Cartesian's while its SSIM line sits below. See section 3 of
[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for how to present that.

---

## 7. Tab 6: About this sample

Provenance: the source file path in the raw dataset, acquisition parameters from
the DICOM header (TE, TR, field strength, pixel spacing), tags, the expert tumour
mask as a red overlay where one exists, and the three derived statistics.

The expander at the bottom states the simulation's assumptions: the images are
real clinical scans, the **k-space is simulated** by forward FFT, and the phase
is synthetic because the source files are magnitude-only. It is worth showing
this tab during the demo rather than waiting for the question.

---

## 8. Code map

| Question | File and function |
|---|---|
| Image to k-space | [kspace.py `to_kspace`](mri_sim/kspace.py#L13) |
| k-space to image | [kspace.py `from_kspace`](mri_sim/kspace.py#L18) |
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

**The three most important functions:** `to_kspace`, `from_kspace`,
`cartesian_mask`. Everything else builds on those.
