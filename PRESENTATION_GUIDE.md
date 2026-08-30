# Presentation Runbook

**For: CSE 220 project defence, tomorrow.**
This is the "what to do and say" document. The physics explanation lives in
[PROJECT_BRIEF.md](PROJECT_BRIEF.md) — read that once tonight, then present from this.

---

## 0. Before you walk in — 5 minute checklist

| Check | Command | Expected |
|---|---|---|
| Free space on C: | — | **Currently 0 bytes. Fix this first.** Streamlit and matplotlib both write to `%TEMP%` on C:, and the demo can fail without it. |
| App launches | `streamlit run app/streamlit_app.py` | Opens `http://localhost:8501` |
| Store is present | `python -m kspace_store.demo --list` | 40 samples listed |
| Figures exist | `python main.py --center-edges --cs` | Writes 15 PNGs to `outputs/` |

Run `python main.py --center-edges --cs` **tonight**, not in the room. It takes a
couple of minutes and produces every figure you might want as a fallback if the
live app misbehaves. Have the `outputs/` folder open in a second window.

---

## 1. Open with one sentence

> An MRI scanner doesn't measure an image. It measures the image's 2-D Fourier
> transform, one sample at a time — and scan time is proportional to how many
> samples it collects. This project shows exactly what breaks when you speed up
> the scan by measuring less.

Then the consequence, which is the whole project:

- Scan time ∝ number of k-space samples.
- The inverse FFT needs *all* of k-space to be exact.
- So: measure less → finish sooner → get a worse image. **How much worse, and can we be clever about which samples we skip?**

---

## 2. Demo flow (aim for 8–10 minutes)

Drive the Streamlit app. Six tabs, in order — they're built as a narrative.

### Tab 1 — Acquire
Pick a brain tumour sample. Start at 100% sampling, then drag the ratio slider down.

**Say:** "At 100% the reconstruction is exact — PSNR is ~320 dB, which is just
float64 round-off, not a real measurement. Now watch what happens as I drop samples."

Switch strategies at a fixed 12.5% and let the artifacts speak:
- **Cartesian** → crisp *ghost copies* of the anatomy, evenly spaced
- **Radial** → *streaks* radiating outward
- **Variable density** → *noise-like grain*, anatomy still readable

**This is the key visual of the whole project.** Same amount of data, three
completely different failure modes.

### Tab 2 — Centre vs edges
The "why" behind everything else. Keep only the centre 10% of k-space, then only
the outer 90%.

**Say:** "Centre only — right shape, right contrast, blurry. Edges only — the
anatomy vanishes, only outlines remain. The centre is a low-pass filter, the edges
are a high-pass filter. That's why every mask I build protects the centre."

Back it with the measured number: **92.6% of the k-space energy sits inside the
central 10% radius — which is about 1% of the samples.**

### Tab 3 — Noise
Drop the SNR slider and show the grain appear.

**Say:** "The important part is *where* the noise is added — in k-space, at the
moment of measurement, not painted onto the finished image. And only on the
points we actually measured. An unsampled point isn't a noisy zero; it was never
measured at all."

### Tab 4 — Compressed sensing
The payoff. Same mask, same samples, two different reconstructions.

**Say:** "Zero-filling assumes everything we didn't measure was zero. That's a lie
— we just didn't look. Compressed sensing instead asks: of all the images
consistent with what we *did* measure, which is the simplest?"

**Real result at 12.5% sampling (8× acceleration):**

| | PSNR | SSIM |
|---|---|---|
| Zero-fill | 25.81 dB | 0.478 |
| CS (FISTA) | 27.26 dB | **0.785** |
| **Gain** | **+1.45 dB** | **+0.31** |

Point at the SSIM. "PSNR barely moved but SSIM went up by 0.31 — the structure
came back. That gap is the whole argument for quoting both metrics."

### Tab 5 — Sweep
The summary chart. One line per strategy, PSNR and SSIM vs sampling ratio.

### Tab 6 — Provenance
Show that this is real clinical data with the source file recorded per sample.
Good pre-emptive answer to "did you just make this up?"

---

## 3. Numbers to know cold

Full sweep on the Shepp-Logan phantom at 256×256 (`python main.py`):

| Strategy | 50% | 25% | 12.5% (8×) |
|---|---|---|---|
| Cartesian | 26.81 dB / 0.683 | 20.54 / 0.549 | 17.43 / 0.535 |
| Radial | 29.64 dB / 0.476 | 23.29 / 0.312 | 19.65 / 0.276 |
| **Variable density** | **31.58 dB / 0.823** | **28.86 / 0.722** | **25.81 / 0.478** |

**Variable density wins at every ratio.** That is the headline result.

### The one anomaly worth pointing out yourself

At 50%, **radial has higher PSNR than Cartesian (29.64 vs 26.81) but much lower
SSIM (0.476 vs 0.683)**. The two metrics disagree.

If you raise this before your instructor does, it looks like you understand your
own data. The explanation: PSNR is pure per-pixel error and is blind to
structure. Radial's streaks are low-energy and spread thin, so they cost little
in squared error — but they're *coherent structure* that wasn't in the original,
and SSIM punishes exactly that. This is the textbook case for why one metric
isn't enough.

### Store facts
- 40 samples, 256×256, from 4 collections (brain tumours with expert masks, clinical pathologies, real Siemens spine DICOM, volumetric CT/MRI)
- k-space dynamic range: **62–85 dB** (mean 73) — why k-space is always shown on a log scale
- Energy inside central 10% radius: **92.6% mean** (min 74%, max 99%)

### Correctness checks you can run live
```bash
python -m mri_sim.motion      # Fourier shift theorem: max error 6.66e-16
python main.py                # FFT round-trip: max error ~5.6e-16
```
Both are machine epsilon. If asked "how do you know it's right?", run these.

---

## 4. Code tour — if asked to show the implementation

Structure is a package, not a script:

| File | What to say |
|---|---|
| [mri_sim/kspace.py](mri_sim/kspace.py) | **The core.** Forward/inverse FFT + all five masks. Open this one first. |
| [mri_sim/cs.py](mri_sim/cs.py) | FISTA compressed sensing |
| [mri_sim/noise.py](mri_sim/noise.py) | Complex Gaussian noise in k-space |
| [mri_sim/motion.py](mri_sim/motion.py) | Patient motion via the Fourier shift theorem |
| [mri_sim/metrics.py](mri_sim/metrics.py) | PSNR / SSIM |
| [mri_sim/visualize.py](mri_sim/visualize.py) | All matplotlib output |
| [main.py](main.py) | Runs the whole sweep end to end |

The three functions to have ready: `to_kspace`, `from_kspace`, `cartesian_mask`.

---

## 5. Questions your instructor will probably ask

**"Why `fftshift`?"**
NumPy's `fft2` puts DC in the corner and wraps negative frequencies to the far
edges. `fftshift` moves DC to the middle, which is the MRI convention and makes
masks easy to write — everything becomes "distance from the centre". It's pure
re-indexing; no information changes.

**"Why `ifftshift` and not `fftshift` on the way back?"**
They're identical for even-sized arrays but differ for odd sizes. Using the wrong
one gives a subtly shifted image. Worth saying — it shows you read the docs.

**"Why take the magnitude?"**
The inverse FFT returns a complex image. With full data the imaginary part is
numerical noise, but once you zero out part of k-space the masked data is no
longer the transform of a real image, so there's a genuine imaginary component.
Real scanners have the same problem for physical reasons (B0 inhomogeneity, coil
phase) and solve it the same way — they display the magnitude image.

**"Why does random sampling beat regular skipping?"**
Regular skipping produces *coherent* aliasing — crisp ghost copies that are just
as sparse as the real anatomy, so no sparsity-based method can tell them apart.
Random sampling spreads the error into *incoherent* low-level grain, which a
sparsity-promoting denoiser can remove. Coherent vs incoherent is the whole
ballgame for compressed sensing.

**"How does this relate to Nyquist?"**
Skipping every Nth k-space line is undersampling in the Fourier domain, so the
aliasing appears in the *image* domain — the image folds onto itself. That's why
Cartesian artifacts are evenly spaced replicas: it's classic aliasing, just with
the two domains swapped from the usual 1-D textbook picture.

**"Is this real MRI data?"**
The images are real clinical scans (DICOM from a Siemens 1.5 T scanner, plus
public brain-tumour and pathology datasets). The **k-space is simulated** — I
start from reconstructed images and run a forward FFT, because raw scanner
k-space isn't publicly available. Be upfront about this; it's in the store's
README and the manifest records the exact source file for every sample.

**"Why is the phase synthetic?"**
Source files are magnitude images — the original scanner phase was thrown away
before they were saved. A purely real image has perfectly Hermitian k-space,
`K(-k) = conj(K(k))`, so half the data would be a free copy of the other half and
any partial-Fourier demo would be unrealistically perfect. Adding a plausible
smooth phase map (B0 bowl + linear ramp + coil ripples) removes that artificial
symmetry. Measured asymmetry is 1.23, i.e. genuinely non-Hermitian.

**"What is CS actually doing, step by step?"**
Two alternating steps, 80 iterations:
1. **Data consistency** — put the measured samples back where they belong.
2. **Sparsify** — wavelet transform, soft-threshold small coefficients to zero, transform back.

Step 1 is a gradient step on the data-fidelity term (the FFT is unitary, so the
step size is exactly 1 — no tuning). Step 2 is the proximal operator of the L1
norm. Alternating them provably converges. FISTA adds Nesterov momentum for
roughly quadratic speed-up at the same cost per iteration.

**"Why does CS only help the random mask?"**
See the coherent/incoherent answer above. On a regular Cartesian mask the ghosts
are sparse too, so sparsity can't distinguish signal from artifact.

---

## 6. If the live demo breaks

1. **Don't debug on stage.** Switch to the pre-generated PNGs in `outputs/`.
2. `summary_metrics.png` and `mask_gallery.png` alone carry most of the story.
3. If Streamlit won't start, the most likely cause is **C: being full** — which is why that's item one on the checklist.

---

## 7. Closing line

> Undersampling isn't just "less data, worse image". *Which* samples you skip
> matters more than how many, and if you skip them randomly you can reconstruct
> non-linearly and get most of the quality back. That's compressed sensing, and
> it's why a modern MRI scan takes minutes instead of an hour.
