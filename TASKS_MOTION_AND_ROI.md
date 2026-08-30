# Implementation Tasks: Motion Simulation & Reduced-FOV ROI Imaging

Two new features. Both are **additive** — do not change anything in Stage 1 or
Stage 2. New modules, new CLI flags (off by default), new figures.

Read [FILE_GUIDE.md](FILE_GUIDE.md) first. The conventions that matter:

- Every k-space array is **centred** (`fftshift` applied, DC at `[ny//2, nx//2]`).
- Reconstruct with `mri_sim.kspace.from_kspace(k)` = `abs(ifft2(ifftshift(k)))`.
- Ground truth images are float in `[0, 1]`, so `data_range=1.0`.

All the numbers quoted below were measured on real store samples before this
document was written. Use them as targets — if your output is far off, you have
a bug.

---

# TASK A — Motion Simulation

## Why

Children move. That is the actual reason paediatric scans have to be fast, and
right now the project cannot show it. This also finally justifies radial
sampling, which currently *loses* on our SSIM table with no explanation.

## The physics, in one line

Motion is the **Fourier shift theorem**:

```
f(x - a)   <-->   F(k) * exp(-j*2*pi*k*a/N)
```

Moving the patient does not change the magnitude of k-space at all — it stamps
a **linear phase ramp** on it. If the patient moves *during* the scan, different
k-space lines get *different* ramps, and that inconsistency is the artifact.

**Rows are time.** In `cartesian_mask` each row is one phase-encode step, one
repetition of the pulse sequence. So row index `i` = when that line was
acquired. Row 0 is early in the scan, row `ny-1` is late.

## The core function

Create `mri_sim/motion.py`:

```python
def apply_motion(kspace_centered, displacements):
    """
    Corrupt k-space with patient translation during the scan.

    displacements : sequence of (dy, dx) in pixels, length ny.
                    displacements[i] = where the patient was when row i
                    was acquired.
    """
    ny, nx = kspace_centered.shape
    cy, cx = ny // 2, nx // 2
    kx = np.arange(nx) - cx          # frequency index along the row
    out = kspace_centered.copy()

    for i, (dy, dx) in enumerate(displacements):
        ky = i - cy                  # this row's frequency index (constant)
        ramp = np.exp(-2j * np.pi * (ky * dy / ny + kx * dx / nx))
        out[i, :] = kspace_centered[i, :] * ramp

    return out
```

**Verify this first.** Apply the *same* `(dy, dx)` to every row and the result
must equal `np.roll(image, (dy, dx), (0, 1))` to machine precision:

```
max|phase-ramp result - np.roll result|  ==  4.44e-16
```

If that check does not pass, nothing downstream is meaningful. Do not proceed.

## Motion models to provide

Write these as small functions returning a displacement list of length `ny`:

| Model | Displacement | Artifact it produces |
|---|---|---|
| `sudden_jerk(ny, amp, at)` | `0` before row `at`, `amp` after | Discrete ghost copies |
| `slow_drift(ny, amp)` | linear `0 -> amp` | Smearing / blur |
| `periodic(ny, amp, cycles)` | `amp * sin(2*pi*cycles*i/ny)` | Regular ghost train |

Measured on `brain-glioma-778`, fully sampled, motion only:

| Model | PSNR | SSIM |
|---|---|---|
| no motion | 322.52 | 1.000 |
| sudden jerk, 5 px at mid-scan | 22.76 | 0.798 |
| slow drift, 0 -> 6 px | 20.06 | 0.704 |
| breathing, 3 px, 8 cycles | 24.37 | 0.739 |

Note how destructive this is: a 5-pixel movement costs ~300 dB. Motion is a far
bigger problem than undersampling.

## The trap that will cost you a day

To show *why radial sampling is used for uncooperative patients*, motion has to
be applied **per spoke**, not per row. Each spoke is acquired at its own moment,
and every spoke crosses the k-space centre — so motion averages into blur
instead of building coherent ghosts.

**`_draw_spokes()` in `kspace.py:347` merges all spokes into one binary mask and
throws away which point belongs to which spoke.** You cannot do per-spoke timing
with it as written.

Add a variant that keeps them separate — do not modify the existing function:

```python
def draw_spokes_indexed(shape, n_spokes):
    """Same geometry as _draw_spokes, but returns an int array where the
    value is (spoke_index + 1) and 0 means unsampled."""
```

Then apply the ramp for spoke `s` to the points where the index equals `s + 1`,
using each point's own `(ky, kx)` rather than a whole-row `ky`.

For reference, applying motion per-*row* and then simply masking gives almost no
separation between strategies — this is the wrong result, and it is what you get
if you skip the step above:

```
breathing motion, 25% sampling, per-ROW timing (WRONG for radial):
  cartesian         PSNR 22.98   SSIM 0.677
  radial            PSNR 23.98   SSIM 0.545
  variable_density  PSNR 24.47   SSIM 0.708
```

Radial should pull clearly ahead once timing is per-spoke. If it does not, your
spoke indexing is wrong.

## Deliverables

- `mri_sim/motion.py` with `apply_motion` + the three models
- `--motion {jerk,drift,periodic}` and `--motion-amp` flags in `main.py`, off by default
- `plot_motion_comparison()` in `visualize.py`: clean | motion | difference,
  and a Cartesian-vs-radial panel under identical motion
- A tab in the Streamlit app with sliders for amplitude and model

## Done when

1. Uniform-shift check matches `np.roll` at `~1e-16`.
2. Periodic motion produces **visible discrete ghosts along the vertical
   (phase-encode) direction** — this is the textbook artifact; if your ghosts
   are horizontal, you have transposed `ky` and `kx`.
3. Radial beats Cartesian under identical motion, with per-spoke timing.

---

# TASK B — Reduced-FOV (ROI) Imaging

## Why

"We only care about the hypothalamus — can we scan just that?" The intuitive
answer is "use only part of k-space," and it is **wrong**. This task shows the
right answer, and the wrong answer is a great teaching moment.

## First, prove the misconception is a misconception

k-space is **not spatially local**. Every k-space point carries information
about every pixel. Zero out one quadrant of k-space and the error is spread
evenly across the whole image:

```
error per image quadrant:  0.0388   0.0223   0.0273   0.0334
```

Ship this as a figure. Knowing *where* the hypothalamus is tells you nothing
about *which k-space samples* to keep.

## The right answer: shrink the FOV, not the k-space region

Two separate relationships — this is the heart of the task:

| Quantity | Controls | Relationship |
|---|---|---|
| `dk` — spacing between samples | **field of view** | `FOV = 1 / dk` |
| `k_max` — how far out you go | **resolution** | `dx = 1 / (2 * k_max)` |

So: use RF pulses to excite **only a small box** around the target. The object
now fits in a small FOV, which means you can space samples `R` times further
apart *without aliasing*, while keeping the same `k_max` and therefore the same
resolution. Fewer samples, same sharpness.

In the simulator you emulate the RF excitation by **masking the image before the
forward FFT**. That is legitimate — spatially restricting excitation is exactly
what a 2D-selective pulse physically does.

## The core function

Create `mri_sim/roi.py`:

```python
def reduced_fov_acquire(image, center, R):
    """
    Simulate an inner-volume scan.
      1. excite only a box of size (N/R) around `center`
      2. sample every Rth point of k-space in both directions
    Returns (excited_object, kspace_sampled, mask).
    """
    ny, nx = image.shape
    assert ny % R == 0, "R must divide the image size exactly"   # see below
    fov = ny // R

    y0 = min(max(0, center[0] - fov // 2), ny - fov)
    x0 = min(max(0, center[1] - fov // 2), nx - fov)
    box = np.zeros_like(image)
    box[y0:y0 + fov, x0:x0 + fov] = 1.0

    excited = image * box                       # <-- the RF excitation
    k = to_kspace(excited)

    samp = np.zeros_like(image)
    samp[::R, ::R] = 1.0                        # coarse grid, FULL k_max
    return excited, k * samp, samp
```

Reconstruct with the normal `from_kspace`, then **multiply by `R * R`** for
density compensation (you kept `1/R^2` of the points).

## Two hard constraints — both verified

**1. `R` must divide `N` exactly.** Otherwise the aliasing period `N/R` is not a
whole number of pixels and the replicas do not land on the grid. On our 256px
store that means `R` in `{2, 4, 8, 16}`. Measured PSNR inside the ROI:

```
R= 2  256%R=0  divides=True    330.57 dB
R= 3  256%R=1  divides=False    38.30 dB     <-- broken
R= 4  256%R=0  divides=True    328.52 dB
R= 5  256%R=1  divides=False    33.67 dB     <-- broken
R= 6  256%R=4  divides=False    40.20 dB     <-- broken
R= 8  256%R=0  divides=True    327.48 dB
R=16  256%R=0  divides=True    334.94 dB
```

Assert it. A silent 40 dB is much worse than a crash.

**2. The RF suppression is not optional — it is the whole thing.** Without it,
everything outside the small FOV folds directly on top of your ROI:

| Setup | k-space used | Speed-up | PSNR in ROI |
|---|---|---|---|
| Excite box + coarse grid | 6.25% | **16x** | **328 dB** |
| Same 16x, ordinary undersampling | 6.25% | 16x | ~25 dB |
| Keep k-space centre instead (low-pass) | 6.25% | 16x | 23.6 dB |
| **No suppression, whole head excited** | 6.25% | 16x | **-14 dB** |

That last row is the headline. Make it a figure.

## Compact reconstruction (optional, but it is the honest version)

You only measured `(N/R)^2` samples, so you can inverse-FFT just those into a
small image — which is what a scanner actually returns:

```python
k_small = kspace_full[::R, ::R]          # 256x256 -> 64x64 at R=4
img_small = np.abs(np.fft.ifft2(np.fft.ifftshift(k_small)))
```

Verified: 64x64 output, **328.48 dB** against the ROI. The ROI appears at a
*wrapped* position inside the small image, not centred.

**Centring it** needs a linear phase ramp on k-space before decimation (the
shift theorem again, same as Task A). I did not get the sign convention working
in a quick attempt — treat it as optional polish, and always check your result
against the full-size reconstruction, which already works perfectly.

## Where to get the ROI centre

12 brain samples already ship with expert tumour segmentation:

```python
sample = KSpaceStore("data/kspace_store").load("brain-pituitary-1111")
ys, xs = np.nonzero(sample.tumor_mask)
center = (int(ys.mean()), int(xs.mean()))
```

Use `brain-pituitary-1111` for the demo — the pituitary sits directly below the
hypothalamus, so it is exactly the anatomy in the original question.

## Deliverables

- `mri_sim/roi.py` with `reduced_fov_acquire` + the non-locality demo
- `--roi` and `--roi-factor` flags in `main.py`, off by default
- ROI-restricted metrics in `metrics.py`:
  `compute_metrics_in_roi(original, recon, roi_mask)` — score only inside the
  box. A scan that nails the lesion and blurs the skull is a **good** scan;
  our current whole-image metrics call that a failure.
- Figures: the k-space non-locality proof, and the 4-row table above as a panel
- A Streamlit tab: click/select an ROI, pick `R`, see it live

## Done when

1. `R=4` on `brain-pituitary-1111` gives **> 300 dB** inside the ROI.
2. Dropping the excitation step drops it to roughly **-14 dB**.
3. Non-divisible `R` raises, rather than silently returning ~40 dB.

---

# Ground rules

- Both features are **off by default**. Stage 1 and Stage 2 output must not change.
- Do not edit `_draw_spokes`, `cartesian_mask`, `variable_density_mask`,
  `to_kspace`, or `from_kspace`. Add alongside them.
- Keep the comment density of the existing code. Every non-obvious line gets a
  "why", not a "what" — we have to explain all of this to the instructor.
- Verify against the numbers in this document as you go. They were all measured
  on real samples in this repo.

Suggested order: Task A shift-theorem check -> A models -> A figures -> B
non-locality demo -> B core -> B metrics -> Streamlit tabs last.
