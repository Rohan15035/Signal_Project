# Tab 1 and Tab 2, explained simply

A non-technical explanation of the first two tabs of the app.

---

## The idea everything is built on

**Any picture can be made by stacking striped patterns on top of each other.**

Imagine transparent sheets. Each sheet has stripes printed on it.

- Some sheets have **fat, wide stripes**, only two or three across the whole
  sheet.
- Some sheets have **very thin stripes**, hundreds of them packed tightly.

Stack enough of these sheets, each faded to the right darkness, and you can
build *any* image: a face, a brain scan, anything.

What each type of sheet contributes:

| Sheet type | What it gives the picture |
|---|---|
| **Fat stripes** | The overall shape. Where it's bright and where it's dark. |
| **Thin stripes** | The fine details. Sharp edges. Small textures. |

**k-space is the list of how dark to make each sheet.**

It is not a strange kind of image. It is one number per sheet, saying "use this
much of this stripe pattern."

It is also arranged in a useful way:

> **Middle of k-space = the fat stripes. Outside of k-space = the thin stripes.**

Middle = overall shape. Outside = fine detail. Both tabs follow from this.

---

## What the MRI machine does

An MRI machine **does not take a photograph.**

It measures the sheets one at a time. It asks "how much of *this* stripe pattern
is in the patient?", records the number, then moves to the next one.

Two facts follow:

1. **Every measurement takes time.** More sheets measured means the patient
   stays in the scanner longer. A full scan can take 40 minutes.
2. **The computer needs all the numbers** to rebuild the picture exactly.

So there is a trade-off: **skip some sheets and finish sooner.**

But when a sheet is skipped, the computer doesn't know its number, so it assumes
**zero**. Zero is wrong; the sheet just wasn't measured.

**That wrong guess is what damages the picture.** Every artifact in this project
comes from it.

---

## Tab 1: what happens when I move the slider

The slider says **"k-space sampled."** It means: **what percentage of the sheets
do we measure?**

- Slider at 100%: measure everything, perfect picture, slow scan
- Slider at 25%: measure a quarter, 4x faster scan, damaged picture

### The five panels, left to right

| Panel | What it is |
|---|---|
| 1. Ground truth | The real picture, which we are trying to get back. |
| 2. Full k-space | All the sheet numbers, if we measured everything. |
| 3. The mask | **Which sheets we chose to measure.** White = measured, black = skipped. |
| 4. What the scanner got | The numbers we actually have. The skipped ones are now zero. |
| 5. Reconstruction | The picture rebuilt from only those numbers. |

### When you drag the slider

**Panels 1 and 2 don't change.** Same patient, same data.

**Panels 3, 4 and 5 change.** The mask gets emptier, the scanner gets less, and
the picture gets worse.

> You can say: *"I'm not changing the patient. I'm only changing how much of
> them we measured."*

### The three sampling strategies

The same number of sheets is skipped each time, but chosen in three different
ways, and the picture breaks in three different ways.

**Cartesian: skip in a regular pattern**
Measure every 2nd sheet, or every 4th.

Result: **ghost copies.** Faint duplicates of the brain appear, evenly spaced,
overlapping the real one.

*Why:* when the missing sheets follow a regular pattern, the errors line up and
add up into something visible. A repeating gap creates a repeating copy.

This is the **worst** kind of damage, because the ghosts look like real anatomy
and could be mistaken for something real.

**Radial: measure in a star pattern**
Like spokes on a bicycle wheel, all passing through the middle.

Result: **streaks** coming out from bright edges.

*Why:* every spoke crosses the middle, so the fat stripes get measured many
times. Near the outside the spokes spread apart and leave gaps, and those gaps
show up as streaks.

**Random (variable density): skip randomly, but protect the middle**
Pick sheets at random, but make the middle very likely to be picked and the
outside unlikely.

Result: **faint grain**, like TV static. The brain is still clearly readable.

*Why:* random errors don't line up with each other. They spread into a thin haze
across the picture instead of forming a copy.

In our tests this strategy gives the best PSNR at every slider position.

> In one line: *"Faint static is easy to ignore. A fake copy of the anatomy is
> not. So random skipping beats regular skipping."*

---

## Tab 2: why the middle matters

Tab 1 showed *that* the strategies differ. Tab 2 shows *why*.

Every strategy in Tab 1 protects the middle of k-space. This tab shows the
reason for that choice.

### The experiment

Two pictures, side by side:

- **Left:** keep only the **middle** sheets and throw the outside away.
- **Right:** keep only the **outside** sheets and throw the middle away.

**Both sides keep the same number of sheets.** At the 10% setting it's 6,557
versus 6,549.

So if the two pictures look very different, the difference is not about *how
many* sheets were kept, but *which ones*.

### What you see

**Left, middle only (the fat stripes):**
The brain is there, with the right shape and brightness, but it is **blurry**,
as if out of focus. The detail is gone, but the image is still recognisable.

**Right, outside only (the thin stripes):**
The brain is **gone**. The picture is almost black. Stretch the contrast and a
faint **outline** appears: just the edges.

### Why the right side is black

The centre point of k-space is the sheet with **no stripes at all**, a flat grey
sheet. That single number is the **average brightness of the whole image**.

Throwing away the middle throws that number away, so the average brightness of
the picture becomes zero and it looks black.

### The numbers

At the 10% setting, on the phantom:

| | Sheets kept | Share of the picture's energy |
|---|---|---|
| Middle only | 10% | **95%** |
| Outside only | 10% | **0.13%** |

**Same number of sheets, about 700x difference in how much energy they carry.**

This is why every sampling strategy in the project protects the middle.

### When you drag this slider

**Drag right:** the middle circle grows, so the left picture gets sharper (more
detail allowed in). The right picture shows more structure but stays dark.

**Drag left:** the left picture gets blurrier. The right picture becomes almost
completely black.

### Why there are three small pictures on each side

**Mask | As reconstructed | Contrast stretched**

The third one is needed because the "outside only" picture really is almost
black. On a normal brightness scale it looks like an empty frame. Stretching the
contrast shows the outline that is in it.

> Say: *"The normal version shows the brightness is gone. The stretched version
> shows the edges are still there. Either one alone would be misleading."*

### One more thing you might be asked

The faint rings around sharp edges in the "middle only" picture are called
**Gibbs ringing.**

They appear because the sheets are cut off abruptly: everything inside the
circle is kept and nothing outside. Cutting a signal off sharply always causes
some overshoot and ripple at edges. It is the same effect as with the 1-D
Fourier series from lectures, in 2-D.

---

## Summary

> **Tab 1:** An MRI measures striped patterns, not pixels. Skipping some makes
> the scan faster but forces the computer to guess zero, and *which* ones are
> skipped decides whether you get ghosts, streaks, or mild static.

> **Tab 2:** The middle of k-space holds the shape and the outside holds the
> detail. The middle holds 95% of the energy in only 10% of the samples, which
> is why every good sampling pattern protects it.
