# Tab 1 and Tab 2, explained simply

No jargon. Read this if the other documents feel heavy.

---

## The one idea everything is built on

**Any picture can be made by stacking striped patterns on top of each other.**

Imagine transparent sheets. Each sheet has stripes printed on it.

- Some sheets have **fat, wide stripes** — only two or three stripes across the
  whole sheet.
- Some sheets have **very thin, fine stripes** — hundreds of them, packed tight.

Stack enough of these sheets, each faded to just the right darkness, and you can
build *any* image. A face. A brain scan. Anything.

Now the important part:

| Sheet type | What it gives the picture |
|---|---|
| **Fat stripes** | The overall shape. Where it's bright, where it's dark. |
| **Thin stripes** | The fine details. Sharp edges. Small textures. |

**k-space is just the list of how dark to make each sheet.**

That's it. k-space is not a weird alien image. It's a shopping list. One number
per sheet, saying "use this much of this stripe pattern."

And it's arranged in a very convenient way:

> **Middle of k-space = the fat stripes. Outside of k-space = the thin stripes.**

Middle = overall shape. Outside = fine detail. Remember that one line and both
tabs make sense.

---

## What the MRI machine actually does

An MRI machine **does not take a photograph.** It cannot.

What it does is measure the sheets, one at a time. It asks "how much of *this*
stripe pattern is in the patient?", writes the number down, then moves to the
next one.

Two facts follow:

1. **Every measurement takes time.** More sheets measured = longer the patient
   lies in the tube. A full scan can take 40 minutes.
2. **The computer needs all the numbers** to rebuild the picture perfectly.

So there's an obvious temptation: **skip some sheets. Finish sooner.**

But when you skip a sheet, the computer doesn't know what its number was. It just
assumes **zero**. And zero is wrong — you didn't measure it, that's all.

**That wrong guess is what makes the picture go bad.** Every artifact in this
whole project comes from that one lie.

---

## Tab 1 — what happens when I move the slider

The slider says **"k-space sampled."** It means: **what percentage of the sheets
do we bother to measure?**

- Slider at 100% → measure everything → perfect picture, slow scan
- Slider at 25% → measure a quarter → 4× faster scan, damaged picture

### The five panels, left to right

| Panel | What it is |
|---|---|
| 1. Ground truth | The real picture. What we're trying to get back. |
| 2. Full k-space | All the sheet numbers, if we measured everything. |
| 3. The mask | **Which sheets we chose to measure.** White = measured, black = skipped. |
| 4. What the scanner got | The numbers we actually have. The skipped ones are now zero. |
| 5. Reconstruction | The picture rebuilt from only those numbers. |

### When you drag the slider

**Panels 1 and 2 don't change.** Same patient, same real data.

**Panels 3, 4, 5 change.** The mask gets emptier, the scanner gets less, the
picture gets worse.

> Good line to say: *"I'm not changing the patient. I'm only changing how much of
> them we bothered to measure."*

### The three sampling strategies

Same number of sheets skipped. Three different ways of choosing which. And the
picture breaks in three completely different ways.

**Cartesian — skip in a regular pattern**
Measure every 2nd sheet, or every 4th. Very orderly.

Result: **ghost copies.** You see faint duplicates of the brain, evenly spaced,
overlapping the real one.

*Why:* when your mistakes follow a regular pattern, the mistakes line up with
each other and add up into something you can see. A repeating error creates a
repeating fake.

This is the **worst** kind of damage — the ghosts look like real anatomy. A
doctor could mistake one for something real.

**Radial — measure in a star pattern**
Like spokes on a bicycle wheel, all passing through the middle.

Result: **streaks** shooting outward from bright edges.

*Why:* every spoke crosses the middle, so the fat stripes get measured over and
over — great. But out at the edges the spokes spread apart, leaving gaps. Those
gaps show up as streaks.

**Random (variable density) — skip randomly, but protect the middle**
Pick sheets at random. But make the middle almost certain to get picked, and the
outside unlikely.

Result: **faint grain**, like TV static. The brain is still clearly readable.

*Why:* random mistakes don't line up with each other. They scatter into a thin
haze across the whole picture instead of piling into a fake copy.

**This one wins.** It wins at every slider position. That's the main result of the
project.

> The point in one line: *"Faint static is easy to ignore. A fake copy of the
> anatomy is not. So random skipping beats orderly skipping."*

---

## Tab 2 — proving the middle matters

Tab 1 showed *that* the strategies differ. Tab 2 shows *why*.

Every strategy in Tab 1 deliberately protects the middle of k-space. This tab
proves that isn't arbitrary.

### The experiment

Two pictures, side by side:

- **Left:** keep only the **middle** sheets. Throw the outside away.
- **Right:** keep only the **outside** sheets. Throw the middle away.

**Both sides keep exactly the same number of sheets.** I checked — at the 10%
setting it's 6,557 versus 6,549. Basically identical.

So if the two pictures look wildly different, it can't be about *how many*
sheets. It has to be about *which ones*.

### What you see

**Left — middle only (the fat stripes):**
The brain is there. Right shape. Right brightness. Just **blurry**, like it's out
of focus. All the detail is gone but you know exactly what you're looking at.

**Right — outside only (the thin stripes):**
The brain is **gone**. The picture is almost pure black. Stretch the contrast and
you find a faint **outline** — just the edges, floating in darkness.

### Why the right side is black

The very centre point of k-space is the sheet with **no stripes at all** — a flat
grey sheet. That single number is the **average brightness of the entire image**.

Throw away the middle, and you throw that away. The average brightness of the
picture becomes zero. So it's black. Not "darker" — *zero*.

### The number that proves it

At the 10% setting, on the phantom:

| | Sheets kept | How much of the picture's "ink" they hold |
|---|---|---|
| Middle only | 10% | **95%** |
| Outside only | 10% | **0.13%** |

**Same number of sheets. About 700× difference in how much information they
carry.**

That's the whole justification for the project's design in one table.

### When you drag this slider

**Drag right** → the middle circle grows → left picture gets sharper (more detail
allowed in). Right picture picks up more structure but stays dark.

**Drag left** → left picture gets blurrier and blurrier. Right picture goes almost
completely black.

### Why there are three small pictures on each side

**Mask · As reconstructed · Contrast stretched.**

The third one exists for an honest reason. The "outside only" picture is *really*
almost black — on a normal brightness scale it looks like an empty frame, and you
can't see anything at all. Stretching the contrast reveals the outline that's
actually in there.

> Say: *"The normal version proves the brightness is gone. The stretched version
> proves the edges are still there. I show both because either one alone would be
> misleading."*

### One extra thing you might get asked

The faint rings around sharp edges in the "middle only" picture are called
**Gibbs ringing.**

It happens because we cut the sheets off abruptly — we take everything inside the
circle and nothing outside. Chopping a signal off sharply always causes a little
overshoot and ripple at edges. It's the same effect from the 1-D Fourier series
in your lectures, just in 2-D.

---

## The two-sentence summary

> **Tab 1:** An MRI measures striped patterns, not pixels. Skipping some makes
> the scan faster but forces the computer to guess zero, and *which* ones you
> skip decides whether you get ghosts, streaks, or harmless static.

> **Tab 2:** The middle of k-space holds the shape, the outside holds the detail
> — and the middle holds 95% of the information in only 10% of the samples. That
> is why every good sampling pattern protects it.
