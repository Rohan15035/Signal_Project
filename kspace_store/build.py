"""Read the raw dataset once and write the k-space sample store.

    python -m kspace_store.build                        # build everything
    python -m kspace_store.build --sources dicom        # just one collection
    python -m kspace_store.build --size 512 --out data/kspace_store_512
    python -m kspace_store.build --gallery              # + a contact sheet

The slow offline half: it touches the 12 GB dataset, needs
h5py/pillow/pydicom/nibabel, and takes a couple of minutes. The web app never
runs it -- it only reads the small store this produces.

Per slice it writes samples/<id>.npz (kspace, image, phase, and tumor_mask if
the source had one), two PNG previews, a manifest record, and a README.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np

from . import catalog, prepare, sources

# Default locations, relative to the project root.
DEFAULT_DATASET_ROOT = os.path.join("Dataset", "archive (1)")
DEFAULT_OUT = os.path.join("data", "kspace_store")

DEFAULT_PHASE_STRENGTH = 1.0

# Radii at which to report enclosed energy -- the evidence for "the centre of
# k-space carries the image".
ENERGY_RADII = [0.02, 0.05, 0.10, 0.25, 0.50]


def run_reader(name: str, args: dict):
    """Call the right reader from sources.py for a catalogue recipe."""
    if name == "brain_tumor_mat":
        return sources.read_brain_tumor_mat(**args)
    if name == "nins_jpeg":
        return sources.read_nins_jpeg(**args)
    if name == "dicom":
        return sources.read_dicom_slice(**args)
    if name == "nifti":
        return sources.read_nifti_slice(**args)
    raise ValueError(f"unknown reader '{name}'")


def process_recipe(
    recipe: dict,
    dataset_root: str,
    out_dir: str,
    size: int,
    phase_strength: float,
) -> dict:
    """Turn one catalogue recipe into files on disk plus a manifest record."""
    raw_image, meta, extra = run_reader(recipe["reader"], recipe["args"])

    # Most sources are already centred; a recipe can ask for a content-aware
    # crop instead, as the spine DICOMs do.
    crop = recipe.get("crop", "center")
    image = prepare.to_store_image(raw_image, size=size, crop=crop)

    # Seed derived from the sample id: stable across rebuilds, different per
    # sample.
    seed = _stable_seed(recipe["id"])
    kspace, phase = prepare.build_kspace(image, seed=seed, phase_strength=phase_strength)

    arrays = {
        # complex64, not complex128: half the size, and 7 digits is far more
        # than an 8-bit display can show.
        "kspace": kspace.astype(np.complex64),
        "image": image.astype(np.float32),
        "phase": phase.astype(np.float32),
    }

    tumor_mask = extra.get("tumor_mask")
    if tumor_mask is not None:
        # Reuse the box computed from the image so the mask cannot drift out of
        # alignment, then re-binarise after the smooth resize.
        box = prepare.square_crop_box(np.asarray(raw_image, dtype=np.float64), crop)
        mask = prepare.apply_crop(tumor_mask.astype(np.float64), box)
        mask = prepare.resize_square(mask, size)
        arrays["tumor_mask"] = (mask > 0.5).astype(np.uint8)

    npz_rel = f"samples/{recipe['id']}.npz"
    npz_path = os.path.join(out_dir, npz_rel)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    image_png_rel = f"previews/{recipe['id']}_image.png"
    kspace_png_rel = f"previews/{recipe['id']}_kspace.png"
    _save_gray_png(image, os.path.join(out_dir, image_png_rel))
    _save_kspace_png(kspace, os.path.join(out_dir, kspace_png_rel))

    stats = {
        f"energy_within_r{radius:g}": round(
            prepare.center_energy_fraction(kspace, radius), 6
        )
        for radius in ENERGY_RADII
    }
    stats["hermitian_asymmetry"] = round(_hermitian_asymmetry(kspace), 6)
    stats["kspace_dynamic_range_db"] = round(_dynamic_range_db(kspace), 2)
    if "tumor_mask" in arrays:
        stats["tumor_area_px"] = int(arrays["tumor_mask"].sum())

    return {
        "id": recipe["id"],
        "title": recipe["title"],
        "collection": recipe["collection"],
        "collection_note": recipe["collection_note"],
        "tags": recipe["tags"],
        "source_file": sources.relative_to(recipe["args"]["path"], dataset_root),
        "shape": [int(size), int(size)],
        "crop": crop,
        # How this slice was chosen, when a rule rather than a hard-coded path
        # made the choice.
        "selection": recipe.get("selection"),
        "acquisition": meta,
        "phase": {
            "model": "synthetic smooth phase (B0 bowl + linear ramp + coil ripples)",
            "seed": seed,
            "strength": phase_strength,
        },
        "stats": stats,
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in arrays.items()
        },
        "files": {
            "npz": npz_rel,
            "image_png": image_png_rel,
            "kspace_png": kspace_png_rel,
        },
    }


def _stable_seed(sample_id: str) -> int:
    """Deterministic 32-bit FNV-1a seed from a sample id.

    Python's hash() is randomised per process (PYTHONHASHSEED), which would
    give a different phase map on every rebuild.
    """
    hash_value = 2166136261
    for byte in sample_id.encode("utf-8"):
        hash_value = ((hash_value ^ byte) * 16777619) & 0xFFFFFFFF
    return int(hash_value)


def _hermitian_asymmetry(kspace: np.ndarray) -> float:
    """How far this k-space is from conjugate symmetry, in [0, ~1.4].

    A real image gives K(-k) = conj(K(k)) and would score ~0, making half the
    data redundant. The synthetic phase breaks that, as real scanner data does;
    reporting the number keeps us honest about what is being simulated.
    """
    # Mirror on the un-shifted array, where index -k is just (N - k) mod N.
    uncentered = np.fft.ifftshift(kspace)
    mirrored = uncentered[
        np.ix_(
            (-np.arange(uncentered.shape[0])) % uncentered.shape[0],
            (-np.arange(uncentered.shape[1])) % uncentered.shape[1],
        )
    ]
    numerator = np.linalg.norm(uncentered - np.conj(mirrored))
    denominator = np.linalg.norm(uncentered)
    return float(numerator / denominator) if denominator else 0.0


def _dynamic_range_db(kspace: np.ndarray) -> float:
    """Peak-to-median k-space magnitude in dB, typically 60-80 for a medical image.

    The DC term dwarfs everything else, which is why k-space is always shown on
    a log scale.
    """
    magnitude = np.abs(kspace)
    median = float(np.median(magnitude))
    peak = float(magnitude.max())
    if median <= 0 or peak <= 0:
        return 0.0
    return 20.0 * np.log10(peak / median)


def _save_gray_png(image01: np.ndarray, path: str) -> None:
    """Save a [0, 1] float image as an 8-bit grayscale PNG."""
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.clip(image01, 0.0, 1.0)
    Image.fromarray((data * 255).round().astype(np.uint8), mode="L").save(path)


def _save_kspace_png(kspace: np.ndarray, path: str) -> None:
    """Save centered k-space as a false-colour log-magnitude PNG.

    Without the log it is one white pixel on black. The colour map is cosmetic
    -- it makes the faint high-frequency structure at the edges visible.
    """
    from matplotlib import colormaps
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)

    log_magnitude = np.log1p(np.abs(kspace))
    peak = float(log_magnitude.max())
    normalized = log_magnitude / peak if peak > 0 else log_magnitude

    rgba = colormaps["inferno"](normalized)             # -> (ny, nx, 4) floats
    rgb = (rgba[..., :3] * 255).round().astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_gallery(out_dir: str, records: list[dict], path: str) -> str:
    """Write a single contact sheet showing every sample image in the store."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    n_cols = 8
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.25 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, record in zip(axes, records):
        with np.load(os.path.join(out_dir, record["files"]["npz"])) as data:
            axis.imshow(data["image"], cmap="gray", vmin=0, vmax=1)
        axis.set_title(record["id"], fontsize=6)
        axis.axis("off")

    for axis in axes[n:]:
        axis.axis("off")

    fig.suptitle(f"k-space sample store -- {n} samples", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def write_store_readme(out_dir: str, manifest: dict) -> None:
    """Document the store next to the store, so it can travel on its own."""
    collections = {}
    for record in manifest["samples"]:
        collections.setdefault(record["collection"], []).append(record)

    lines = [
        "# MRI k-space sample store",
        "",
        f"Generated by `python -m kspace_store.build` on {manifest['built_at']}.",
        f"{manifest['n_samples']} samples at "
        f"{manifest['resolution']}x{manifest['resolution']}.",
        "",
        "## What a sample is",
        "",
        "Each sample is one 2-D slice taken from the raw dataset, normalised to",
        "[0, 1], resized, given a synthetic smooth phase map, and Fourier",
        "transformed into **centered** k-space (`fftshift` applied, so DC sits at",
        "the middle of the array).",
        "",
        "`samples/<id>.npz` contains:",
        "",
        "| array | dtype | meaning |",
        "|---|---|---|",
        "| `kspace` | complex64 | centered k-space, the fully-sampled measurement |",
        "| `image` | float32 | ground-truth magnitude image in [0, 1] |",
        "| `phase` | float32 | the synthetic phase map, radians |",
        "| `tumor_mask` | uint8 | expert tumour mask (brain-tumour samples only) |",
        "",
        "The three are consistent: `kspace == fftshift(fft2(image * exp(1j*phase)))`,",
        "so `abs(ifft2(ifftshift(kspace)))` returns `image` to float precision.",
        "",
        "## Reading it",
        "",
        "```python",
        "from kspace_store.store import KSpaceStore",
        "",
        "store = KSpaceStore('data/kspace_store')",
        "print(store.ids())                     # every sample id",
        "sample = store.load('brain-glioma-1000')",
        "sample.kspace                          # (N, N) complex64, centered",
        "sample.image                           # (N, N) float32 ground truth",
        "sample.meta['acquisition']             # scanner / diagnosis metadata",
        "```",
        "",
        "The reader needs only numpy and the standard library -- no h5py,",
        "pydicom or nibabel. Those are build-time dependencies only.",
        "",
        "## Metadata",
        "",
        "`manifest.json` holds one record per sample: title, collection,",
        "provenance path back into the raw dataset, tags, acquisition parameters",
        "(echo time, repetition time, field strength and pixel spacing for the",
        "DICOM samples; diagnosis for the pathology samples), and derived",
        "statistics:",
        "",
        "- `energy_within_r*` -- fraction of total k-space energy inside a disc of",
        "  that radius (as a fraction of the k-space radius). The reason every",
        "  sampling mask in this project protects the centre.",
        "- `hermitian_asymmetry` -- how far the data is from conjugate symmetry.",
        "  A real-valued image would score ~0; the synthetic phase pushes this up,",
        "  which is what real scanner data looks like.",
        "- `kspace_dynamic_range_db` -- peak-to-median magnitude ratio. Explains why",
        "  k-space is always displayed on a log scale.",
        "",
        "## Contents",
        "",
    ]

    for collection, records in collections.items():
        note = records[0]["collection_note"]
        lines += [
            f"### {collection} ({len(records)} samples)",
            "",
            note,
            "",
        ]
        for record in records:
            lines.append(f"- `{record['id']}` -- {record['title']}")
        lines.append("")

    lines += [
        "## Provenance and honesty notes",
        "",
        "- These are **simulated** k-space data. A real scanner measures k-space",
        "  directly; we start from reconstructed images and run a forward FFT.",
        "- The phase map is synthetic. The source files are magnitude images, so",
        "  the original scanner phase no longer exists.",
        "- The NINS samples are JPEG-compressed at source, so some high-frequency",
        "  content was already lost before this pipeline saw them.",
        "- Every sample records the exact raw file it came from in",
        "  `manifest.json -> source_file`.",
        "",
    ]

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the MRI k-space sample store from the raw dataset."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help="path to the extracted 'archive (1)' folder")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="where to write the store")
    parser.add_argument("--size", type=int, default=prepare.DEFAULT_SIZE,
                        help="store resolution (square), default 256")
    parser.add_argument("--sources", nargs="*", default=None,
                        choices=list(catalog.BUILDERS),
                        help="subset of collections to build (default: all)")
    parser.add_argument("--phase-strength", type=float, default=DEFAULT_PHASE_STRENGTH,
                        help="0 gives a real-valued image (Hermitian k-space)")
    parser.add_argument("--gallery", action="store_true",
                        help="also write a contact sheet of every sample")
    parser.add_argument("--gallery-path", default=os.path.join("outputs", "kspace_store_gallery.png"))
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset_root):
        print(f"dataset root not found: {args.dataset_root}", file=sys.stderr)
        return 1

    print(f"scanning dataset at {args.dataset_root} ...")
    recipes = catalog.build_catalog(args.dataset_root, args.sources)
    print(f"selected {len(recipes)} slices")

    os.makedirs(args.out, exist_ok=True)

    records: list[dict] = []
    for index, recipe in enumerate(recipes, start=1):
        print(f"  [{index:2d}/{len(recipes)}] {recipe['id']}")
        records.append(process_recipe(
            recipe,
            dataset_root=args.dataset_root,
            out_dir=args.out,
            size=args.size,
            phase_strength=args.phase_strength,
        ))

    manifest = {
        "schema_version": 1,
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "resolution": args.size,
        "n_samples": len(records),
        "dataset_root": args.dataset_root.replace("\\", "/"),
        "kspace_convention": (
            "centered (fftshift applied); kspace = fftshift(fft2(image * exp(1j*phase)))"
        ),
        "phase_strength": args.phase_strength,
        "energy_radii": ENERGY_RADII,
        "samples": records,
    }

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    write_store_readme(args.out, manifest)

    total_bytes = sum(
        os.path.getsize(os.path.join(args.out, record["files"]["npz"]))
        for record in records
    )
    print(f"\nwrote {len(records)} samples to {args.out}")
    print(f"  arrays : {total_bytes / 1e6:.1f} MB")
    print(f"  manifest: {manifest_path}")

    if args.gallery:
        path = save_gallery(args.out, records, args.gallery_path)
        print(f"  gallery : {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
