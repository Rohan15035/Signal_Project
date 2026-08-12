"""
main.py -- run the whole MRI k-space simulation end to end.

For every combination of (sampling strategy, undersampling ratio) it:

    1. builds the k-space sampling mask,
    2. zero-fills the unsampled k-space points,
    3. reconstructs by inverse FFT + magnitude,
    4. scores the result against the original with PSNR and SSIM,
    5. saves a four-panel comparison figure,

then writes a summary chart (PSNR/SSIM vs ratio) and a CSV of every number.

Usage
-----
    python main.py                          # Shepp-Logan phantom, defaults
    python main.py --image brain.png        # use your own grayscale image
    python main.py --size 512               # work at 512x512
    python main.py --ratios 1.0 0.5 0.25    # choose the sampling ratios
    python main.py --outdir results         # where the figures go

Run `python main.py --help` for the full list.
"""

from __future__ import annotations

import argparse
import csv
import os

from mri_sim import kspace as ks
from mri_sim import cs, io_utils, metrics, noise, visualize

# The three strategies, in the order they appear in every figure and table.
STRATEGIES = ["cartesian", "radial", "variable_density"]

# Default sampling ratios: fully sampled, then halving. 0.125 is an 8x
# accelerated scan -- aggressive enough that all three strategies visibly fail
# in their own characteristic way, which is the point of the comparison.
DEFAULT_RATIOS = [1.0, 0.5, 0.25, 0.125]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MRI k-space reconstruction simulator (Stage 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default=None,
        help="path to a grayscale image file; omit to use the Shepp-Logan phantom",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="resize the image to SIZE x SIZE (use 0 to keep the native size)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=DEFAULT_RATIOS,
        help="k-space sampling ratios to test, each in (0, 1]",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="directory for the output figures and CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the variable-density mask (reproducibility)",
    )

    # ---- Stage 2: all optional, all off by default -------------------------
    # None of these replace anything above; they add extra runs and extra
    # figures on top of the Stage 1 sweep.
    stage2 = parser.add_argument_group("Stage 2 (optional)")
    stage2.add_argument(
        "--sample",
        default=None,
        help="use a sample from the k-space store instead of an image file "
             "(e.g. brain-glioma-778); see `python -m kspace_store.demo --list`",
    )
    stage2.add_argument(
        "--store",
        default=os.path.join("data", "kspace_store"),
        help="path to the k-space store, used with --sample",
    )
    stage2.add_argument(
        "--noise-snr",
        type=float,
        default=None,
        help="add complex Gaussian noise in k-space at this SNR in dB "
             "(40 = clean scan, 20 = grainy, 10 = bad); omit for a noiseless scan",
    )
    stage2.add_argument(
        "--center-edges",
        action="store_true",
        help="also write the centre-only vs edges-only demonstration figure",
    )
    stage2.add_argument(
        "--cs",
        action="store_true",
        help="also run compressed sensing (FISTA) against zero-filling on the "
             "random mask, at the most aggressive ratio requested",
    )
    stage2.add_argument(
        "--cs-lambda",
        type=float,
        default=0.01,
        help="compressed-sensing regularisation strength",
    )
    stage2.add_argument(
        "--cs-iters",
        type=int,
        default=80,
        help="compressed-sensing iterations",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[dict]:
    """
    Execute the full experiment grid and return one result dict per run.
    """
    size = args.size if args.size and args.size > 0 else None

    # ---- Step 1: the "object being scanned" --------------------------------
    if args.sample:
        # Stage 2: take a real slice out of the k-space store. Note the k-space
        # is *read from disk*, not computed here -- the store already holds it,
        # complete with the synthetic phase that makes it non-Hermitian.
        from kspace_store.store import KSpaceStore

        sample = KSpaceStore(args.store).load(args.sample)
        image = sample.image.astype(float)
        full_kspace = sample.kspace.astype(complex)
        source = f"{args.sample} ({sample.meta['source_file']})"
    else:
        image = io_utils.load_image(args.image, size=size)
        source = args.image if args.image else "skimage.data.shepp_logan_phantom()"
        # ---- Step 2: forward FFT -> synthetic k-space ----------------------
        # Done ONCE. Every experiment below re-uses this same full k-space and
        # differs only in which of its points the "scanner" is allowed to keep.
        full_kspace = ks.to_kspace(image)

    print(f"Image  : {source}")
    print(f"Shape  : {image.shape}, range [{image.min():.3f}, {image.max():.3f}]")

    # Sanity check: reconstructing the untouched k-space must return the
    # original image to within floating-point noise. If this ever fails, the
    # fftshift/ifftshift pairing is wrong. (Store samples are stored as
    # complex64, so their round trip is limited by float32 precision, not
    # float64 -- hence the looser tolerance in that case.)
    roundtrip_error = float(abs(ks.from_kspace(full_kspace) - image).max())
    tolerance = 1e-5 if args.sample else 1e-9
    print(f"FFT round-trip max error: {roundtrip_error:.2e}")
    assert roundtrip_error < tolerance, "forward/inverse FFT round trip is broken"

    if args.noise_snr is not None:
        print(f"Noise  : complex Gaussian in k-space at {args.noise_snr:g} dB SNR, "
              f"added only to sampled points")

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Step 3-5: build masks, reconstruct, score -------------------------
    # First pass: compute everything and keep the arrays we need for plotting.
    runs = []
    for kind in STRATEGIES:
        for ratio in args.ratios:
            # The variable-density mask is the only randomised one, so it is
            # the only one that takes a seed.
            extra = {"seed": args.seed} if kind == "variable_density" else {}
            mask = ks.build_mask(kind, image.shape, ratio, **extra)

            if args.noise_snr is None:
                kspace_masked = ks.apply_mask(full_kspace, mask)
            else:
                # Stage 2: mask first, then add noise to what was measured --
                # the physical order. See mri_sim/noise.py.
                kspace_masked = noise.simulate_acquisition(
                    full_kspace, mask, snr_db=args.noise_snr, seed=args.seed
                )
            reconstruction = ks.from_kspace(kspace_masked)
            scores = metrics.compute_metrics(image, reconstruction)

            runs.append(
                {
                    "mask": kind,
                    "target_ratio": ratio,
                    "achieved_ratio": ks.sampling_ratio(mask),
                    "acceleration": ks.acceleration_factor(mask),
                    "psnr": scores["psnr"],
                    "ssim": scores["ssim"],
                    "_mask": mask,
                    "_kspace_masked": kspace_masked,
                    "_reconstruction": reconstruction,
                }
            )

    # One shared upper limit for every error-map colour bar, so a dark patch
    # in one figure means the same amount of error as a dark patch in another.
    error_vmax = max(
        float(abs(image - r["_reconstruction"]).max()) for r in runs
    )

    # ---- Step 6: figures ---------------------------------------------------
    print(f"\nWriting figures to {os.path.abspath(args.outdir)}")
    for r in runs:
        filename = f"{r['mask']}_ratio{int(round(r['target_ratio'] * 1000)):04d}.png"
        path = os.path.join(args.outdir, filename)
        visualize.plot_reconstruction_panel(
            original=image,
            kspace_masked=r["_kspace_masked"],
            reconstruction=r["_reconstruction"],
            mask_kind=r["mask"],
            target_ratio=r["target_ratio"],
            mask=r["_mask"],
            metrics={"psnr": r["psnr"], "ssim": r["ssim"]},
            save_path=path,
            error_vmax=error_vmax,
        )
        print(f"  {filename}")

    # A gallery of the three mask shapes at the most aggressive ratio tested.
    gallery_ratio = min(args.ratios)
    gallery = {
        r["mask"]: r["_mask"]
        for r in runs
        if r["target_ratio"] == gallery_ratio
    }
    visualize.plot_mask_gallery(
        gallery, gallery_ratio, os.path.join(args.outdir, "mask_gallery.png")
    )
    print("  mask_gallery.png")

    visualize.plot_metrics_summary(
        runs, os.path.join(args.outdir, "summary_metrics.png")
    )
    print("  summary_metrics.png")

    # ---- Stage 2 extras ----------------------------------------------------
    # Both are additive: they write their own figures and do not touch the
    # Stage 1 sweep above or the metrics table.

    if args.center_edges:
        # Same fraction of k-space kept in both cases, so the only difference
        # is *which* frequencies were kept.
        demo_ratio = 0.10
        center_mask = ks.build_mask("center_only", image.shape, demo_ratio)
        edges_mask = ks.build_mask("edges_only", image.shape, demo_ratio)
        visualize.plot_center_vs_edges(
            original=image,
            center_mask=center_mask,
            center_recon=ks.from_kspace(ks.apply_mask(full_kspace, center_mask)),
            edges_mask=edges_mask,
            edges_recon=ks.from_kspace(ks.apply_mask(full_kspace, edges_mask)),
            save_path=os.path.join(args.outdir, "center_vs_edges.png"),
        )
        print("  center_vs_edges.png")

    if args.cs:
        # Compressed sensing needs *incoherent* artifacts, so it is run on the
        # random variable-density mask -- see the module docstring in cs.py for
        # why it cannot help a regular Cartesian mask.
        cs_ratio = min(args.ratios)
        cs_mask = ks.build_mask("variable_density", image.shape, cs_ratio, seed=args.seed)
        if args.noise_snr is None:
            acquired = ks.apply_mask(full_kspace, cs_mask)
        else:
            acquired = noise.simulate_acquisition(
                full_kspace, cs_mask, snr_db=args.noise_snr, seed=args.seed
            )

        print(f"\nCompressed sensing at {cs_ratio * 100:g}% sampling "
              f"(lambda={args.cs_lambda}, {args.cs_iters} iterations)...")
        comparison = cs.compare_with_zero_fill(
            acquired, cs_mask, image,
            lambda_=args.cs_lambda, n_iter=args.cs_iters,
        )
        zf_scores, cs_scores = comparison["zero_fill_metrics"], comparison["cs_metrics"]
        print(f"  zero-fill : PSNR {zf_scores['psnr']:6.2f} dB   SSIM {zf_scores['ssim']:.4f}")
        print(f"  CS (FISTA): PSNR {cs_scores['psnr']:6.2f} dB   SSIM {cs_scores['ssim']:.4f}"
              f"   ({cs_scores['psnr'] - zf_scores['psnr']:+.2f} dB, "
              f"{cs_scores['ssim'] - zf_scores['ssim']:+.4f} SSIM)")

        visualize.plot_cs_comparison(
            image, comparison, cs_mask,
            os.path.join(args.outdir, "compressed_sensing.png"),
        )
        print("  compressed_sensing.png")

    return runs


def write_table(runs: list[dict], outdir: str) -> None:
    """Print the results as a text table and save the same data as CSV."""
    header = (
        f"{'strategy':<18}{'target':>8}{'achieved':>10}{'accel':>8}"
        f"{'PSNR (dB)':>12}{'SSIM':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r['mask']:<18}"
            f"{r['target_ratio'] * 100:>7.1f}%"
            f"{r['achieved_ratio'] * 100:>9.1f}%"
            f"{r['acceleration']:>7.1f}x"
            f"{r['psnr']:>12.2f}"
            f"{r['ssim']:>9.4f}"
        )

    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        # The leading-underscore keys hold big numpy arrays for plotting; they
        # do not belong in the CSV.
        fields = ["mask", "target_ratio", "achieved_ratio", "acceleration", "psnr", "ssim"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nMetrics table saved to {csv_path}")


def main() -> None:
    args = parse_args()
    for ratio in args.ratios:
        if not (0.0 < ratio <= 1.0):
            raise SystemExit(f"--ratios values must be in (0, 1]; got {ratio}")

    runs = run(args)
    write_table(runs, args.outdir)


if __name__ == "__main__":
    main()
