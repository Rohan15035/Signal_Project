"""
demo.py -- run the Stage 1 sampling experiment on a sample from the store.

    python -m kspace_store.demo                          # first sample
    python -m kspace_store.demo --sample nins-glioma     # a specific one
    python -m kspace_store.demo --list                   # what's available
    python -m kspace_store.demo --sample spine-0001-t2-sagittal --ratios 0.5 0.25

This exists to prove one point: the store drops straight into the Stage 1
simulator. It builds the same three masks from `mri_sim.kspace`, but instead
of starting from the Shepp-Logan phantom it starts from real k-space read out
of the store -- no forward FFT needed at demo time, because the store already
holds the k-space.

It also prints, per sample, the number that motivates the whole "protect the
centre" design: what fraction of the total k-space energy lives in the middle.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from mri_sim.kspace import build_mask, sampling_ratio
from mri_sim.metrics import compute_metrics
from mri_sim.visualize import plot_reconstruction_panel

from .store import KSpaceStore

MASK_KINDS = ["cartesian", "radial", "variable_density"]
DEFAULT_RATIOS = [0.5, 0.25, 0.125]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct a store sample at several undersampling ratios."
    )
    parser.add_argument("--store", default=os.path.join("data", "kspace_store"))
    parser.add_argument("--sample", default=None, help="sample id (default: the first)")
    parser.add_argument("--list", action="store_true", help="list samples and exit")
    parser.add_argument("--ratios", type=float, nargs="*", default=DEFAULT_RATIOS)
    parser.add_argument("--masks", nargs="*", default=MASK_KINDS, choices=MASK_KINDS)
    parser.add_argument("--outdir", default=os.path.join("outputs", "store_demo"))
    args = parser.parse_args(argv)

    store = KSpaceStore(args.store)

    if args.list:
        for record in store.records():
            print(f"{record['id']:34s} {record['collection']:22s} {record['title']}")
        return 0

    sample = store.load(args.sample or store.ids()[0])
    stats = sample.meta["stats"]

    print(f"\n{sample.title}   [{sample.id}]")
    print(f"  source     : {sample.meta['source_file']}")
    print(f"  k-space    : {sample.kspace.shape} {sample.kspace.dtype}, centered")
    print(f"  energy in the central 10% radius : "
          f"{stats['energy_within_r0.1'] * 100:.1f}%  "
          f"(that is ~1% of the samples)")
    print(f"  k-space dynamic range            : "
          f"{stats['kspace_dynamic_range_db']:.0f} dB\n")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"  {'mask':18s} {'target':>7s} {'actual':>7s} {'accel':>6s} "
          f"{'PSNR':>7s} {'SSIM':>6s}")
    for kind in args.masks:
        for ratio in args.ratios:
            mask = build_mask(kind, sample.shape, ratio)
            reconstruction = sample.reconstruct(mask)
            metrics = compute_metrics(sample.image, reconstruction)
            achieved = sampling_ratio(mask)

            print(f"  {kind:18s} {ratio:7.3f} {achieved:7.3f} "
                  f"{1 / achieved:5.1f}x {metrics['psnr']:7.2f} {metrics['ssim']:6.3f}")

            plot_reconstruction_panel(
                original=sample.image,
                kspace_masked=sample.kspace * mask,
                reconstruction=reconstruction,
                mask_kind=kind,
                target_ratio=ratio,
                mask=mask,
                metrics=metrics,
                save_path=os.path.join(
                    args.outdir, f"{sample.id}_{kind}_r{int(ratio * 1000):04d}.png"
                ),
            )

    print(f"\nfigures written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
