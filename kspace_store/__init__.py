"""
kspace_store -- build and read a curated library of MRI k-space samples.

WHAT THIS IS FOR
----------------
`mri_sim` (Stage 1) simulates the MRI pipeline on a single image at a time,
usually the Shepp-Logan phantom. That is enough to prove the maths works, but
it is not enough for a live demo: for that you want a *shelf* of ready-made
subjects -- different anatomy, different pathology, different scanner contrast
-- that a web app can load instantly without touching the 12 GB raw dataset.

This package does exactly that. It reads the raw dataset once (offline, slow),
converts each chosen slice into synthetic k-space, and writes a small
self-contained store to disk:

    data/kspace_store/
        manifest.json         <- one JSON record per sample (all the metadata)
        samples/<id>.npz      <- the actual arrays (k-space, image, phase, ...)
        previews/<id>_*.png   <- ready-to-serve thumbnails for a web UI
        README.md             <- describes the format, generated at build time

Two entry points:

    python -m kspace_store.build      # writes the store (run once)
    from kspace_store.store import KSpaceStore   # reads it (used by the app)

The reader (`store.py`) depends on nothing but numpy + the standard library,
so the web app can consume the store without pydicom/nibabel/h5py installed.
"""

from __future__ import annotations

__all__ = ["store", "prepare", "sources", "catalog"]
