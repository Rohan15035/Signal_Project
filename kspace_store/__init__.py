"""Build and read a curated library of MRI k-space samples.

mri_sim works on one image at a time, which proves the maths but is not enough
for a live demo -- there you want a shelf of ready-made subjects a web app can
load instantly without touching the 12 GB raw dataset.

So this reads the raw dataset once (offline, slow), converts each chosen slice
to synthetic k-space, and writes a self-contained store:

    data/kspace_store/
        manifest.json         one JSON record per sample
        samples/<id>.npz      the arrays (k-space, image, phase, ...)
        previews/<id>_*.png   thumbnails for a web UI
        README.md             generated at build time

    python -m kspace_store.build                 # write it (run once)
    from kspace_store.store import KSpaceStore   # read it

store.py needs only numpy and the standard library, so the app can read the
store without pydicom/nibabel/h5py installed.
"""

from __future__ import annotations

__all__ = ["store", "prepare", "sources", "catalog"]
