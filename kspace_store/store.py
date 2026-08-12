"""
store.py -- read the built k-space store.

This is the half of the package a web app or notebook uses. It deliberately
depends on **numpy and the standard library only**: no h5py, pydicom, nibabel
or pillow. Those are needed to *build* the store, never to read it, so a
deployed app can ship with a tiny dependency list.

Typical use::

    from kspace_store.store import KSpaceStore

    store = KSpaceStore("data/kspace_store")

    for record in store.records():                 # metadata only, no arrays
        print(record["id"], record["title"])

    sample = store.load("spine-0001-t2-sagittal")  # arrays, lazily read
    sample.kspace          # (N, N) complex64, centered (DC in the middle)
    sample.image           # (N, N) float32 ground truth in [0, 1]
    sample.reconstruct(mask)   # zero-fill an arbitrary mask and inverse FFT

Everything is cached in memory after first read, so dragging a slider in a
web app re-reads nothing from disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """
    One entry of the store: its metadata plus its arrays.

    Attributes
    ----------
    id : str
    meta : dict
        The manifest record: title, collection, tags, source_file,
        acquisition parameters, derived stats.
    kspace : complex64 array
        Centered, fully-sampled k-space. This is the "measurement" a
        simulated scan draws from.
    image : float32 array
        Ground-truth magnitude image in [0, 1]; the reference for PSNR/SSIM.
    phase : float32 array
        The synthetic phase map used when generating the k-space.
    tumor_mask : uint8 array or None
        Expert tumour segmentation, where the source dataset provided one.
    """

    id: str
    meta: dict
    kspace: np.ndarray
    image: np.ndarray
    phase: np.ndarray
    tumor_mask: np.ndarray | None = field(default=None)

    # -- convenience ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.kspace.shape

    @property
    def title(self) -> str:
        return self.meta.get("title", self.id)

    def reconstruct(self, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Zero-fill reconstruction: keep the sampled points, inverse FFT, take
        the magnitude.

        This is the same three-line operation as Stage 1's
        `mri_sim.kspace.reconstruct`, repeated here so the store stays
        importable on its own.

        Parameters
        ----------
        mask : boolean array of the same shape, or None
            True where k-space was measured. None means "fully sampled",
            which returns the ground-truth image back.
        """
        kspace = self.kspace if mask is None else self.kspace * mask.astype(bool)
        # ifftshift undoes the centering, because numpy's ifft2 expects DC in
        # the corner. Then |.| discards the phase, as a scanner console does.
        return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))

    def log_kspace(self, mask: np.ndarray | None = None) -> np.ndarray:
        """log(1 + |k|) of the (optionally masked) k-space, for display."""
        kspace = self.kspace if mask is None else self.kspace * mask.astype(bool)
        return np.log1p(np.abs(kspace))


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class KSpaceStore:
    """
    Read-only view of a built store directory.

    Parameters
    ----------
    root : str
        Directory containing `manifest.json`, `samples/` and `previews/`.
    """

    def __init__(self, root: str = os.path.join("data", "kspace_store")):
        self.root = root
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"no manifest at {manifest_path}. Build the store first:\n"
                f"    python -m kspace_store.build"
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self._by_id = {record["id"]: record for record in self.manifest["samples"]}
        # Bind the cache to the instance so two stores do not share entries.
        self._load_cached = lru_cache(maxsize=None)(self._load_uncached)

    # -- metadata ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._by_id

    def ids(self) -> list[str]:
        """Every sample id, in manifest order."""
        return [record["id"] for record in self.manifest["samples"]]

    def records(self, collection: str | None = None, tag: str | None = None) -> list[dict]:
        """
        Manifest records, optionally filtered. Cheap -- no arrays are read.

        Handy for building a gallery/dropdown in a UI:

            store.records(collection="MRI_Dataset")
            store.records(tag="tumour")
        """
        records = self.manifest["samples"]
        if collection is not None:
            records = [r for r in records if r["collection"] == collection]
        if tag is not None:
            records = [r for r in records if tag in r["tags"]]
        return records

    def record(self, sample_id: str) -> dict:
        """The manifest record for one sample."""
        if sample_id not in self._by_id:
            raise KeyError(f"no sample '{sample_id}' in {self.root}")
        return self._by_id[sample_id]

    def collections(self) -> list[str]:
        """Distinct collection names, in first-seen order."""
        seen: list[str] = []
        for record in self.manifest["samples"]:
            if record["collection"] not in seen:
                seen.append(record["collection"])
        return seen

    def tags(self) -> list[str]:
        """Every distinct tag used in the store, sorted."""
        return sorted({tag for record in self.manifest["samples"] for tag in record["tags"]})

    def preview_paths(self, sample_id: str) -> dict:
        """Absolute paths to the PNG previews, for serving from a web app."""
        files = self.record(sample_id)["files"]
        return {
            "image": os.path.join(self.root, files["image_png"]),
            "kspace": os.path.join(self.root, files["kspace_png"]),
        }

    # -- arrays -----------------------------------------------------------

    def load(self, sample_id: str) -> Sample:
        """Load one sample (cached after the first call)."""
        return self._load_cached(sample_id)

    def _load_uncached(self, sample_id: str) -> Sample:
        record = self.record(sample_id)
        path = os.path.join(self.root, record["files"]["npz"])
        with np.load(path) as data:
            return Sample(
                id=sample_id,
                meta=record,
                kspace=data["kspace"],
                image=data["image"],
                phase=data["phase"],
                tumor_mask=data["tumor_mask"] if "tumor_mask" in data.files else None,
            )

    def load_all(self) -> list[Sample]:
        """Every sample. ~30 MB in memory for the default 40x256x256 store."""
        return [self.load(sample_id) for sample_id in self.ids()]

    def __repr__(self) -> str:
        return (
            f"KSpaceStore({self.root!r}, n={len(self)}, "
            f"resolution={self.manifest['resolution']})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience, for quick interactive use
# ---------------------------------------------------------------------------


def open_store(root: str = os.path.join("data", "kspace_store")) -> KSpaceStore:
    """`KSpaceStore(root)`, spelled as a function."""
    return KSpaceStore(root)
