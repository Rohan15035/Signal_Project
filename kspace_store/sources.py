"""Readers for the four raw formats in the dataset.

Each reader hides one format behind the same contract:

    reader(...) -> (image, metadata, extra_arrays)

image is a raw 2-D float array, not yet normalised or resized (prepare.py does
that); metadata is provenance worth showing in the UI; extra holds any aligned
arrays, currently just the tumour mask. Keeping them this dumb lets catalog.py
mix sources freely and keeps build.py from ever knowing what a DICOM is.

Only needed at build time -- reading the finished store needs only numpy.
"""

from __future__ import annotations

import os

import numpy as np

# The published label encoding for the brain-tumour collection.
BRAIN_TUMOR_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary tumor"}


def read_brain_tumor_mat(path: str) -> tuple[np.ndarray, dict, dict]:
    """Read one slice from the brain-tumour .mat collection (MATLAB v7.3 = HDF5).

    Each file holds a `cjdata` struct: image (512x512 int16, contrast-enhanced
    T1), label (1 meningioma, 2 glioma, 3 pituitary), tumorMask, tumorBorder,
    and an anonymised PID.

    MATLAB is column-major and h5py reads row-major, so everything arrives
    transposed; we transpose back to the intended orientation.
    """
    import h5py                                    # build-time dependency only

    with h5py.File(path, "r") as handle:
        data = handle["cjdata"]
        image = np.array(data["image"], dtype=np.float64).T
        mask = np.array(data["tumorMask"], dtype=np.uint8).T
        label = int(np.array(data["label"]).ravel()[0])
        # PID is stored as uint16 character codes, not a string.
        pid = "".join(chr(c) for c in np.array(data["PID"]).ravel())

    metadata = {
        "modality": "MRI",
        "weighting": "T1-weighted, contrast-enhanced",
        "anatomy": "brain",
        "plane": "axial / coronal / sagittal (varies by case)",
        "finding": BRAIN_TUMOR_LABELS.get(label, f"label {label}"),
        "label_code": label,
        "patient_id": pid,
        "native_shape": list(image.shape),
    }
    return image, metadata, {"tumor_mask": mask}


def brain_tumor_mask_area(path: str) -> tuple[int, int]:
    """Probe returning (label, tumour area in pixels) without loading the image.

    Lets the catalogue skip tiny lesions, where you cannot tell whether
    undersampling destroyed the tumour or not.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["cjdata"]
        label = int(np.array(data["label"]).ravel()[0])
        area = int(np.array(data["tumorMask"]).sum())
    return label, area


def read_nins_jpeg(
    path: str,
    diagnosis: str,
    plane: str = "unspecified",
) -> tuple[np.ndarray, dict, dict]:
    """Read one 320x320 JPEG brain slice.

    These are JPEG-compressed, so high-frequency content is already partly
    gone and faint 8x8 blocks remain. Fine for teaching, and a good talking
    point: JPEG discards high frequencies for the same reason undersampling
    does.
    """
    from PIL import Image

    with Image.open(path) as handle:
        # RGB containers holding identical channels; "L" collapses them.
        image = np.asarray(handle.convert("L"), dtype=np.float64)

    metadata = {
        "modality": "MRI",
        "weighting": "unspecified (screen-captured clinical slice)",
        "anatomy": "brain",
        # Folders mix planes with no labels, so the catalogue infers this (see
        # nins_slice_scores) and passes it in rather than us guessing here.
        "plane": plane,
        "finding": diagnosis,
        "native_shape": list(image.shape),
        "caveat": "JPEG-compressed source; some high-frequency detail was "
                  "already lost before we saw it",
    }
    return image, metadata, {}


def nins_slice_scores(path: str) -> tuple[float, float]:
    """Two cheap scores for picking a slice: (fill, symmetry).

    The folders are unlabelled mixtures of planes and patients, so candidates
    have to be judged by their pixels.

    fill is the fraction of the frame that is tissue -- slices from the top of
    the head are mostly black and make a dull demo.

    symmetry is the correlation with the left-right mirror. The head is nearly
    symmetric about the midline, so axial and coronal slices score ~0.9+ while
    sagittal ones (face on one side, back of head on the other) score much
    lower. That is what lets us prefer axial views.
    """
    from PIL import Image

    with Image.open(path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.float64) / 255.0

    fill = float((image > 0.1).mean())

    mirrored = np.fliplr(image)
    # corrcoef is undefined for a constant image.
    if image.std() < 1e-8:
        return fill, 0.0
    symmetry = float(np.corrcoef(image.ravel(), mirrored.ravel())[0, 1])
    return fill, symmetry


def read_dicom_slice(path: str) -> tuple[np.ndarray, dict, dict]:
    """Read one Siemens DICOM spine slice, keeping the acquisition parameters.

    Educationally the best source, because the header records how the scan was
    made: TE, TR, field strength, pixel spacing. Short TE/TR gives T1 contrast
    (fat bright, fluid dark), long TE/TR gives T2 (fluid bright) -- the same
    anatomy imaged twice, which demos well side by side.

    Already anonymised: name, id and birth date are blank in the files.
    """
    import pydicom

    dataset = pydicom.dcmread(path)
    image = dataset.pixel_array.astype(np.float64)

    def tag(name, default=None):
        value = getattr(dataset, name, default)
        return None if value is None else value

    spacing = tag("PixelSpacing")

    metadata = {
        "modality": str(tag("Modality", "MR")),
        "series": str(tag("SeriesDescription", "")),
        "anatomy": "lumbar spine",
        "plane": _plane_from_series(str(tag("SeriesDescription", ""))),
        "weighting": _weighting_from_series(str(tag("SeriesDescription", ""))),
        "echo_time_ms": _as_float(tag("EchoTime")),
        "repetition_time_ms": _as_float(tag("RepetitionTime")),
        "field_strength_T": _as_float(tag("MagneticFieldStrength")),
        "slice_thickness_mm": _as_float(tag("SliceThickness")),
        "pixel_spacing_mm": [float(v) for v in spacing] if spacing else None,
        "scanning_sequence": str(tag("ScanningSequence", "")),
        "native_shape": list(image.shape),
        "note": "real clinical DICOM, anonymised in the source dataset",
    }
    return image, metadata, {}


def _as_float(value):
    """Unwrap pydicom DSfloat/IS wrappers safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weighting_from_series(series: str) -> str:
    """Infer T1/T2 weighting from a Siemens series name like 't2_tse_sag'."""
    name = series.lower()
    if name.startswith("t1"):
        return "T1-weighted"
    if name.startswith("t2"):
        return "T2-weighted"
    return "unspecified"


def _plane_from_series(series: str) -> str:
    """Infer the imaging plane from the series name suffix."""
    name = series.lower()
    if "sag" in name:
        return "sagittal"
    if "tra" in name or "ax" in name:
        return "axial"
    if "cor" in name:
        return "coronal"
    return "unspecified"


def read_nifti_slice(
    path: str,
    plane: str = "axial",
    index: float = 0.5,
    metadata_extra: dict | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Pull one 2-D slice out of a 3-D (or 4-D) NIfTI volume.

    Orientation: NIfTI stores an affine mapping voxel axes to physical axes and
    scanners write different orders, so as_closest_canonical reorders to RAS+
    and "axial" means the same thing for every file. The rot90 afterwards puts
    the slice the conventional way up rather than mirrored or sideways.

    4-D files here are vector fields (fMRI, flow) shaped (x, y, z, 3); the norm
    over the last axis makes each voxel a scalar we can image.

    index is either a fraction of the way through the volume or an absolute
    slice number.
    """
    import nibabel as nib

    volume_img = nib.as_closest_canonical(nib.load(path))
    volume = np.asanyarray(volume_img.dataobj, dtype=np.float64)

    if volume.ndim == 4:
        volume = np.linalg.norm(volume, axis=3) if volume.shape[3] > 1 \
            else volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {volume.shape}")

    # After as_closest_canonical the axes are (R, A, S).
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    n_slices = volume.shape[axis]
    slice_index = int(index) if isinstance(index, int) else int(round(index * (n_slices - 1)))
    slice_index = int(np.clip(slice_index, 0, n_slices - 1))

    slab = np.take(volume, slice_index, axis=axis)

    image = np.rot90(slab)

    metadata = {
        "plane": plane,
        "slice_index": slice_index,
        "n_slices": int(n_slices),
        "voxel_size_mm": [round(float(z), 3) for z in volume_img.header.get_zooms()[:3]],
        "native_shape": list(image.shape),
        "source_volume_shape": [int(s) for s in volume.shape],
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return image, metadata, {}


def relative_to(path: str, root: str) -> str:
    """Path relative to the dataset root, with forward slashes.

    Stored in the manifest so every sample traces back to the exact file it
    came from.
    """
    return os.path.relpath(path, root).replace("\\", "/")
