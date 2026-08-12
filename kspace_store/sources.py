"""
sources.py -- readers for the four raw formats in the dataset.

The dataset ships four completely different file formats. Each reader below
hides one of them behind the same tiny contract:

    reader(...) -> (image, metadata, extra_arrays)

        image   : 2-D float array, raw values, NOT yet normalised or resized
                  (prepare.to_store_image does that afterwards)
        metadata: dict of provenance / acquisition facts worth showing in the UI
        extra   : dict of any additional aligned 2-D arrays (currently only the
                  tumour mask, when the source provides one)

Keeping the readers this dumb means `catalog.py` can mix and match sources
freely, and `build.py` never has to know what a DICOM is.

These readers are only needed at *build* time. Reading the finished store
needs nothing but numpy.
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# 1. BrainTumorDataPublic -- MATLAB v7.3 files (which are really HDF5)
# ---------------------------------------------------------------------------

# The published label encoding for this dataset.
BRAIN_TUMOR_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary tumor"}


def read_brain_tumor_mat(path: str) -> tuple[np.ndarray, dict, dict]:
    """
    Read one slice from the brain-tumour .mat collection.

    Each file holds a MATLAB struct called `cjdata` with:
        image       512x512 int16   -- contrast-enhanced T1-weighted brain MRI
        label       1x1             -- 1 meningioma, 2 glioma, 3 pituitary
        tumorMask   512x512 uint8   -- expert-drawn tumour region
        tumorBorder 1xN             -- the mask outline as x,y pairs
        PID                         -- anonymised patient id

    MATLAB stores arrays column-major and h5py reads them row-major, so every
    array comes out transposed relative to MATLAB. We transpose back so the
    image and its mask are oriented the way the dataset authors intended.
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
    """
    Cheap probe used by the catalogue: return (label, tumour area in pixels)
    without loading the image itself.

    Used to skip cases where the tumour is a handful of pixels -- those make a
    poor demo, because you cannot see whether undersampling destroyed the
    lesion or not.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["cjdata"]
        label = int(np.array(data["label"]).ravel()[0])
        area = int(np.array(data["tumorMask"]).sum())
    return label, area


# ---------------------------------------------------------------------------
# 2. NINS_Dataset -- ordinary JPEG images, one folder per diagnosis
# ---------------------------------------------------------------------------


def read_nins_jpeg(
    path: str,
    diagnosis: str,
    plane: str = "unspecified",
) -> tuple[np.ndarray, dict, dict]:
    """
    Read one 320x320 JPEG brain slice.

    Caveat worth being upfront about: these are JPEG-compressed, so they have
    already lost some high-frequency content and carry faint 8x8 block
    artifacts. For a teaching demo that is fine -- and it is itself a nice
    talking point, since JPEG throws away high frequencies for exactly the
    same reason that MRI undersampling does.
    """
    from PIL import Image

    with Image.open(path) as handle:
        # The files are RGB containers holding identical channels; "L"
        # collapses them to a single luminance channel.
        image = np.asarray(handle.convert("L"), dtype=np.float64)

    metadata = {
        "modality": "MRI",
        "weighting": "unspecified (screen-captured clinical slice)",
        "anatomy": "brain",
        # These folders mix axial, coronal and sagittal slices with no labels,
        # so the plane is inferred by the catalogue (see nins_slice_scores)
        # rather than read from a header. Passed in rather than guessed here.
        "plane": plane,
        "finding": diagnosis,
        "native_shape": list(image.shape),
        "caveat": "JPEG-compressed source; some high-frequency detail was "
                  "already lost before we saw it",
    }
    return image, metadata, {}


def nins_slice_scores(path: str) -> tuple[float, float]:
    """
    Two cheap scores the catalogue uses to choose a slice: (fill, symmetry).

    The folders are unlabelled mixtures of axial, coronal and sagittal slices
    from several patients, so we have to judge each candidate by its pixels.

    fill : fraction of the frame that is tissue rather than background.
        Slices from the top of the head are mostly black and make a dull demo;
        they score low.

    symmetry : correlation between the slice and its left-right mirror image.
        The head is very nearly symmetric about the midline, so an *axial* or
        *coronal* slice scores ~0.9+, while a *sagittal* slice -- which has
        the face on one side and the back of the head on the other -- scores
        far lower. This is what lets us prefer axial views, where most
        pathology in this collection is actually visible.
    """
    from PIL import Image

    with Image.open(path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.float64) / 255.0

    fill = float((image > 0.1).mean())

    mirrored = np.fliplr(image)
    # corrcoef is undefined for a constant image; guard against blank slices.
    if image.std() < 1e-8:
        return fill, 0.0
    symmetry = float(np.corrcoef(image.ravel(), mirrored.ravel())[0, 1])
    return fill, symmetry


# ---------------------------------------------------------------------------
# 3. MRI_Dataset -- real Siemens DICOM spine studies (.ima files)
# ---------------------------------------------------------------------------


def read_dicom_slice(path: str) -> tuple[np.ndarray, dict, dict]:
    """
    Read one DICOM slice, keeping the acquisition parameters.

    This source is the most valuable one educationally, because the DICOM
    header records how the scan was actually made: echo time (TE), repetition
    time (TR), field strength, pixel spacing. A short TE/TR gives T1 contrast
    (fat bright, fluid dark) and a long TE/TR gives T2 contrast (fluid bright)
    -- the *same anatomy* imaged twice, which is a great thing to put side by
    side in a demo.

    The study is already anonymised: patient name, id and birth date are blank
    in the files.
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
    """DICOM values come back as pydicom DSfloat/IS wrappers; unwrap safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weighting_from_series(series: str) -> str:
    """Infer T1/T2 weighting from the Siemens series name, e.g. 't2_tse_sag'."""
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


# ---------------------------------------------------------------------------
# 4. 3D_volumetric_imaging -- NIfTI volumes (.nii)
# ---------------------------------------------------------------------------


def read_nifti_slice(
    path: str,
    plane: str = "axial",
    index: float = 0.5,
    metadata_extra: dict | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """
    Pull a single 2-D slice out of a 3-D (or 4-D) NIfTI volume.

    Two details that are easy to get wrong:

    * ORIENTATION. NIfTI files store an affine matrix describing how voxel
      axes map onto physical axes, and different scanners write different
      orders. `as_closest_canonical` reorders the volume to RAS+ (x->Right,
      y->Anterior, z->Superior) so that "axial" means the same thing for every
      file. We then transpose and flip so the slice displays the conventional
      way up rather than mirrored or on its side.

    * 4-D VOLUMES. Some files here are vector fields (fMRI, flow), shaped
      (x, y, z, 3). Taking the Euclidean norm over the last axis turns the
      vector at each voxel into a scalar magnitude we can image.

    Parameters
    ----------
    plane : "axial" | "coronal" | "sagittal"
    index : float in [0, 1] (fractional position through the volume) or an int
            (absolute slice number)
    """
    import nibabel as nib

    volume_img = nib.as_closest_canonical(nib.load(path))
    volume = np.asanyarray(volume_img.dataobj, dtype=np.float64)

    if volume.ndim == 4:
        # Vector-valued voxels (or a trailing singleton axis) -> magnitude.
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

    # Put the slice the right way up for display: rows should run
    # superior -> inferior (or anterior -> posterior for an axial view).
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


# ---------------------------------------------------------------------------
# Small shared helper
# ---------------------------------------------------------------------------


def relative_to(path: str, root: str) -> str:
    """Path relative to the dataset root, with forward slashes.

    Stored in the manifest so every sample can be traced back to the exact
    file it came from -- provenance matters if anyone asks "where did this
    image come from?".
    """
    return os.path.relpath(path, root).replace("\\", "/")
