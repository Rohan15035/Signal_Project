"""
catalog.py -- decides *which* 40 slices go into the store.

The raw dataset has ~43,000 files. A demo needs a few dozen good ones, chosen
to cover as much variety as possible:

    - three tumour types with expert masks        (BrainTumorDataPublic)
    - a spread of brain pathologies + normals     (NINS_Dataset)
    - the same spine imaged with T1 and T2        (MRI_Dataset, real DICOM)
    - non-brain anatomy and non-MRI modalities    (3D_volumetric_imaging)

Every selection rule in this file is deterministic: fixed file lists, fixed
strides, fixed tie-breakers, no randomness. Rebuilding the store on another
machine must produce the same 40 samples, otherwise the metrics in a report
would not be reproducible.

Each function returns a list of "recipes". A recipe is a plain dict telling
build.py which reader to call, with what arguments, and how to label the
result in the manifest.
"""

from __future__ import annotations

import glob
import os

from . import sources

# Folder names inside the dataset archive.
BRAIN_TUMOR_DIR = "BrainTumorDataPublic"
NINS_DIR = os.path.join("NINS_Dataset", "NINS_Dataset")
DICOM_DIR = "MRI_Dataset"
VOLUME_DIR = "3D_volumetric_imaging"


# ---------------------------------------------------------------------------
# 1. Brain tumour .mat files -- 4 cases per tumour type
# ---------------------------------------------------------------------------

# Scan every Nth file rather than all 3064: opening a file is ~2 ms, and a
# stride of 37 still visits ~80 candidates spread across the whole collection
# (the files are ordered by tumour type, so a stride samples all three).
_TUMOR_SCAN_STRIDE = 37

# Reject tiny lesions: if the tumour is a few dozen pixels you cannot judge by
# eye whether a reconstruction preserved it, which defeats the point.
_MIN_TUMOR_AREA_PX = 2000


def brain_tumor_recipes(dataset_root: str, per_class: int = 4) -> list[dict]:
    """Pick `per_class` well-sized cases of each tumour type."""
    folder = os.path.join(dataset_root, BRAIN_TUMOR_DIR)
    # Sort numerically (1.mat, 2.mat, ... not 1.mat, 10.mat, 100.mat) so the
    # stride walks the collection in the authors' original order.
    files = sorted(
        glob.glob(os.path.join(folder, "*.mat")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    chosen: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for path in files[::_TUMOR_SCAN_STRIDE]:
        if all(len(v) >= per_class for v in chosen.values()):
            break
        label, area = sources.brain_tumor_mask_area(path)
        if label not in chosen or len(chosen[label]) >= per_class:
            continue
        if area < _MIN_TUMOR_AREA_PX:
            continue

        finding = sources.BRAIN_TUMOR_LABELS[label]
        case = os.path.splitext(os.path.basename(path))[0]
        chosen[label].append({
            "id": f"brain-{finding.split()[0].lower()}-{case}",
            "title": f"Brain MRI, {finding} (case {case})",
            "reader": "brain_tumor_mat",
            "args": {"path": path},
            "collection": "BrainTumorDataPublic",
            "collection_note": (
                "3064 contrast-enhanced T1 brain slices with expert-drawn "
                "tumour masks; published by Cheng et al."
            ),
            "tags": ["brain", "tumour", finding, "has-mask"],
            "tumor_area_px": area,
        })

    return [recipe for label in (1, 2, 3) for recipe in chosen[label]]


# ---------------------------------------------------------------------------
# 2. NINS pathology JPEGs -- one representative slice per diagnosis
# ---------------------------------------------------------------------------

# Hand-picked diagnoses: a normal baseline plus conditions whose appearance is
# distinctive enough to recognise in a reconstruction (mass, bleed, big
# ventricles, diffuse white-matter change).
_NINS_CLASSES = [
    "Normal",
    "Glioma",
    "meningioma",
    "pituitary tumor",
    "Brain Tumor",
    "Stroke(infarct)",
    "Stroke (Haemorrhage)",
    "Cerebral Hemorrhage",
    "Obstructive Hydrocephalus",
    "Mid triventricular hydrocephalus",
    "Brain Atrophy",
    "White Matter Disease",
    "Malformation (Chiari I)",
    "Cerebral abscess",
]

# How many candidates to score per folder before picking the best one.
# Reading a 320x320 JPEG is ~1 ms, so 120 candidates per class is cheap.
_NINS_CANDIDATES = 120

# Above this left-right mirror correlation we call a slice axial/coronal
# rather than sagittal. Measured scores cluster around 0.90-0.98 for axial
# slices and 0.35-0.60 for sagittal ones, so the exact cut hardly matters.
_NINS_AXIAL_SYMMETRY = 0.90


def nins_recipes(dataset_root: str, classes: list[str] | None = None) -> list[dict]:
    """
    One slice per diagnosis: prefer a well-filled *axial* view.

    The folders mix planes and patients with no labels, so selection is a
    two-stage rule over the pixels themselves (see sources.nins_slice_scores):

        1. keep the candidates that look axial, i.e. left-right symmetric;
        2. among those, take the one showing the most brain.

    Stage 1 matters because the first files in each folder tend to be
    mid-sagittal head slices, which look nearly identical from one diagnosis
    to the next -- 14 samples that all look the same is a poor demo, and most
    of these conditions (tumour, infarct, hydrocephalus) are read off axial
    images anyway.

    If a folder has no axial candidate we fall back to the whole list rather
    than dropping the diagnosis. That is not a defeat: the classes it happens
    to for -- pituitary tumour, Chiari I malformation -- are precisely the
    ones a radiologist would show mid-sagittal.
    """
    folder_root = os.path.join(dataset_root, NINS_DIR)
    recipes: list[dict] = []

    for diagnosis in (classes or _NINS_CLASSES):
        folder = os.path.join(folder_root, diagnosis)
        candidates = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if not candidates:
            continue

        scored = [(sources.nins_slice_scores(p), p) for p in candidates[:_NINS_CANDIDATES]]
        axial = [item for item in scored if item[0][1] >= _NINS_AXIAL_SYMMETRY]
        pool = axial or scored

        # Sort key is (fill, path): the path breaks ties deterministically so
        # the same file wins on every machine.
        (fill, symmetry), best = max(pool, key=lambda item: (item[0][0], item[1]))
        plane = "axial (inferred from left-right symmetry)" if axial \
            else "sagittal / off-axial (no symmetric slice in this folder)"

        slug = _slugify(diagnosis)
        recipes.append({
            "id": f"nins-{slug}",
            "title": f"Brain MRI, {diagnosis}",
            "reader": "nins_jpeg",
            "args": {"path": best, "diagnosis": diagnosis, "plane": plane},
            "selection": {
                "rule": "most brain-filled axial-looking slice",
                "brain_fill": round(fill, 4),
                "mirror_symmetry": round(symmetry, 4),
                "candidates_scored": len(scored),
            },
            "collection": "NINS_Dataset",
            "collection_note": (
                "Clinical brain MRI slices grouped into 37 diagnosis folders "
                "(National Institute of Neurosciences dataset)."
            ),
            "tags": ["brain", slug],
        })

    return recipes


# ---------------------------------------------------------------------------
# 3. DICOM spine -- the same anatomy under T1 and T2 contrast
# ---------------------------------------------------------------------------

# Two patients x four series. Series folder names are consistent across
# patients even though the enclosing study folder name is not, so we glob the
# study level. Pairing T1 and T2 of the same plane is the whole point: it
# shows that "MRI contrast" is a property of the acquisition, not the anatomy.
_DICOM_PATIENTS = ["0001", "0003"]
_DICOM_SERIES = [
    ("T1_TSE_SAG_320_0003", "T1 sagittal"),
    ("T2_TSE_SAG_384_0002", "T2 sagittal"),
    ("T1_TSE_TRA_0005", "T1 axial"),
    ("T2_TSE_TRA_384_0004", "T2 axial"),
]


def dicom_recipes(dataset_root: str) -> list[dict]:
    """Middle slice of four series from each of two lumbar-spine studies."""
    recipes: list[dict] = []

    for patient in _DICOM_PATIENTS:
        for series_folder, label in _DICOM_SERIES:
            matches = glob.glob(os.path.join(
                dataset_root, DICOM_DIR, patient, "*", series_folder, "*.ima"
            ))
            if not matches:
                continue
            matches.sort()
            # The middle slice of a spine series is the one through the cord /
            # mid-vertebral body -- the anatomically interesting one.
            path = matches[len(matches) // 2]

            recipes.append({
                "id": f"spine-{patient}-{_slugify(label)}",
                "title": f"Lumbar spine, {label} (patient {patient})",
                "reader": "dicom",
                "args": {"path": path},
                # The spine sits off to one side of a mostly-empty frame in
                # these studies, so crop to the signal instead of the middle
                # of the image (see prepare.content_crop_square).
                "crop": "content",
                "collection": "MRI_Dataset",
                "collection_note": (
                    "Real anonymised Siemens 1.5 T lumbar-spine studies, "
                    "stored as DICOM with full acquisition parameters."
                ),
                "tags": ["spine", label.split()[0].lower(), "dicom", "real-scan"],
            })

    return recipes


# ---------------------------------------------------------------------------
# 4. Volumetric NIfTI -- non-brain anatomy and non-MRI modalities
# ---------------------------------------------------------------------------

# (folder, filename glob, plane, fractional slice position, extra metadata)
_VOLUMES = [
    ("Knee_MRI", "*.nii", "sagittal", 0.5, {
        "id": "knee-mri-sagittal",
        "title": "Knee MRI, sagittal slice",
        "modality": "MRI",
        "anatomy": "knee",
        "weighting": "radial gradient-echo acquisition",
        "tags": ["knee", "joint", "mri"],
    }),
    ("Abdominal_MRI", "*.nii", "axial", 0.5, {
        "id": "abdomen-mri-axial",
        "title": "Abdominal MRI, axial slice",
        "modality": "MRI",
        "anatomy": "abdomen",
        "weighting": "T2 HASTE, breath-hold",
        "tags": ["abdomen", "mri", "breath-hold"],
    }),
    ("Jaw_CT", "*.nii", "axial", 0.5, {
        "id": "jaw-ct-axial",
        "title": "Jaw CT, axial slice",
        "modality": "CT",
        "anatomy": "jaw",
        "weighting": "n/a (X-ray attenuation)",
        "tags": ["jaw", "ct", "bone", "high-contrast"],
    }),
    ("Walnut_CT_complete", "*.nii", "axial", 0.5, {
        "id": "walnut-ct-axial",
        "title": "Walnut micro-CT, axial slice",
        "modality": "CT",
        "anatomy": "walnut (non-medical test object)",
        "weighting": "n/a (X-ray attenuation)",
        "tags": ["walnut", "ct", "fine-detail", "test-object"],
    }),
    # A second plane through the jaw volume rather than a second organ: the
    # coronal view shows the sinuses and tooth roots, which is a much sharper
    # test of high-frequency detail than the axial view.
    #
    # (The Brain_FMRI and Cuboid_flow volumes in this dataset are *vector
    # fields*, not images -- their voxels are unit direction vectors, so the
    # magnitude is 1 everywhere inside the brain and the "image" is a flat
    # white blob. They are deliberately excluded.)
    ("Jaw_CT", "*.nii", "coronal", 0.45, {
        "id": "jaw-ct-coronal",
        "title": "Jaw CT, coronal slice",
        "modality": "CT",
        "anatomy": "jaw / sinuses",
        "weighting": "n/a (X-ray attenuation)",
        "tags": ["jaw", "ct", "bone", "sinus", "fine-detail"],
    }),
    ("Teapot", "*.nii", "axial", 0.5, {
        "id": "teapot-axial",
        "title": "Utah teapot phantom, axial slice",
        "modality": "synthetic",
        "anatomy": "n/a (synthetic control object)",
        "weighting": "n/a",
        "tags": ["synthetic", "control", "geometric"],
    }),
]


def volume_recipes(dataset_root: str) -> list[dict]:
    """One representative slice from each volumetric dataset."""
    recipes: list[dict] = []

    for folder, pattern, plane, position, extra in _VOLUMES:
        matches = sorted(glob.glob(
            os.path.join(dataset_root, VOLUME_DIR, folder, pattern)
        ))
        if not matches:
            continue
        path = matches[0]

        extra = dict(extra)                       # never mutate the module table
        recipe_id = extra.pop("id")
        title = extra.pop("title")
        tags = extra.pop("tags")

        recipes.append({
            "id": recipe_id,
            "title": title,
            "reader": "nifti",
            "args": {
                "path": path,
                "plane": plane,
                "index": position,
                "metadata_extra": extra,
            },
            "collection": "3D_volumetric_imaging",
            "collection_note": (
                "Assorted 3-D medical and test volumes stored as NIfTI; one "
                "2-D slice is extracted per volume."
            ),
            "tags": tags,
        })

    return recipes


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

# Which builders run for each --sources value on the command line.
BUILDERS = {
    "brain_tumor": brain_tumor_recipes,
    "nins": nins_recipes,
    "dicom": dicom_recipes,
    "volumes": volume_recipes,
}


def build_catalog(dataset_root: str, wanted: list[str] | None = None) -> list[dict]:
    """
    Assemble the full recipe list.

    Parameters
    ----------
    dataset_root : path to the extracted `archive (1)` folder
    wanted : subset of BUILDERS keys, or None for all four
    """
    recipes: list[dict] = []
    for name in (wanted or list(BUILDERS)):
        if name not in BUILDERS:
            raise ValueError(f"unknown source '{name}', expected one of {list(BUILDERS)}")
        recipes.extend(BUILDERS[name](dataset_root))
    return recipes


def _slugify(text: str) -> str:
    """Lower-case, hyphen-separated, filesystem- and URL-safe id fragment."""
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")
