"""Loading the image that stands in for the thing being scanned.

A real scanner measures k-space directly; we start from a normal grayscale
image and forward-FFT it (see kspace.py) to manufacture synthetic k-space.

The default is the Shepp-Logan phantom: piecewise-constant with sharp
elliptical edges, so undersampling artifacts are easy to see.
"""

from __future__ import annotations

import numpy as np
from skimage import color, data, img_as_float, io, transform


def load_phantom(size: int | None = None) -> np.ndarray:
    """Shepp-Logan phantom as float64 in [0, 1], optionally resized to size x size.

    Native size is 400x400; 256 is noticeably faster and changes no conclusions.
    """
    image = data.shepp_logan_phantom()
    image = img_as_float(image)
    return _postprocess(image, size)


def load_image_file(path: str, size: int | None = None) -> np.ndarray:
    """Load any image file as float64 grayscale in [0, 1] -- the bring-your-own entry point."""
    image = io.imread(path)

    # Drop alpha if present (RGBA -> RGB), then RGB -> gray.
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    # Rescales integer dtypes (uint8 0..255) into 0..1.
    image = img_as_float(image)
    return _postprocess(image, size)


def load_image(path: str | None = None, size: int | None = None) -> np.ndarray:
    """Dispatcher for main.py: phantom if path is None, else the file at path."""
    if path is None:
        return load_phantom(size=size)
    return load_image_file(path, size=size)


def _postprocess(image: np.ndarray, size: int | None) -> np.ndarray:
    """Optional square resize, then rescale to [0, 1].

    Forcing every input onto [0, 1] keeps PSNR numbers comparable across images.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")

    if size is not None:
        # anti_aliasing low-pass filters first, so we do not introduce aliasing
        # before the experiment has even started.
        image = transform.resize(
            image, (size, size), anti_aliasing=True, preserve_range=True
        )

    image = image.astype(np.float64)

    lo, hi = float(image.min()), float(image.max())
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)

    return image
