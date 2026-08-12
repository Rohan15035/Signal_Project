"""
io_utils.py -- loading the image that stands in for "the thing being scanned".

In a real MRI scanner there is no starting image: the scanner directly measures
k-space samples. Here we cheat for simulation purposes -- we take a normal
grayscale image and run a forward FFT on it (see kspace.py) to manufacture
synthetic k-space data. Everything downstream of that point behaves exactly
like the real pipeline.

The default subject is the Shepp-Logan phantom, the standard synthetic test
object in medical imaging: it is piecewise-constant with sharp elliptical
boundaries, which makes undersampling artifacts very easy to see.
"""

from __future__ import annotations

import numpy as np
from skimage import color, data, img_as_float, io, transform


def load_phantom(size: int | None = None) -> np.ndarray:
    """
    Load the Shepp-Logan phantom as a float grayscale image in [0, 1].

    Parameters
    ----------
    size : int or None
        If given, the phantom is resized to (size, size). The native phantom
        from scikit-image is 400x400; shrinking it to 256 makes the whole
        pipeline noticeably faster without changing any of the conclusions.

    Returns
    -------
    2-D float64 array with values in [0, 1].
    """
    image = data.shepp_logan_phantom()          # float64, already in [0, 1]
    image = img_as_float(image)
    return _postprocess(image, size)


def load_image_file(path: str, size: int | None = None) -> np.ndarray:
    """
    Load an arbitrary image file from disk and convert it to a grayscale
    float image in [0, 1].

    Colour images are converted to luminance; images with an alpha channel
    have it dropped. This is the "swap in your own image" entry point.

    Parameters
    ----------
    path : str
        Path to any image format that scikit-image / imageio can read
        (.png, .jpg, .tif, ...).
    size : int or None
        Optional square resize, as in :func:`load_phantom`.
    """
    image = io.imread(path)

    # Drop an alpha channel if present (RGBA -> RGB), then RGB -> gray.
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    # img_as_float rescales integer dtypes (e.g. uint8 0..255) into 0..1.
    image = img_as_float(image)
    return _postprocess(image, size)


def load_image(path: str | None = None, size: int | None = None) -> np.ndarray:
    """
    Convenience dispatcher used by main.py.

    If `path` is None the Shepp-Logan phantom is used, otherwise the file at
    `path` is loaded. Either way the result is a 2-D float64 array in [0, 1].
    """
    if path is None:
        return load_phantom(size=size)
    return load_image_file(path, size=size)


def _postprocess(image: np.ndarray, size: int | None) -> np.ndarray:
    """
    Shared tail end of the loaders: optional square resize, then rescale to
    the full [0, 1] range.

    Rescaling matters because PSNR needs a known data range. By forcing every
    input image onto [0, 1] we can use data_range=1.0 everywhere and the PSNR
    numbers from different images stay comparable.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")

    if size is not None:
        # anti_aliasing=True low-pass filters before downsampling, which avoids
        # introducing aliasing *before* we have even started the experiment.
        image = transform.resize(
            image, (size, size), anti_aliasing=True, preserve_range=True
        )

    image = image.astype(np.float64)

    # Rescale to [0, 1]. Guard against a constant image (max == min).
    lo, hi = float(image.min()), float(image.max())
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)

    return image
