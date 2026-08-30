"""MRI k-Space Reconstruction Simulator (CSE 220).

    image --FFT--> k-space --mask--> undersampled --iFFT--> reconstruction

io_utils loads the test image, kspace does the transforms and sampling masks,
metrics scores the result, visualize draws the figures.
"""

__version__ = "1.0.0"
