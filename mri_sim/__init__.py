"""
MRI k-Space Reconstruction Simulator
====================================

A small package that simulates the MRI acquisition/reconstruction pipeline:

    image  --FFT-->  k-space  --sampling mask-->  undersampled k-space
                                                        |
                                                     inverse FFT
                                                        |
                                                        v
                                              reconstructed image

Modules
-------
io_utils  : loading the test image (Shepp-Logan phantom or a user file)
kspace    : forward FFT, sampling masks, masking, inverse FFT reconstruction
metrics   : PSNR / SSIM wrappers, whole-image and ROI-restricted
visualize : matplotlib figures
noise     : Stage 2 -- complex Gaussian noise in k-space
cs        : Stage 2 -- compressed sensing (FISTA)
roi       : reduced-FOV / inner-volume imaging -- scanning only the region
            that matters, via a selective RF excitation plus a coarser
            k-space grid (`FOV = 1/dk`, resolution untouched)

CSE 220 course project. Everything past `kspace`/`metrics`/`visualize` is
additive: the Stage 1 pipeline behaves identically whether or not it is used.
"""

__version__ = "1.0.0"
