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
metrics   : PSNR / SSIM wrappers
visualize : matplotlib figures

Stage 1 of the CSE 220 course project.
"""

__version__ = "1.0.0"
