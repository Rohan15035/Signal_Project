"""
noise.py -- Stage 2: simulating scanner noise, in the right place.

THE KEY IDEA: NOISE BELONGS IN K-SPACE, NOT IN THE IMAGE
--------------------------------------------------------
It is tempting to simulate a noisy scan by adding grain to the reconstructed
image. That is wrong, and the difference matters for this project.

What a scanner actually digitises is a voltage induced in a receive coil. That
voltage carries thermal (Johnson) noise from the patient and the electronics,
and it is added to the signal *at the moment of measurement* -- that is, to the
k-space samples. Two consequences that image-domain noise cannot reproduce:

1. The noise is **complex**: the receiver demodulates into two channels (real
   and imaginary, "I and Q"), each picking up its own independent Gaussian
   noise. So the right model is `n = n_real + 1j * n_imag`.

2. The noise is **white in k-space**, i.e. the same average power at every
   spatial frequency. The DC sample is huge and the high-frequency samples are
   tiny (see the 60-85 dB dynamic range in the sample store), so the *same*
   absolute noise wipes out the outer samples while barely touching the middle.
   Fine detail therefore degrades long before overall contrast does.

Because the inverse FFT is linear, white complex Gaussian noise in k-space
becomes white complex Gaussian noise in the image -- and then the magnitude
operation turns it into Rician-distributed noise, which is why the background
of a real MRI image is never truly black but a faint grey haze. All of that
falls out automatically here; we do not have to model it separately.

INTERACTION WITH UNDERSAMPLING (the point of the Stage 2 demo)
-------------------------------------------------------------
Noise is added only to the points the scanner actually *measures*. An
unmeasured point is not a noisy zero -- it is simply absent. So a mask that
keeps 25% of k-space also admits 25% of the noise energy, and the total noise
in the reconstruction falls as the square root of the number of samples,
exactly as `SNR ~ sqrt(N)` in real MRI.

This creates the honest trade-off students usually miss: undersampling reduces
the noise you let in, but it also removes signal and adds artifacts. Faster
scans are not simply "noisier" -- they are noisier *per unit of signal*, which
is a different and more interesting statement.
"""

from __future__ import annotations

import numpy as np


def noise_sigma_for_snr(kspace_centered: np.ndarray, snr_db: float) -> float:
    """
    Per-channel noise standard deviation that yields a target k-space SNR.

    SNR is defined here on the k-space samples, in the usual power sense:

        SNR_dB = 10 * log10( mean(|K|^2) / noise_power )

    The noise is complex with independent real and imaginary parts, each of
    standard deviation sigma, so its total power is E[|n|^2] = 2 * sigma^2.
    Solving for sigma:

        sigma = sqrt( mean(|K|^2) / (2 * 10^(SNR_dB / 10)) )

    Parameters
    ----------
    kspace_centered : complex array
        The clean, fully-sampled k-space. Its mean power sets the scale, so
        the same `snr_db` means the same thing for a bright image and a dim
        one.
    snr_db : float
        Target signal-to-noise ratio in decibels. Roughly: 40 dB is a clean
        clinical scan, 20 dB is visibly grainy, 10 dB is a bad day.

    Returns
    -------
    float : sigma, the per-channel standard deviation.
    """
    signal_power = float(np.mean(np.abs(kspace_centered) ** 2))
    if signal_power <= 0.0:
        return 0.0
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(noise_power / 2.0))


def complex_gaussian_noise(
    shape: tuple[int, int],
    sigma: float,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Draw complex white Gaussian noise: independent N(0, sigma^2) in the real
    and imaginary parts.

    This is the "circularly symmetric complex Gaussian" of communications
    theory, and it is the standard model for MRI receiver noise.
    """
    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, sigma, size=shape)
    imag = rng.normal(0.0, sigma, size=shape)
    return real + 1j * imag


def add_kspace_noise(
    kspace_centered: np.ndarray,
    snr_db: float,
    mask: np.ndarray | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Add complex Gaussian noise to k-space, optionally only where it is sampled.

    Parameters
    ----------
    kspace_centered : complex array
        Clean k-space (centered convention, as everywhere in this package).
    snr_db : float
        Target SNR in decibels; see :func:`noise_sigma_for_snr`. Pass
        `float("inf")` for a noiseless scan.
    mask : array or None
        If given, noise is added only at sampled points. This is the
        physically correct choice: an unsampled point was never measured, so
        it cannot carry measurement noise. Passing None adds noise everywhere,
        which is only useful for showing what "noise on a full scan" means.
    seed : int or None
        Reproducibility. The same seed gives the same noise draw.

    Returns
    -------
    Complex array, same shape: the noisy k-space.
    """
    if not np.isfinite(snr_db):
        return kspace_centered.copy()

    sigma = noise_sigma_for_snr(kspace_centered, snr_db)
    noise = complex_gaussian_noise(kspace_centered.shape, sigma, seed=seed)

    if mask is not None:
        # Zero the noise wherever nothing was measured, so that
        # `apply_mask` afterwards cannot resurrect noise into empty k-space.
        noise = noise * (mask != 0)

    return kspace_centered + noise


def simulate_acquisition(
    kspace_centered: np.ndarray,
    mask: np.ndarray,
    snr_db: float = float("inf"),
    seed: int | None = 0,
) -> np.ndarray:
    """
    Model one accelerated, noisy scan end to end, in the physical order.

        measure the sampled points  ->  each measurement picks up noise

    Order matters. Adding noise *then* masking would be modelling a scanner
    that acquires the whole of k-space, corrupts it, and then throws most of
    it away -- which is not what an accelerated scan does, and would make the
    noise level independent of the acceleration.

    Returns
    -------
    Complex array: the acquired (masked, noisy) k-space, ready for
    `kspace.from_kspace`.
    """
    acquired = kspace_centered * mask
    return add_kspace_noise(acquired, snr_db=snr_db, mask=mask, seed=seed)


def measured_snr_db(clean_kspace: np.ndarray, noisy_kspace: np.ndarray) -> float:
    """
    Read the SNR back off a pair of k-spaces, for verifying the simulation.

    Should return approximately the `snr_db` that was requested -- a useful
    self-check to run in front of an examiner.
    """
    noise = noisy_kspace - clean_kspace
    signal_power = float(np.mean(np.abs(clean_kspace) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    if noise_power <= 0.0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)
