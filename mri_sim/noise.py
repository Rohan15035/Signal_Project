"""Stage 2: scanner noise, added in k-space rather than in the image.

A scanner digitises a coil voltage, so noise lands on the k-space samples at
the moment of measurement. Two things follow that image-domain noise cannot
reproduce: the noise is complex (the receiver has independent I and Q
channels), and it is white in k-space, so the same absolute noise swamps the
tiny outer samples while barely touching the huge DC one -- fine detail
degrades long before contrast does.

Magnitude of complex Gaussian noise is Rician, which is why the background of
a real MRI is a faint grey haze rather than black. That falls out for free.

Noise is only added where the mask actually measures, so a 25% mask admits 25%
of the noise energy (SNR ~ sqrt(N), as in real MRI). Undersampling therefore
lets in less noise but also less signal -- faster scans are noisier per unit
of signal, not simply noisier.
"""

from __future__ import annotations

import numpy as np


def noise_sigma_for_snr(kspace_centered: np.ndarray, snr_db: float) -> float:
    """Per-channel sigma hitting a target k-space SNR.

    SNR_dB = 10*log10(mean(|K|^2) / noise_power), and complex noise has power
    2*sigma^2, so sigma = sqrt(mean(|K|^2) / (2 * 10^(SNR_dB/10))).

    Scaling by the mean power makes a given snr_db mean the same thing for a
    bright image and a dim one. Roughly: 40 dB clean, 20 dB grainy, 10 dB bad.
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
    """Circularly symmetric complex Gaussian noise -- the standard MRI receiver model."""
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
    """Add complex Gaussian noise to k-space; snr_db of inf means a noiseless scan.

    With a mask, noise lands only on sampled points -- an unsampled point was
    never measured, so it cannot carry measurement noise. mask=None adds noise
    everywhere, which only shows what noise on a full scan looks like.
    """
    if not np.isfinite(snr_db):
        return kspace_centered.copy()

    sigma = noise_sigma_for_snr(kspace_centered, snr_db)
    noise = complex_gaussian_noise(kspace_centered.shape, sigma, seed=seed)

    if mask is not None:
        # Stops a later apply_mask from resurrecting noise into empty k-space.
        noise = noise * (mask != 0)

    return kspace_centered + noise


def simulate_acquisition(
    kspace_centered: np.ndarray,
    mask: np.ndarray,
    snr_db: float = float("inf"),
    seed: int | None = 0,
) -> np.ndarray:
    """One accelerated noisy scan: measure the sampled points, then pick up noise.

    Order matters. Noising first and masking after would model a scanner that
    acquires all of k-space, corrupts it, then throws most of it away -- which
    would make the noise level independent of the acceleration.
    """
    acquired = kspace_centered * mask
    return add_kspace_noise(acquired, snr_db=snr_db, mask=mask, seed=seed)


def measured_snr_db(clean_kspace: np.ndarray, noisy_kspace: np.ndarray) -> float:
    """Read SNR back off a pair of k-spaces -- a self-check to run for an examiner."""
    noise = noisy_kspace - clean_kspace
    signal_power = float(np.mean(np.abs(clean_kspace) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    if noise_power <= 0.0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)
