"""Lognormal-mode-integrated Mie cross sections.

Reference quantities for the physics verification of the SARB mode optics
lookup tables. The layer optical depth formula in ``mode_physics`` is

    tau = sigma_table * 1e-12 * N * delp / g

where ``N`` is the *total* number mixing ratio of the mode (particles per kg)
and ``sigma_table`` is read from the SARB ``extpsw_mie`` lookup table. For that
product to equal the true mode extinction, ``sigma_table`` must be the
*number-averaged per-particle* extinction cross section of the mode,

    <sigma> = integral sigma(r) n(r) dr / integral n(r) dr ,

with ``n(r)`` the lognormal number size distribution. This module computes that
number-averaged cross section directly from homogeneous-sphere Mie theory
(``mie_sphere``), so it can be compared against the table value.

Monodisperse evaluation (the cross section of a single sphere at the median
radius) is also provided, because the table is indexed by a single radius and a
naive table would store the monodisperse value rather than the mode integral.
"""

import numpy as np

from mie_sphere import mie_cross_sections_um2


def lognormal_number_pdf(radius_um, median_radius_um, sigma_g):
    """Lognormal number distribution dN/dr (unnormalised shape, per um)."""
    radius = np.asarray(radius_um, dtype=np.float64)
    log_sigma = np.log(float(sigma_g))
    exponent = -((np.log(radius) - np.log(float(median_radius_um))) ** 2) / (2.0 * log_sigma ** 2)
    return np.exp(exponent) / (radius * log_sigma * np.sqrt(2.0 * np.pi))


def _quadrature_radii(median_radius_um, sigma_g, n_quad, n_sigma):
    log_sigma = np.log(float(sigma_g))
    lo = np.log(float(median_radius_um)) - n_sigma * log_sigma
    hi = np.log(float(median_radius_um)) + n_sigma * log_sigma
    ln_r = np.linspace(lo, hi, int(n_quad))
    radius = np.exp(ln_r)
    # weight is the lognormal pdf expressed over ln(r): exp(-(lnr-lnrg)^2/2s^2)/(s sqrt(2pi))
    weight = np.exp(-((ln_r - np.log(float(median_radius_um))) ** 2) / (2.0 * log_sigma ** 2))
    weight /= log_sigma * np.sqrt(2.0 * np.pi)
    return ln_r, radius, weight


def mode_averaged_cross_sections_um2(
    n_real,
    n_imag,
    median_radius_um,
    sigma_g,
    wavelength_um,
    n_quad=240,
    n_sigma=5.0,
):
    """Number-averaged per-particle ext/sca/abs cross sections (um^2).

    Integrates homogeneous-sphere Mie cross sections over a lognormal number
    distribution of median ``median_radius_um`` and geometric width
    ``sigma_g`` at ``wavelength_um``. The result is the cross section per
    particle that, multiplied by the total particle number, gives the mode
    extinction.
    """
    ln_r, radius, weight = _quadrature_radii(median_radius_um, sigma_g, n_quad, n_sigma)
    ext = np.empty_like(radius)
    sca = np.empty_like(radius)
    absn = np.empty_like(radius)
    for index, value in enumerate(radius):
        cross = mie_cross_sections_um2(n_real, n_imag, float(value), wavelength_um)
        ext[index] = cross["ext"]
        sca[index] = cross["sca"]
        absn[index] = cross["abs"]

    norm = np.trapezoid(weight, ln_r)
    return {
        "ext": float(np.trapezoid(ext * weight, ln_r) / norm),
        "sca": float(np.trapezoid(sca * weight, ln_r) / norm),
        "abs": float(np.trapezoid(absn * weight, ln_r) / norm),
        "number_norm": float(norm),
        "n_quad": int(n_quad),
    }


def monodisperse_cross_sections_um2(n_real, n_imag, radius_um, wavelength_um):
    """Single-sphere ext/sca/abs cross sections (um^2) at ``radius_um``."""
    cross = mie_cross_sections_um2(n_real, n_imag, radius_um, wavelength_um)
    return {"ext": cross["ext"], "sca": cross["sca"], "abs": cross["abs"]}
