import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _riccati_psi(n, z):
    return z * spherical_jn(n, z)


def _riccati_psi_derivative(n, z):
    return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)


def _riccati_xi(n, z):
    return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))


def _riccati_xi_derivative(n, z):
    hankel = spherical_jn(n, z) + 1j * spherical_yn(n, z)
    hankel_derivative = spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True)
    return hankel + z * hankel_derivative


def _series_limit(x):
    if x <= 0.0 or not np.isfinite(x):
        raise ValueError("size parameter must be positive and finite")
    return max(1, int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0)))


def mie_efficiencies(n_real, n_imag, radius_um, wavelength_um):
    """Return homogeneous-sphere Mie efficiencies.

    The imaginary refractive index is treated as absorbing regardless of sign,
    matching the lookup path that compares against abs(n_imag).
    """
    radius = float(radius_um)
    wavelength = float(wavelength_um)
    if radius <= 0.0 or wavelength <= 0.0:
        raise ValueError("radius_um and wavelength_um must be positive")

    x = 2.0 * np.pi * radius / wavelength
    m = complex(float(n_real), abs(float(n_imag)))
    n_values = np.arange(1, _series_limit(x) + 1, dtype=np.float64)

    mx = m * x
    psi_x = _riccati_psi(n_values, x)
    psi_mx = _riccati_psi(n_values, mx)
    psi_x_prime = _riccati_psi_derivative(n_values, x)
    psi_mx_prime = _riccati_psi_derivative(n_values, mx)
    xi_x = _riccati_xi(n_values, x)
    xi_x_prime = _riccati_xi_derivative(n_values, x)

    a_num = m * psi_mx * psi_x_prime - psi_x * psi_mx_prime
    a_den = m * psi_mx * xi_x_prime - xi_x * psi_mx_prime
    b_num = psi_mx * psi_x_prime - m * psi_x * psi_mx_prime
    b_den = psi_mx * xi_x_prime - m * xi_x * psi_mx_prime

    a_n = a_num / a_den
    b_n = b_num / b_den
    weights = 2.0 * n_values + 1.0

    q_ext = (2.0 / x ** 2) * np.sum(weights * np.real(a_n + b_n))
    q_sca = (2.0 / x ** 2) * np.sum(weights * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2))
    q_abs = max(float(q_ext - q_sca), 0.0)

    # asymmetry parameter g (Bohren & Huffman 4.80):
    # g*q_sca = (4/x^2)[ sum_n n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1})
    #                    + sum_n (2n+1)/(n(n+1)) Re(a_n b*_n) ]
    n = n_values
    coupling = (n * (n + 2.0) / (n + 1.0))[:-1] * np.real(
        a_n[:-1] * np.conj(a_n[1:]) + b_n[:-1] * np.conj(b_n[1:])
    )
    cross = ((2.0 * n + 1.0) / (n * (n + 1.0))) * np.real(a_n * np.conj(b_n))
    g_q_sca = (4.0 / x ** 2) * (np.sum(coupling) + np.sum(cross))
    asymmetry = float(g_q_sca / q_sca) if q_sca > 0.0 else 0.0
    return {
        "q_ext": float(q_ext),
        "q_sca": float(q_sca),
        "q_abs": q_abs,
        "asymmetry": asymmetry,
        "size_parameter": float(x),
        "series_terms": int(n_values.size),
    }


def mie_cross_sections_um2(n_real, n_imag, radius_um, wavelength_um):
    efficiencies = mie_efficiencies(n_real, n_imag, radius_um, wavelength_um)
    area = np.pi * float(radius_um) ** 2
    return {
        "ext": efficiencies["q_ext"] * area,
        "sca": efficiencies["q_sca"] * area,
        "abs": efficiencies["q_abs"] * area,
        "q_ext": efficiencies["q_ext"],
        "q_sca": efficiencies["q_sca"],
        "q_abs": efficiencies["q_abs"],
        "asymmetry": efficiencies["asymmetry"],
        "size_parameter": efficiencies["size_parameter"],
        "series_terms": efficiencies["series_terms"],
    }
