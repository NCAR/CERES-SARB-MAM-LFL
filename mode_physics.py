import math

import numpy as np


def lognormal_volume_factor(dry_radius_um, sigma_g):
    radius_m = float(dry_radius_um) * 1.0e-6
    log_sigma = math.log(float(sigma_g))
    return (4.0 / 3.0) * math.pi * radius_m ** 3 * math.exp(4.5 * log_sigma ** 2)


def derive_number_mixing_ratio(dry_volume_m3_per_kg, dry_radius_um, sigma_g):
    dry_volume = np.asarray(dry_volume_m3_per_kg, dtype=np.float32)
    particle_volume = lognormal_volume_factor(dry_radius_um, sigma_g)
    return np.divide(
        dry_volume,
        particle_volume,
        out=np.zeros_like(dry_volume, dtype=np.float32),
        where=dry_volume > 0.0,
    ).astype(np.float32)


def _q_arrays(q):
    if not q:
        raise ValueError("q must contain at least one species")
    return {species: np.asarray(values, dtype=np.float32) for species, values in q.items()}


def _dry_volume(species_info, q):
    shape = next(iter(q.values())).shape
    total = np.zeros(shape, dtype=np.float32)
    for species, values in q.items():
        rho_kg_m3 = float(species_info[species]["density"]) * 1000.0
        total += values / rho_kg_m3
    return total.astype(np.float32)


def _mixed_hygroscopicity(species_info, q, rh):
    numerator = np.zeros_like(rh, dtype=np.float32)
    denominator = np.zeros_like(rh, dtype=np.float32)
    for species, values in q.items():
        rho_kg_m3 = float(species_info[species]["density"]) * 1000.0
        b0, b1, b2 = species_info[species]["hygroscopicity"]
        coeff = float(b0) + float(b1) * rh + float(b2) * rh ** 2
        volume = values / rho_kg_m3
        numerator += volume * coeff
        denominator += volume
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(rh, dtype=np.float32),
        where=denominator > 0.0,
    ).astype(np.float32)


def kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh, temperature):
    dry_radius_m = float(dry_radius_um) * 1.0e-6
    if not np.isfinite(dry_radius_m) or dry_radius_m <= 0.0:
        raise ValueError("dry_radius_um must be positive")

    rh_arr, b_arr, temp_arr = np.broadcast_arrays(
        np.asarray(rh, dtype=np.float64),
        np.asarray(hygroscopicity, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    wet = np.full(rh_arr.shape, dry_radius_m, dtype=np.float64)

    finite = np.isfinite(rh_arr) & np.isfinite(b_arr) & np.isfinite(temp_arr)
    active = finite & (rh_arr > 0.0) & (b_arr > 0.0)
    if not np.any(active):
        return (wet * 1.0e6).astype(np.float32)

    rh_safe = np.clip(np.where(np.isfinite(rh_arr), rh_arr, 0.0), 1.0e-12, 1.0 - 1.0e-6)
    b_safe = np.maximum(np.where(np.isfinite(b_arr), b_arr, 0.0), 0.0)
    temp_safe = np.clip(np.where(np.isfinite(temp_arr), temp_arr, 280.0), 180.0, 330.0)
    log_rh = np.log(rh_safe)

    molecular_weight_water = 18.016
    surface_tension = 0.076
    density_water = 1000.0
    gas_constant = 8.3143e3
    kelvin_a = (
        2.0
        * molecular_weight_water
        * surface_tension
        / (gas_constant * density_water * temp_safe)
    )

    low = np.full(rh_arr.shape, dry_radius_m * (1.0 + 1.0e-8), dtype=np.float64)
    neg_log_rh = np.maximum(-log_rh, 1.0e-12)
    high_factor = np.maximum(2.0, 2.0 * np.cbrt(1.0 + b_safe / neg_log_rh))
    high = dry_radius_m * high_factor

    for _ in range(80):
        mid = 0.5 * (low + high)
        denominator = mid ** 3 - dry_radius_m ** 3
        residual = kelvin_a / mid - b_safe * dry_radius_m ** 3 / denominator - log_rh
        high = np.where(active & (residual > 0.0), mid, high)
        low = np.where(active & (residual <= 0.0), mid, low)

    wet = np.where(active, 0.5 * (low + high), wet)
    wet = np.where(np.isfinite(wet), wet, dry_radius_m)
    return (wet * 1.0e6).astype(np.float32)


def mix_mode_state(species_info, q, refractive, rh, temperature, dry_radius_um):
    q = _q_arrays(q)
    rh_array = np.asarray(rh, dtype=np.float32)
    temperature_array = np.asarray(temperature, dtype=np.float32)

    dry_volume = _dry_volume(species_info, q)
    hygroscopicity = _mixed_hygroscopicity(species_info, q, rh_array)
    r_w_um = kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh_array, temperature_array)

    n_re_num = np.zeros_like(dry_volume, dtype=np.float32)
    n_im_num = np.zeros_like(dry_volume, dtype=np.float32)
    denominator = np.zeros_like(dry_volume, dtype=np.float32)
    for species, values in q.items():
        rho_kg_m3 = float(species_info[species]["density"]) * 1000.0
        volume = values / rho_kg_m3
        denominator += volume
        n_re_num += volume * float(refractive[species][0])
        n_im_num += volume * float(refractive[species][1])

    n_re = np.divide(
        n_re_num,
        denominator,
        out=np.ones_like(dry_volume, dtype=np.float32),
        where=denominator > 0.0,
    )
    n_im = np.divide(
        n_im_num,
        denominator,
        out=np.zeros_like(dry_volume, dtype=np.float32),
        where=denominator > 0.0,
    )

    return {
        "dry_volume": dry_volume.astype(np.float32),
        "hygroscopicity": hygroscopicity.astype(np.float32),
        "r_w_um": r_w_um.astype(np.float32),
        "n_re": n_re.astype(np.float32),
        "n_im": n_im.astype(np.float32),
    }
