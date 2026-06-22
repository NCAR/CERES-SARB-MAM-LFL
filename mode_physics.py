import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator


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


def _finite_array(name, values):
    array = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _nonnegative_array(name, values):
    array = _finite_array(name, values)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def layer_optical_depth(delp_pa, number_per_kg, cross_section_um2):
    delp = _nonnegative_array("delp_pa", delp_pa)
    number = _nonnegative_array("number_per_kg", number_per_kg)
    cross_section = _nonnegative_array("cross_section_um2", cross_section_um2)
    delp, number, cross_section = np.broadcast_arrays(delp, number, cross_section)
    tau = cross_section * np.float32(1.0e-12) * number * delp / np.float32(9.8)
    return tau.astype(np.float32)


def _lower_bin_indices(values, table):
    table = np.asarray(table, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    clipped = np.clip(values, table[0], table[-1])
    return (np.searchsorted(table, clipped, side="right") - 1).astype(np.int32)


def _lookup_table_values(ds_table, variable):
    required_dims = ("n_real", "n_imag", "radius")
    values = ds_table[variable]
    if len(values.dims) != len(required_dims) or set(values.dims) != set(required_dims):
        raise ValueError(f"{variable} must use dimensions {required_dims}")
    return values.transpose(*required_dims).values


def lookup_mode_optics(n_re, n_im, r_w_um, ds_table):
    n_re_table = ds_table["n_real"].values
    n_im_table = ds_table["n_imag"].values
    radius_table = ds_table["radius"].values
    ext_table = _lookup_table_values(ds_table, "ext")
    abs_table = _lookup_table_values(ds_table, "abs")
    asm_table = _lookup_table_values(ds_table, "asm")

    n_re_values, n_im_values, radius_values = np.broadcast_arrays(
        _finite_array("n_re", n_re),
        np.abs(_finite_array("n_im", n_im)),
        _finite_array("r_w_um", r_w_um),
    )
    i_re = _lower_bin_indices(n_re_values, n_re_table)
    i_im = _lower_bin_indices(n_im_values, n_im_table)
    i_radius = _lower_bin_indices(radius_values, radius_table)

    ext = ext_table[i_re, i_im, i_radius]
    abs_ = abs_table[i_re, i_im, i_radius]
    asm = asm_table[i_re, i_im, i_radius]
    return ext.astype(np.float32), abs_.astype(np.float32), asm.astype(np.float32)


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


def _kohler_wet_radius_um_exact(dry_radius_um, hygroscopicity, rh, temperature):
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


# --------------------------------------------------------------------------- #
# Köhler growth-factor lookup table (LUT-warm-started Newton solve)
#
# The Köhler equation is scale-free in the growth factor GF = r_w / r_d:
#     ln(RH) = a/GF - B/(GF**3 - 1),   a = A/r_d   (A = Kelvin coefficient).
# So GF depends only on the three dimensionless inputs (a, B, RH). We tabulate
# GF on (ln a, B, s) with s = -ln(1-RH) (which spreads the steep RH->1 growth),
# look it up as a warm start, then run a few Newton steps on the exact equation
# so the result matches the bisection to machine precision and stays monotone.
# --------------------------------------------------------------------------- #
# Kelvin coefficient A (metres) = _KELVIN_A_NUM / T, with radius in metres.
_KELVIN_A_NUM = 2.0 * 18.016 * 0.076 / (8.3143e3 * 1000.0)
_GF_LUT = None  # cache: (interp, log_a_min, log_a_max, b_max, s_max)


def _koehler_gf_solve(a, hygroscopicity, log_rh, iters=100):
    """Vectorised nondimensional Köhler bisection for GF = r_w/r_d.

    Solves ``ln RH = a/GF - B/(GF**3 - 1)`` (same equation, same bracketing as
    ``_kohler_wet_radius_um_exact``). Used to build the LUT; GF=1 where inactive.
    """
    a, b, log_rh = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(hygroscopicity, dtype=np.float64),
        np.asarray(log_rh, dtype=np.float64),
    )
    gf = np.ones(a.shape, dtype=np.float64)
    active = (b > 0.0) & (log_rh < 0.0)
    if not np.any(active):
        return gf
    neg_log_rh = np.maximum(-log_rh, 1.0e-12)
    low = np.full(a.shape, 1.0 + 1.0e-9)
    high = np.maximum(2.0, 2.0 * np.cbrt(1.0 + b / neg_log_rh))
    for _ in range(iters):
        mid = 0.5 * (low + high)
        residual = a / mid - b / (mid ** 3 - 1.0) - log_rh
        high = np.where(active & (residual > 0.0), mid, high)
        low = np.where(active & (residual <= 0.0), mid, low)
    return np.where(active, 0.5 * (low + high), 1.0)


def _koehler_gf_lut(n_a=48, n_b=64, n_s=160):
    """Build (once) and cache a trilinear ln(GF) table over (ln a, B, s)."""
    global _GF_LUT
    if _GF_LUT is not None:
        return _GF_LUT
    a_grid = np.geomspace(1.0e-4, 1.0, n_a)
    b_grid = np.concatenate([[0.0], np.geomspace(1.0e-3, 8.0, n_b - 1)])
    s_grid = np.linspace(0.0, 14.0, n_s)  # s=-ln(1-RH); RH up to 1-8.3e-7
    aa, bb, ss = np.meshgrid(a_grid, b_grid, s_grid, indexing="ij")
    rh = 1.0 - np.exp(-ss)
    log_rh = np.log(np.clip(rh, 1.0e-12, 1.0 - 1.0e-6))
    gf = _koehler_gf_solve(aa, bb, log_rh)
    interp = RegularGridInterpolator(
        (np.log(a_grid), b_grid, s_grid), np.log(gf),
        method="linear", bounds_error=False, fill_value=None)
    _GF_LUT = (interp, float(np.log(a_grid[0])), float(np.log(a_grid[-1])),
               float(b_grid[-1]), float(s_grid[-1]))
    return _GF_LUT


def _kohler_wet_radius_um_lut(dry_radius_um, hygroscopicity, rh, temperature,
                              newton=4):
    """Köhler wet radius via a LUT warm start + Newton polish on the exact
    equation. Matches the bisection to machine precision while replacing 80
    iterations with one table lookup plus ``newton`` Newton steps."""
    r_d_um = float(dry_radius_um)
    if not np.isfinite(r_d_um) or r_d_um <= 0.0:
        raise ValueError("dry_radius_um must be positive")
    interp, log_a_min, log_a_max, b_max, s_max = _koehler_gf_lut()

    rh_arr, b_arr, t_arr = np.broadcast_arrays(
        np.asarray(rh, dtype=np.float64),
        np.asarray(hygroscopicity, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    rh_safe = np.clip(np.where(np.isfinite(rh_arr), rh_arr, 0.0), 0.0, 1.0 - 1.0e-6)
    b_safe = np.maximum(np.where(np.isfinite(b_arr), b_arr, 0.0), 0.0)
    t_safe = np.clip(np.where(np.isfinite(t_arr), t_arr, 280.0), 180.0, 330.0)
    active = (b_safe > 0.0) & (rh_safe > 0.0)

    log_rh = np.log(np.clip(rh_safe, 1.0e-12, 1.0 - 1.0e-6))
    a = _KELVIN_A_NUM / (t_safe * r_d_um * 1.0e-6)
    s = -np.log(np.maximum(1.0 - rh_safe, 1.0e-12))

    pts = np.column_stack([
        np.clip(np.log(a), log_a_min, log_a_max).ravel(),
        np.clip(b_safe, 0.0, b_max).ravel(),
        np.clip(s, 0.0, s_max).ravel(),
    ])
    gf = np.maximum(np.exp(interp(pts)).reshape(rh_safe.shape), 1.0 + 1.0e-12)

    # Newton polish on the singularity-free form
    #     g(GF) = (a/GF - ln RH)(GF^3 - 1) - B,
    # which is smooth and monotone even as GF -> 1 (where the raw Köhler
    # residual a/GF - B/(GF^3-1) - ln RH is stiff). One step from GF=1 already
    # lands at the small-growth limit 1 + B/(3(a - ln RH)).
    for _ in range(int(newton)):
        g3 = gf ** 3 - 1.0
        term = a / gf - log_rh
        g = term * g3 - b_safe
        gp = -a / gf ** 2 * g3 + term * 3.0 * gf ** 2
        step = np.where(active & (gp > 0.0), g / gp, 0.0)
        gf = np.maximum(gf - step, 1.0 + 1.0e-12)

    gf = np.where(active, gf, 1.0)
    return (r_d_um * gf).astype(np.float32)


def kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh, temperature,
                         method="lut"):
    """Köhler equilibrium wet radius (µm).

    ``method='lut'`` (default, production) uses the cached growth-factor table
    with Newton polish; ``method='exact'`` uses the reference bisection.
    """
    if method == "exact":
        return _kohler_wet_radius_um_exact(dry_radius_um, hygroscopicity, rh, temperature)
    return _kohler_wet_radius_um_lut(dry_radius_um, hygroscopicity, rh, temperature)


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

    if "WAT" in refractive:
        growth = np.maximum((r_w_um / float(dry_radius_um)) ** 3 - 1.0, 0.0).astype(np.float32)
        water_volume = dry_volume * growth
        denominator += water_volume
        n_re_num += water_volume * float(refractive["WAT"][0])
        n_im_num += water_volume * float(refractive["WAT"][1])

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
