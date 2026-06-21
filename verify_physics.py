"""Parameter-space physics verification for MAM internal mixing and optics.

Unlike the production scripts, which process full 3-D aerosol fields, this
module verifies each stage of the internal-mixing / radiative-property chain as
a *function on parameter space*: it builds parameter grids (relative humidity,
composition, refractive index, radius, ...), evaluates the production function,
and compares against an independent reference (analytic identity, conservation
law, equation residual, or an independent Mie calculation).

Each check yields a ``Result`` with a status:

    PASS    matches the reference within tolerance
    FAIL    exceeds tolerance (a bug or inconsistency)
    WARN    skipped (missing data) or borderline
    FINDING a quantified physics result that is not a simple pass/fail
            (e.g. the LUT-vs-Mie cross-section gap, the sigma_g mismatch)

Run all stages::

    python verify_physics.py

Run selected stages::

    python verify_physics.py --stage E,G

The implementation under test is the native-grid path (``mode_physics`` /
``mode_optics``); ``microphysics`` is used as an independent cross-check.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from mode_config import load_config, resolved_allocations
from mode_physics import (
    derive_number_mixing_ratio,
    kohler_wet_radius_um,
    layer_optical_depth,
    lognormal_volume_factor,
    lookup_mode_optics,
    mix_mode_state,
)
from mie_lognormal import mode_averaged_cross_sections_um2
from mie_sphere import mie_cross_sections_um2, mie_efficiencies

RNG = np.random.default_rng(20100101)
G = 9.8


def _simplified_wet_radius_um(r_d_um, B, rh):
    """microphysics.wet_radius with the Kelvin term dropped (A ~ 0), inlined to
    avoid the numba dependency: r_w = r_d (1 - B/ln RH)^(1/3)."""
    log_rh = np.log(np.clip(np.asarray(rh, dtype=np.float64), 0.0, 0.99))
    return r_d_um * (1.0 - np.asarray(B, dtype=np.float64) / log_rh) ** (1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    stage: str
    check: str
    n: int
    metric: str
    value: float
    tol: float
    status: str
    note: str = ""


def _passfail(value, tol):
    return "PASS" if (np.isfinite(value) and value <= tol) else "FAIL"


def _rel_err(test, ref):
    test = np.asarray(test, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    denom = np.maximum(np.abs(ref), 1.0e-30)
    return np.abs(test - ref) / denom


def _max_finite(array):
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.max()) if array.size else float("nan")


def _config():
    for name in ("aerosol.yaml", "aerosol_ceres.yaml"):
        if os.path.exists(name):
            return load_config(name)
    raise FileNotFoundError("no aerosol config found")


def _path(template):
    return os.path.expandvars(str(template))


# --------------------------------------------------------------------------- #
# Stage A: mass -> mode allocation
# --------------------------------------------------------------------------- #
def stage_a(config):
    results = []
    scheme = config["Schemes"]["MAM4"]

    raw = scheme.get("allocations", {})
    max_dev = 0.0
    for weights in raw.values():
        max_dev = max(max_dev, abs(sum(float(v) for v in weights.values()) - 1.0))
    results.append(
        Result("A", "fixed allocations sum to 1", len(raw), "max|sum-1|", max_dev, 1e-6,
               _passfail(max_dev, 1e-6), "raw YAML weights before normalisation")
    )

    allocations = resolved_allocations(config, "MAM4")
    max_dev = max(abs(sum(w.values()) - 1.0) for w in allocations.values())
    results.append(
        Result("A", "resolved allocations sum to 1", len(allocations), "max|sum-1|", max_dev, 1e-6,
               _passfail(max_dev, 1e-6), "mass conserved across modes (each species fully distributed)")
    )

    # size-bin allocations vs an INDEPENDENT lognormal-pdf nearest-mode reference
    # (recomputed here, not via allocate_size_bins_to_modes, to avoid a circular check)
    # only lognormal modes are size-bin allocation targets; skip monodisperse
    # (sigma_g=1) bin-modes for dust/sea salt
    mode_specs = {m: s for m, s in scheme["modes"].items() if float(s["sigma_g"]) > 1.0}
    max_dev = 0.0
    nbins = 0
    for group in scheme.get("size_bins", {}).values():
        for species, radius in zip(group["species"], group["radii_um"]):
            nbins += 1
            raw = {}
            for mode, spec in mode_specs.items():
                ls = np.log(float(spec["sigma_g"]))
                r = max(float(radius), 1e-12)
                raw[mode] = (np.exp(-((np.log(r) - np.log(float(spec["dry_radius_um"]))) ** 2) / (2.0 * ls ** 2))
                             / (r * ls * np.sqrt(2.0 * np.pi)))
            total = sum(raw.values())
            ref = {m: v / total for m, v in raw.items()}
            for mode in mode_specs:
                max_dev = max(max_dev, abs(allocations[species].get(mode, 0.0) - ref.get(mode, 0.0)))
    results.append(
        Result("A", "size-bin split vs independent lognormal", nbins, "max|delta|", max_dev, 1e-6,
               _passfail(max_dev, 1e-6), "weights = lognormal pdf at bin radius, normalised")
    )
    return results


# --------------------------------------------------------------------------- #
# Stage B: dry-volume internal mixing
# --------------------------------------------------------------------------- #
def _synthetic_species_info():
    return {
        "SU": {"density": 1.7, "hygroscopicity": [2.42848, -3.85261, 1.88159]},
        "BC": {"density": 1.0, "hygroscopicity": [0.01, 0.0, 0.0]},
        "POM": {"density": 1.8, "hygroscopicity": [0.14, 0.0, 0.0]},
        "DU": {"density": 2.6, "hygroscopicity": [0.14, 0.0, 0.0]},
        "SS": {"density": 2.2, "hygroscopicity": [4.83257, -6.92329, 3.27805]},
    }


def _random_q(species, n):
    return {s: RNG.uniform(0.0, 1e-8, size=n).astype(np.float32) for s in species}


def stage_b(config):
    results = []
    info = _synthetic_species_info()
    species = list(info)
    n = 4000
    q = _random_q(species, n)
    rh = RNG.uniform(0.0, 0.99, size=n).astype(np.float32)
    temp = RNG.uniform(230.0, 310.0, size=n).astype(np.float32)
    refractive = {s: (1.5, 0.0) for s in species}

    state = mix_mode_state(info, q, refractive, rh, temp, dry_radius_um=0.055)
    reference = np.zeros(n, dtype=np.float64)
    for s in species:
        reference += q[s].astype(np.float64) / (info[s]["density"] * 1000.0)
    err = _max_finite(_rel_err(state["dry_volume"], reference))
    results.append(
        Result("B", "dry volume = sum q/(rho*1000)", n, "max rel err", err, 1e-5,
               _passfail(err, 1e-5), "kg/kg / (g/cm3 * 1000) -> m3/kg")
    )

    # additivity: V(2q) = 2 V(q)
    q2 = {s: (2.0 * q[s]).astype(np.float32) for s in species}
    state2 = mix_mode_state(info, q2, refractive, rh, temp, dry_radius_um=0.055)
    err = _max_finite(_rel_err(state2["dry_volume"], 2.0 * state["dry_volume"].astype(np.float64)))
    results.append(
        Result("B", "dry volume additive/linear", n, "max rel err", err, 1e-5,
               _passfail(err, 1e-5), "V(2q)=2V(q)")
    )
    return results


# --------------------------------------------------------------------------- #
# Stage C: number concentration + lognormal geometry + sigma_g mismatch
# --------------------------------------------------------------------------- #
def stage_c(config):
    results = []

    # analytic particle volume factor
    cases = [(0.055, 1.8), (0.012, 1.6), (0.40, 1.8), (0.05, 1.6)]
    max_err = 0.0
    for r_d, sg in cases:
        ref = (4.0 / 3.0) * np.pi * (r_d * 1e-6) ** 3 * np.exp(4.5 * np.log(sg) ** 2)
        max_err = max(max_err, abs(lognormal_volume_factor(r_d, sg) - ref) / ref)
    results.append(
        Result("C", "lognormal volume factor analytic", len(cases), "max rel err", max_err, 1e-6,
               _passfail(max_err, 1e-6), "v=(4/3)pi r^3 exp(4.5 ln^2 sigma)")
    )

    # N * v_particle == V_dry, and N == 0 where V == 0
    n = 5000
    v = RNG.uniform(0.0, 1e-9, size=n).astype(np.float32)
    v[:: 50] = 0.0
    r_d, sg = 0.055, 1.8
    number = derive_number_mixing_ratio(v, r_d, sg)
    vp = lognormal_volume_factor(r_d, sg)
    recovered = number.astype(np.float64) * vp
    pos = v > 0.0
    err = _max_finite(_rel_err(recovered[pos], v[pos]))
    results.append(
        Result("C", "number * v_particle = dry volume", int(pos.sum()), "max rel err", err, 1e-5,
               _passfail(err, 1e-5), "N = V/v_particle")
    )
    zero_ok = float(np.max(np.abs(number[~pos]))) if np.any(~pos) else 0.0
    results.append(
        Result("C", "number = 0 where dry volume = 0", int((~pos).sum()), "max|N|", zero_ok, 0.0,
               _passfail(zero_ok, 0.0), "no spurious particles")
    )

    # sigma_g mismatch: number derivation uses config sigma_g; LUT metadata uses table sigma_g
    scheme = config["Schemes"]["MAM4"]["modes"]
    for mode in ("a1", "a2", "a3", "a4"):
        spec = scheme[mode]
        table_path = _path(spec["filename_sarb"])
        if not os.path.exists(table_path):
            results.append(Result("C", f"sigma_g consistency {mode}", 0, "tau ratio", float("nan"),
                                  0.0, "WARN", "table missing"))
            continue
        with xr.open_dataset(table_path) as ds:
            table_sg = float(ds["sigmag"])
        cfg_sg = float(spec["sigma_g"])
        # tau ~ N ~ 1/v_particle ~ exp(-4.5 ln^2 sigma); ratio of tau(config)/tau(table)
        tau_ratio = np.exp(4.5 * (np.log(table_sg) ** 2 - np.log(cfg_sg) ** 2))
        status = "FINDING" if abs(cfg_sg - table_sg) > 1e-3 else "PASS"
        results.append(
            Result("C", f"sigma_g consistency {mode}", 1, "tau(cfg)/tau(table)", float(tau_ratio),
                   1.0, status, f"config sigma_g={cfg_sg} vs table sigma_g={table_sg}")
        )
    return results


# --------------------------------------------------------------------------- #
# Stage D: mixed hygroscopicity
# --------------------------------------------------------------------------- #
def stage_d(config):
    results = []
    info = _synthetic_species_info()
    species = list(info)
    n = 6000
    q = _random_q(species, n)
    rh = RNG.uniform(0.0, 1.0, size=n).astype(np.float32)
    temp = np.full(n, 280.0, dtype=np.float32)
    refractive = {s: (1.5, 0.0) for s in species}

    state = mix_mode_state(info, q, refractive, rh, temp, dry_radius_um=0.055)

    # manual volume-weighted reference
    num = np.zeros(n, dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    comp = {}
    for s in species:
        b0, b1, b2 = info[s]["hygroscopicity"]
        coeff = b0 + b1 * rh.astype(np.float64) + b2 * rh.astype(np.float64) ** 2
        vol = q[s].astype(np.float64) / (info[s]["density"] * 1000.0)
        num += vol * coeff
        den += vol
        comp[s] = coeff
    reference = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    err = _max_finite(_rel_err(state["hygroscopicity"][den > 0], reference[den > 0]))
    results.append(
        Result("D", "mixed kappa = vol-weighted b(RH)", int((den > 0).sum()), "max rel err", err, 1e-4,
               _passfail(err, 1e-4), "B = sum V_j b_j(RH) / sum V_j")
    )

    # boundedness: mixture within component envelope
    comp_stack = np.vstack([comp[s] for s in species])
    lo = comp_stack.min(axis=0)
    hi = comp_stack.max(axis=0)
    mix = state["hygroscopicity"].astype(np.float64)
    viol = np.maximum(lo - mix, mix - hi)
    worst = _max_finite(viol[den > 0])
    results.append(
        Result("D", "mixture bounded by components", int((den > 0).sum()), "max overshoot", worst, 1e-5,
               _passfail(worst, 1e-5), "convex combination of b_j(RH)")
    )

    # positivity of soluble b(RH) over RH in [0,1]
    rh_line = np.linspace(0.0, 1.0, 101)
    min_b = np.inf
    for s in ("SU", "SS"):
        b0, b1, b2 = info[s]["hygroscopicity"]
        min_b = min(min_b, float((b0 + b1 * rh_line + b2 * rh_line ** 2).min()))
    results.append(
        Result("D", "soluble b(RH) stays positive", 2, "max(-b(RH),0)", max(0.0, -min_b), 0.0,
               _passfail(max(0.0, -min_b), 0.0), "SU,SS quadratics over RH in [0,1]")
    )
    return results


# --------------------------------------------------------------------------- #
# Stage E: Koehler wet radius
# --------------------------------------------------------------------------- #
def _kohler_residual(r_w_um, r_d_um, B, rh, temperature):
    r_w = np.asarray(r_w_um, dtype=np.float64) * 1e-6
    r_d = float(r_d_um) * 1e-6
    A = 2.0 * 18.016 * 0.076 / (8.3143e3 * 1000.0 * np.asarray(temperature, dtype=np.float64))
    denom = r_w ** 3 - r_d ** 3
    rhs = A / r_w - np.asarray(B, dtype=np.float64) * r_d ** 3 / denom
    return rhs - np.log(np.asarray(rh, dtype=np.float64))


def stage_e(config):
    results = []
    r_d = 0.055
    n = 8000
    rh = RNG.uniform(0.05, 0.985, size=n).astype(np.float32)
    B = RNG.uniform(0.05, 1.2, size=n).astype(np.float32)
    temp = RNG.uniform(240.0, 310.0, size=n).astype(np.float32)

    r_w = kohler_wet_radius_um(r_d, B, rh, temp)
    residual = np.abs(_kohler_residual(r_w, r_d, B, rh, temp))
    worst = _max_finite(residual)
    results.append(
        Result("E", "Koehler equation residual ~ 0", n, "max|residual| (ln RH)", worst, 5e-3,
               _passfail(worst, 5e-3), "ln RH = A/r_w - B r_d^3/(r_w^3-r_d^3)")
    )

    # limits: RH<=0 (here RH=0) and non-hygroscopic -> r_w = r_d
    rh_lim = np.array([0.0, 0.8, 0.8, 0.8], dtype=np.float32)
    b_lim = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float32)
    t_lim = np.full(4, 280.0, dtype=np.float32)
    w_lim = kohler_wet_radius_um(r_d, b_lim, rh_lim, t_lim)
    expect_dry = np.array([True, True, True, False])
    dev = float(np.max(np.abs(w_lim[expect_dry] - r_d)))
    results.append(
        Result("E", "dry / non-hygroscopic -> r_w=r_d", int(expect_dry.sum()), "max|r_w-r_d|", dev, 1e-6,
               _passfail(dev, 1e-6), "RH=0, kappa<=0 give no growth")
    )
    results.append(
        Result("E", "hygroscopic cell grows", 1, "max(r_d-r_w,0) (um)", max(0.0, float(r_d - w_lim[3])), 0.0,
               _passfail(max(0.0, float(r_d - w_lim[3])), 0.0), "RH=0.8,kappa=1")
    )

    # monotonicity in RH (fixed B, T) and in B (fixed RH, T)
    rh_line = np.linspace(0.1, 0.98, 60).astype(np.float32)
    w_rh = kohler_wet_radius_um(r_d, np.full(60, 0.6, np.float32), rh_line, np.full(60, 280.0, np.float32))
    drop_rh = max(0.0, -float(np.min(np.diff(w_rh))))
    results.append(
        Result("E", "r_w monotonic increasing in RH", 60, "max decrease", drop_rh, 1e-6,
               _passfail(drop_rh, 1e-6), "fixed kappa=0.6")
    )
    b_line = np.linspace(0.05, 1.2, 60).astype(np.float32)
    w_b = kohler_wet_radius_um(r_d, b_line, np.full(60, 0.85, np.float32), np.full(60, 280.0, np.float32))
    drop_b = max(0.0, -float(np.min(np.diff(w_b))))
    results.append(
        Result("E", "r_w monotonic increasing in kappa", 60, "max decrease", drop_b, 1e-6,
               _passfail(drop_b, 1e-6), "fixed RH=0.85")
    )

    # Kelvin-term effect vs microphysics simplified wet_radius (A ~ 0)
    rh_k = np.linspace(0.5, 0.98, 80).astype(np.float32)
    B_k = np.full(80, 0.6, np.float32)
    full = kohler_wet_radius_um(r_d, B_k, rh_k, np.full(80, 280.0, np.float32)).astype(np.float64)
    simple = _simplified_wet_radius_um(r_d, B_k.astype(np.float64), rh_k.astype(np.float64))
    rel = _max_finite(np.abs(full - simple) / full)
    results.append(
        Result("E", "Kelvin term vs A~0 simplification", 80, "max rel diff", rel, 1.0,
               "FINDING", "curvature lowers full-Koehler r_w at small r_d")
    )
    return results


# --------------------------------------------------------------------------- #
# Stage F: refractive-index internal mixing
# --------------------------------------------------------------------------- #
def stage_f(config):
    results = []
    info = _synthetic_species_info()
    species = list(info)
    refractive = {
        "SU": (1.45, 1e-8), "BC": (1.95, 0.79), "POM": (1.53, 0.006),
        "DU": (1.55, 0.0015), "SS": (1.50, 1e-7), "WAT": (1.34, 0.0),
    }
    n = 6000
    r_d = 0.055
    q = _random_q(species, n)
    rh = RNG.uniform(0.0, 0.98, size=n).astype(np.float32)
    temp = np.full(n, 285.0, dtype=np.float32)

    state = mix_mode_state(info, q, refractive, rh, temp, dry_radius_um=r_d)

    # manual reconstruction including water volume
    vol = {s: q[s].astype(np.float64) / (info[s]["density"] * 1000.0) for s in species}
    vdry = sum(vol.values())
    growth = np.maximum((state["r_w_um"].astype(np.float64) / r_d) ** 3 - 1.0, 0.0)
    vwat = vdry * growth
    num_re = sum(vol[s] * refractive[s][0] for s in species) + vwat * refractive["WAT"][0]
    num_im = sum(vol[s] * refractive[s][1] for s in species) + vwat * refractive["WAT"][1]
    den = vdry + vwat
    ref_re = np.divide(num_re, den, out=np.ones_like(num_re), where=den > 0)
    ref_im = np.divide(num_im, den, out=np.zeros_like(num_im), where=den > 0)
    err_re = _max_finite(_rel_err(state["n_re"][den > 0], ref_re[den > 0]))
    err_im = _max_finite(_rel_err(state["n_im"][den > 0], ref_im[den > 0]))
    results.append(
        Result("F", "n_re = vol-weighted (incl water)", int((den > 0).sum()), "max rel err", err_re, 1e-3,
               _passfail(err_re, 1e-3), "volume mixing rule")
    )
    results.append(
        Result("F", "n_im = vol-weighted (incl water)", int((den > 0).sum()), "max rel err", err_im, 2e-3,
               _passfail(err_im, 2e-3), "volume mixing rule")
    )

    # boundedness within component envelope (dry components + water)
    re_vals = [refractive[s][0] for s in species] + [refractive["WAT"][0]]
    im_vals = [refractive[s][1] for s in species] + [refractive["WAT"][1]]
    mix_re = state["n_re"].astype(np.float64)
    mix_im = state["n_im"].astype(np.float64)
    sel = den > 0
    over_re = _max_finite(np.maximum(min(re_vals) - mix_re[sel], mix_re[sel] - max(re_vals)))
    over_im = _max_finite(np.maximum(min(im_vals) - mix_im[sel], mix_im[sel] - max(im_vals)))
    results.append(
        Result("F", "n_re within component envelope", int(sel.sum()), "max overshoot", over_re, 1e-4,
               _passfail(over_re, 1e-4), "convex combination")
    )
    results.append(
        Result("F", "n_im within component envelope", int(sel.sum()), "max overshoot", over_im, 1e-4,
               _passfail(over_im, 1e-4), "convex combination")
    )

    # water dilution: pure absorbing aerosol n_re decreases monotonically toward 1.34 with RH
    bc_only = {"BC": np.full(60, 5e-9, np.float32)}
    rh_line = np.linspace(0.1, 0.97, 60).astype(np.float32)
    st = mix_mode_state({"BC": info["BC"]}, bc_only, {"BC": refractive["BC"], "WAT": refractive["WAT"]},
                        rh_line, np.full(60, 285.0, np.float32), dry_radius_um=r_d)
    rise = max(0.0, float(np.max(np.diff(st["n_re"].astype(np.float64)))))
    results.append(
        Result("F", "water dilutes n_re toward 1.34", 60, "max increase", rise, 1e-6,
               _passfail(rise, 1e-6), "BC-only, n_re -> water with RH")
    )
    return results


# --------------------------------------------------------------------------- #
# Stage G: optics LUT vs Mie + base-table reproduction
# --------------------------------------------------------------------------- #
def _sw05_wavelength(base_ds):
    bands = np.asarray(base_ds["LFL_SW_bands"].values, dtype=np.float64).ravel()
    return float(0.5 * (bands[4] + bands[5]))  # sw05 -> bands[4:6]


def _validate_mie_reference():
    """G0: confirm the independent Mie reference against analytic limits (mode-independent)."""
    results = []
    ray = mie_efficiencies(1.5, 0.0, 0.005, 0.55)
    m2 = 1.5 ** 2
    ray_ref = (8.0 / 3.0) * ray["size_parameter"] ** 4 * abs((m2 - 1.0) / (m2 + 2.0)) ** 2
    ray_err = abs(ray["q_sca"] - ray_ref) / ray_ref
    results.append(Result("G", "Mie Rayleigh limit (small x)", 1, "rel err", ray_err, 0.05,
                          _passfail(ray_err, 0.05), "q_sca vs Rayleigh"))
    big = [mie_efficiencies(1.5, 0.0, r, 0.55)["q_ext"] for r in (5.0, 10.0, 18.0)]
    far = max(abs(q - 2.0) for q in big)
    results.append(Result("G", "Mie geometric limit (large x)", 3, "max|Qext-2|", far, 0.35,
                          _passfail(far, 0.35), "extinction paradox Qext->2"))
    nonabs = mie_cross_sections_um2(1.5, 0.0, 0.2, 0.55)
    absdiff = abs(nonabs["ext"] - nonabs["sca"])
    results.append(Result("G", "Mie non-absorbing ext=sca", 1, "|ext-sca|", absdiff, 1e-9,
                          _passfail(absdiff, 1e-9), "no absorption when n_imag=0"))
    return results


def _g_mode_checks(mode, base, fine, wavelength):
    """G1/G2/G3 for one mode's base + fine sw05 tables."""
    results = []
    table_sg = float(base["sigmag"])
    n_real_c = np.asarray(base["refindex_real_sw"].values[4], dtype=np.float64)
    n_imag_c = np.asarray(base["refindex_im_sw"].values[4], dtype=np.float64)
    extp = np.asarray(base["extpsw_mie"].values[4, 0], dtype=np.float64)  # (im, re, radius)
    n_real_f = np.asarray(fine["n_real"].values, dtype=np.float64)
    n_imag_f = np.asarray(fine["n_imag"].values, dtype=np.float64)
    ext_f = np.asarray(fine["ext"].values, dtype=np.float64)              # (n_real, n_imag, radius)

    # G1: fine LUT reproduces base-table Mie interpolation
    worst, cnt = 0.0, 0
    for k in (20, 60, 120, 200, 300):
        interp = RegularGridInterpolator((n_imag_c, n_real_c), extp[:, :, k],
                                         bounds_error=False, fill_value=None)
        for ir in (10, 40, 70, 95):
            for ii in (0, 15, 45, 80):
                ref = float(interp([[n_imag_f[ii], n_real_f[ir]]])[0])
                worst = max(worst, abs(float(ext_f[ir, ii, k]) - ref) / max(abs(ref), 1e-12))
                cnt += 1
    results.append(Result("G", "%s fine LUT reproduces base interp" % mode, cnt, "max rel err", worst, 2e-3,
                          _passfail(worst, 2e-3), "refine_lut RegularGridInterpolator"))

    # G2: lookup_mode_optics indexing reproduces direct table lookup
    m = 1200
    nre = RNG.uniform(n_real_f[0], n_real_f[-1], m).astype(np.float32)
    nim = RNG.uniform(0.0, n_imag_f[-1], m).astype(np.float32)
    rw = RNG.uniform(fine["radius"].values[0], fine["radius"].values[-1], m).astype(np.float32)
    got_ext, _got_abs, _got_asm = lookup_mode_optics(nre, nim, rw, fine)
    ire = np.clip(np.searchsorted(n_real_f, nre, side="right") - 1, 0, len(n_real_f) - 1)
    iim = np.clip(np.searchsorted(n_imag_f, np.abs(nim), side="right") - 1, 0, len(n_imag_f) - 1)
    rad_f = np.asarray(fine["radius"].values, dtype=np.float64)
    irw = np.clip(np.searchsorted(rad_f, rw, side="right") - 1, 0, len(rad_f) - 1)
    idx_err = _max_finite(np.abs(got_ext.astype(np.float64) - ext_f[ire, iim, irw]))
    results.append(Result("G", "%s lookup floor-bin/|n_im|/clip exact" % mode, m, "max|delta|", idx_err, 1e-6,
                          _passfail(idx_err, 1e-6), "lower-bin index into fine LUT"))

    # G3: table must equal the number-averaged (mode-integrated) Mie at its own sigma_g
    i_re = int(np.argmin(np.abs(n_real_f - 1.5)))
    rad = np.asarray(fine["radius"].values, dtype=np.float64)
    sel = (rad >= 0.05) & (rad <= 10.0)
    sample = np.linspace(rad[sel][0], rad[sel][-1], 24)
    modeavg_ratio, qeff = [], []
    for r in sample:
        kk = int(np.argmin(np.abs(rad - r)))
        tab = float(ext_f[i_re, 0, kk])
        if tab <= 0:
            continue
        avg = mode_averaged_cross_sections_um2(float(n_real_f[i_re]), 0.0, float(rad[kk]),
                                               table_sg, wavelength)["ext"]
        modeavg_ratio.append(avg / tab)
        qeff.append(tab / (np.pi * rad[kk] ** 2))
    modeavg_ratio = np.array(modeavg_ratio)
    dev = abs(float(np.median(modeavg_ratio)) - 1.0)
    results.append(Result("G", "%s table = mode-integrated Mie" % mode, len(modeavg_ratio),
                          "|median modeavg/table - 1|", dev, 0.1, _passfail(dev, 0.1),
                          "sigma_g=%g; tau=sigma*N requires this" % table_sg))
    results.append(Result("G", "%s effective Q_ext" % mode, len(qeff), "median Q_eff",
                          float(np.median(qeff)), 2.0, "FINDING",
                          "mode-integrated ~ 2*exp(2 ln^2 %g)" % table_sg))
    return results


def _g_mono_checks(base, fine, wavelength):
    """Validate the monodisperse LUT (bin-resolved dust/sea-salt) = single-particle Mie."""
    results = []
    n_real_c = np.asarray(base["refindex_real_sw"].values[4], dtype=np.float64)
    n_imag_c = np.asarray(base["refindex_im_sw"].values[4], dtype=np.float64)
    extp = np.asarray(base["extpsw_mie"].values[4, 0], dtype=np.float64)  # (im, re, radius)
    n_real_f = np.asarray(fine["n_real"].values, dtype=np.float64)
    n_imag_f = np.asarray(fine["n_imag"].values, dtype=np.float64)
    ext_f = np.asarray(fine["ext"].values, dtype=np.float64)
    rad = np.asarray(fine["radius"].values, dtype=np.float64)

    worst, cnt = 0.0, 0
    for k in (20, 60, 120, 200, 300):
        interp = RegularGridInterpolator((n_imag_c, n_real_c), extp[:, :, k],
                                         bounds_error=False, fill_value=None)
        for ir in (10, 40, 70, 95):
            for ii in (0, 15, 45, 80):
                ref = float(interp([[n_imag_f[ii], n_real_f[ir]]])[0])
                worst = max(worst, abs(float(ext_f[ir, ii, k]) - ref) / max(abs(ref), 1e-12))
                cnt += 1
    results.append(Result("G", "mono fine LUT reproduces base interp", cnt, "max rel err", worst, 2e-3,
                          _passfail(worst, 2e-3), "refine_lut RegularGridInterpolator"))

    # base table = single-particle Mie at its coarse nodes (im=0), below the X_CAP asymptote
    devs, qeff = [], []
    for ir in (1, 3, 5):
        for k in (10, 80, 180):
            tab = float(extp[0, ir, k])
            if tab <= 0:
                continue
            mono = mie_cross_sections_um2(float(n_real_c[ir]), 0.0, float(rad[k]), wavelength)["ext"]
            devs.append(abs(tab - mono) / mono)
            qeff.append(tab / (np.pi * rad[k] ** 2))
    worst_mono = float(np.max(devs))
    results.append(Result("G", "mono table = single-particle Mie", len(devs), "max rel err", worst_mono, 5e-3,
                          _passfail(worst_mono, 5e-3), "sigma_g=1.0; bin-resolved dust/sea-salt"))
    results.append(Result("G", "mono effective Q_ext", len(qeff), "median Q_eff",
                          float(np.median(qeff)), 2.0, "FINDING", "single-particle Mie ~ 2"))
    return results


def _g_spectral_checks(base):
    """Validate LUT absorption (-> SSA) and asymmetry vs mode-integrated Mie across SW+LW bands.

    Homogeneous-sphere Mie (scipy Bessel) is only reliable up to size parameter
    ~200; generate_lut caps there (X_CAP) and the lognormal-mode reference is only
    trustworthy where its integration tail stays below that. We therefore validate
    only where the +N_SIGMA tail keeps x < X_CAP; beyond that the table uses the
    geometric asymptote by design. The reference integration width matches the
    generator (N_SIGMA=4).
    """
    x_cap, n_sigma = 200.0, 4.0
    results = []
    sg = float(base["sigmag"])
    tail = np.exp(n_sigma * np.log(sg))
    rad = np.asarray(base["particle_radius"].values, dtype=np.float64)
    sw = np.asarray(base["LFL_SW_bands"].values, dtype=np.float64)
    lw = np.asarray(base["LFL_LW_bands"].values, dtype=np.float64)
    families = [
        ("sw", 0.5 * (sw[:-1] + sw[1:]), "refindex_real_sw", "refindex_im_sw",
         "extpsw_mie", "abspsw_mie", "asmpsw", (0, 4, 9, 13)),
        ("lw", 0.5 * (lw[:-1] + lw[1:]), "refindex_real_lw", "refindex_im_lw",
         "extplw_mie", "absplw_mie", "asmplw", (0, 3, 7)),
    ]
    ssa_dev, asm_dev, n = 0.0, 0.0, 0
    for _kind, mids, re_name, im_name, ext_name, abs_name, asm_name, bands in families:
        re_g = np.asarray(base[re_name].values, dtype=np.float64)
        im_g = np.asarray(base[im_name].values, dtype=np.float64)
        ext_t = np.asarray(base[ext_name].values, dtype=np.float64)
        abs_t = np.asarray(base[abs_name].values, dtype=np.float64)
        asm_t = np.asarray(base[asm_name].values, dtype=np.float64)
        for b in bands:
            wl = float(mids[b])
            for ri in (1, 4):
                for ii in (0, 4, 8):
                    for k in (8, 30, 80, 150):
                        if 2.0 * np.pi * rad[k] * tail / wl > x_cap:
                            continue  # tail exceeds reliable Mie regime
                        ext = float(ext_t[b, 0, ii, ri, k])
                        if ext <= 0:
                            continue
                        cs = mode_averaged_cross_sections_um2(
                            float(re_g[b, ri]), float(im_g[b, ii]), float(rad[k]), sg, wl, n_sigma=n_sigma)
                        if cs["ext"] <= 0:
                            continue
                        ssa_dev = max(ssa_dev, abs((1.0 - float(abs_t[b, 0, ii, ri, k]) / ext)
                                                   - cs["sca"] / cs["ext"]))
                        asm_dev = max(asm_dev, abs(float(asm_t[b, 0, ii, ri, k]) - cs["asymmetry"]))
                        n += 1
    results.append(Result("G", "spectral SSA vs mode-integrated Mie", n, "max|delta SSA|", ssa_dev, 0.02,
                          _passfail(ssa_dev, 0.02), "1-abs/ext across SW+LW bands (x<200)"))
    results.append(Result("G", "spectral asymmetry vs mode-integrated Mie", n, "max|delta g|", asm_dev, 0.05,
                          _passfail(asm_dev, 0.05), "scattering-weighted g, SW+LW (x<200); quadrature-limited"))
    return results


def stage_g(config):
    results = _validate_mie_reference()
    specs = config["Schemes"]["MAM4"]["modes"]
    for mode in ("a1", "a2", "a3", "a4"):
        spec = specs.get(mode)
        if spec is None:
            continue
        base_path = _path(spec["filename_sarb"])
        fine_path = base_path.replace("larc", "sw5_larc", 1)
        if not (os.path.exists(base_path) and os.path.exists(fine_path)):
            results.append(Result("G", "%s optics data present" % mode, 0, "-", float("nan"), 0.0,
                                  "WARN", "SARB LUTs missing locally"))
            continue
        base = xr.open_dataset(base_path)
        fine = xr.open_dataset(fine_path)
        try:
            results.extend(_g_mode_checks(mode, base, fine, _sw05_wavelength(base)))
            if mode == "a1":
                results.extend(_g_spectral_checks(base))
        finally:
            base.close()
            fine.close()

    # monodisperse LUT used by the bin-resolved dust/sea-salt modes
    bin_spec = specs.get("du1")
    if bin_spec is not None:
        mono_base = _path(bin_spec["filename_sarb"])
        mono_fine = mono_base.replace("larc", "sw5_larc", 1)
        if os.path.exists(mono_base) and os.path.exists(mono_fine):
            base = xr.open_dataset(mono_base)
            fine = xr.open_dataset(mono_fine)
            try:
                results.extend(_g_mono_checks(base, fine, _sw05_wavelength(base)))
            finally:
                base.close()
                fine.close()
    return results


# --------------------------------------------------------------------------- #
# Stage H: layer optical depth
# --------------------------------------------------------------------------- #
def stage_h(config):
    n = 6000
    delp = RNG.uniform(0.0, 5000.0, n).astype(np.float32)
    number = RNG.uniform(0.0, 1e10, n).astype(np.float32)
    cross = RNG.uniform(0.0, 200.0, n).astype(np.float32)
    tau = layer_optical_depth(delp, number, cross)
    ref = cross.astype(np.float64) * 1e-12 * number.astype(np.float64) * delp.astype(np.float64) / G
    err = _max_finite(_rel_err(tau, ref))
    return [
        Result("H", "tau = sigma*1e-12*N*delp/g", n, "max rel err", err, 1e-5,
               _passfail(err, 1e-5), "um^2->m^2 (1e-12); delp/g = kg/m2; dimensionless tau"),
    ]


# --------------------------------------------------------------------------- #
# Stage I: external mixing of modes
# --------------------------------------------------------------------------- #
def _mode_ds(ext, sca, asm):
    dims = ("lev", "lat", "lon")
    coords = {"lev": np.array([1000.0, 850.0], np.float32),
              "lat": np.array([-30.0, 30.0], np.float32),
              "lon": np.array([0.0, 120.0, 240.0], np.float32)}
    delp = np.full(ext.shape, 100.0, np.float32)
    return xr.Dataset({
        "DELP": (dims, delp),
        "Extinction_Layer_Optical_Depth": (dims, ext),
        "Scattering_Layer_Optical_Depth": (dims, sca),
        "Layer_Asymmetry_Parameter": (dims, asm),
        "Extinction_Column_Optical_Depth": (("lat", "lon"), ext.sum(0)),
    }, coords=coords)


def stage_i(config):
    from mode_external_mix import mix_mode_datasets
    shape = (2, 2, 3)
    datasets = []
    exts, scas, asms = [], [], []
    for _ in range(4):
        ext = RNG.uniform(0.0, 0.5, shape).astype(np.float32)
        sca = (ext * RNG.uniform(0.0, 1.0, shape)).astype(np.float32)
        asm = RNG.uniform(-0.2, 0.9, shape).astype(np.float32)
        datasets.append(_mode_ds(ext, sca, asm))
        exts.append(ext.astype(np.float64))
        scas.append(sca.astype(np.float64))
        asms.append(asm.astype(np.float64))
    mixed = mix_mode_datasets(datasets)
    ref_ext = sum(exts)
    ref_sca = sum(scas)
    ref_wsum = sum(s * a for s, a in zip(scas, asms))
    ref_asm = np.where(ref_sca > 1e-5, ref_wsum / np.where(ref_sca > 0, ref_sca, 1.0), 0.0)
    e1 = _max_finite(_rel_err(mixed["Extinction_Layer_Optical_Depth"].values, ref_ext))
    e2 = _max_finite(_rel_err(mixed["Scattering_Layer_Optical_Depth"].values, ref_sca))
    e3 = _max_finite(np.abs(mixed["Layer_Asymmetry_Parameter"].values - ref_asm))
    return [
        Result("I", "external ext = sum of modes", ref_ext.size, "max rel err", e1, 1e-5,
               _passfail(e1, 1e-5), "additive extinction"),
        Result("I", "external sca = sum of modes", ref_sca.size, "max rel err", e2, 1e-5,
               _passfail(e2, 1e-5), "additive scattering"),
        Result("I", "asymmetry scattering-weighted", ref_asm.size, "max|delta|", e3, 1e-5,
               _passfail(e3, 1e-5), "g = sum(sca*g)/sum(sca)"),
    ]


# --------------------------------------------------------------------------- #
# Stage J: VIS correction
# --------------------------------------------------------------------------- #
def stage_j(config):
    lat = np.linspace(-80.0, 80.0, 20).astype(np.float32)
    lon = np.linspace(0.0, 350.0, 30).astype(np.float32)
    ext = xr.DataArray(RNG.uniform(0.0, 2.0, (20, 30)).astype(np.float32), dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lon})
    inte = xr.DataArray(RNG.uniform(1e-3, 2.0, (20, 30)).astype(np.float32), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})
    factor, stats = compute_vis_factor_local(ext, inte)
    raw = (ext.values / inte.values)
    ref = np.clip(raw, 0.25, 4.0)
    err = _max_finite(np.abs(factor.values - ref))
    results = [
        Result("J", "factor = clip(ext/int, 0.25, 4)", ext.size, "max|delta|", err, 1e-6,
               _passfail(err, 1e-6), "bounded column-ratio scaling"),
    ]
    # applying factor preserves external column where factor uncapped
    uncapped = (raw >= 0.25) & (raw <= 4.0)
    corrected = (inte.values * factor.values)
    preserve = _max_finite(np.abs(corrected[uncapped] - ext.values[uncapped]) /
                           np.maximum(ext.values[uncapped], 1e-12))
    results.append(
        Result("J", "uncapped correction preserves column", int(uncapped.sum()), "max rel err",
               preserve, 1e-5, _passfail(preserve, 1e-5), "corrected col == external col")
    )
    return results


def compute_vis_factor_local(ext, inte):
    from vis_correction import compute_vis_factor
    return compute_vis_factor(ext, inte)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
STAGES = {
    "A": ("mass -> mode allocation", stage_a),
    "B": ("dry-volume internal mixing", stage_b),
    "C": ("number conc + lognormal + sigma_g", stage_c),
    "D": ("mixed hygroscopicity", stage_d),
    "E": ("Koehler wet radius", stage_e),
    "F": ("refractive-index mixing", stage_f),
    "G": ("optics LUT vs Mie", stage_g),
    "H": ("layer optical depth", stage_h),
    "I": ("external mixing", stage_i),
    "J": ("VIS correction", stage_j),
}

_SYMBOL = {"PASS": "ok ", "FAIL": "XX ", "WARN": "-- ", "FINDING": "** "}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="ALL", help="comma list of stage letters or ALL")
    parser.add_argument("--aerosol", default=None, help="aerosol config (default auto)")
    args = parser.parse_args(argv)

    config = load_config(args.aerosol) if args.aerosol else _config()
    chosen = list(STAGES) if args.stage.upper() == "ALL" else [s.strip().upper() for s in args.stage.split(",")]

    print("%-2s %-3s %-38s %8s %-26s %12s %10s" % ("", "st", "check", "n", "metric", "value", "tol"))
    print("-" * 116)
    results = []
    for letter in chosen:
        title, func = STAGES[letter]
        print("[%s] %s" % (letter, title))
        for r in func(config):
            results.append(r)
            val = "%.4g" % r.value if np.isfinite(r.value) else "nan"
            tol = "%.1g" % r.tol if r.status in ("PASS", "FAIL") else "-"
            print("%s %-3s %-38s %8d %-26s %12s %10s%s" % (
                _SYMBOL.get(r.status, "?  "), r.stage, r.check[:38], r.n, r.metric[:26], val, tol,
                ("  | " + r.note) if r.note else ""))

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_find = sum(1 for r in results if r.status == "FINDING")
    n_warn = sum(1 for r in results if r.status == "WARN")
    print("-" * 116)
    print("PASS=%d FAIL=%d FINDING=%d WARN=%d" % (n_pass, n_fail, n_find, n_warn))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
