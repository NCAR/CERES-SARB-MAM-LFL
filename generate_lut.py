"""Regenerate SARB MAM base optics tables from validated homogeneous-sphere Mie.

Produces corrected `c000003.v2` base tables that fix the two defects found by
`verify_physics.py` (see `docs/physics-verification.md`) in `c000002.v2`:

  1. `extpsw_mie` was ~2.5x (fine) to ~3.6x (coarse) below physical Mie.
  2. `sigmag` had drifted from the canonical MAM4 widths for modes 1 and 3.

The stored cross sections are **number-averaged over the lognormal mode**
(`mie_lognormal`), the quantity the production formula `tau = sigma * N`
requires, evaluated at the canonical `sigma_g` and at each shortwave/longwave
band midpoint. Schema, dims, axes and the `extpsw_scaled = extpsw_mie * C / r^3`
relationship match `c000002.v2` so the new files are drop-in.

The Mie cross section at a given (wavelength, refractive index) is independent
of mode, so it is computed once per (band, n_real, n_imag) node on a shared
radius grid and reused for all 400 median radii and all four modes. Beyond size
parameter `X_CAP` the geometric-optics asymptote (constant efficiency, cross
section ~ pi r^2) is used; the lognormal weight there is negligible.

    python generate_lut.py                 # all modes -> c000003.v2
    python generate_lut.py --modes 1 3     # subset
    python generate_lut.py --check         # validate vs mie_lognormal, no write
"""

import argparse
import os
from datetime import datetime, UTC

import numpy as np
import xarray as xr

from mie_sphere import mie_cross_sections_um2
from mie_lognormal import mode_averaged_cross_sections_um2

# Canonical MAM4 widths (RRTMG/CAM source == config == SARB c000000).
CANONICAL_SIGMA_G = {1: 1.8, 2: 1.6, 3: 1.8, 4: 1.6}
# extpsw_scaled = extpsw_mie * C / r^3 ; C is a fixed per-mode unit/density
# constant carried verbatim from c000002.v2 (independent of the Mie content).
SCALED_C = {1: 706.78, 2: 706.78, 3: 1644.5, 4: 706.78}

X_CAP = 200.0      # above this size parameter use the geometric-optics asymptote
N_SIGMA = 4.0      # lognormal integration half-width
N_GRID = 700       # shared radius-grid points

_W = {}            # per-worker shared state (set by _init_worker)


# --------------------------------------------------------------------------- #
# Worker (top-level for the 'spawn' start method)
# --------------------------------------------------------------------------- #
def _init_worker(radius_grid, medians, mode_sigmas, monodisperse=False):
    _W["radius"] = radius_grid
    _W["medians"] = medians
    _W["mode_sigmas"] = mode_sigmas
    _W["monodisperse"] = monodisperse
    _W["lnr"] = np.log(radius_grid)
    _W["dlnr"] = np.gradient(np.log(radius_grid))


def _mie_curve(wavelength_um, n_real, n_imag, radius=None):
    """Per-particle ext, sca, g*sca (um^2) on a radius grid (default the shared grid)."""
    radius = _W["radius"] if radius is None else radius
    x = 2.0 * np.pi * radius / wavelength_um
    ext = np.empty_like(radius)
    sca = np.empty_like(radius)
    gsca = np.empty_like(radius)
    cap = None
    for i in range(radius.size):
        r = radius[i]
        area = np.pi * r * r
        if cap is None or x[i] <= X_CAP:
            cross = mie_cross_sections_um2(n_real, n_imag, float(r), wavelength_um)
            cap = (cross["ext"] / area, cross["sca"] / area, cross["asymmetry"])
        q_ext, q_sca, g = cap
        ext[i] = q_ext * area
        sca[i] = q_sca * area
        gsca[i] = g * q_sca * area
    return ext, sca, gsca


def _integrate(ext, sca, gsca):
    """Number-averaged ext/abs/asm for every median radius and every mode."""
    lnr = _W["lnr"]
    dlnr = _W["dlnr"]
    ln_med = np.log(_W["medians"])
    out = {}
    for mode, sigma_g in _W["mode_sigmas"].items():
        ln_sg = np.log(sigma_g)
        weight = np.exp(-((lnr[None, :] - ln_med[:, None]) ** 2) / (2.0 * ln_sg ** 2)) * dlnr[None, :]
        norm = weight.sum(axis=1)
        sca_w = weight @ sca
        ext_avg = (weight @ ext) / norm
        sca_avg = sca_w / norm
        asm_avg = np.where(sca_w > 0.0, (weight @ gsca) / sca_w, 0.0)
        out[mode] = (ext_avg, np.maximum(ext_avg - sca_avg, 0.0), asm_avg)
    return out


def _worker(task):
    kind, b, re_idx, im_idx, wavelength, n_real, n_imag = task
    if _W.get("monodisperse"):
        # single-particle Mie evaluated directly at each table radius (no integration)
        ext, sca, gsca = _mie_curve(wavelength, n_real, n_imag, _W["medians"])
        with np.errstate(divide="ignore", invalid="ignore"):
            asm = np.where(sca > 0.0, gsca / sca, 0.0)
        return kind, b, re_idx, im_idx, {"mono": (ext, np.maximum(ext - sca, 0.0), asm)}
    ext, sca, gsca = _mie_curve(wavelength, n_real, n_imag)
    return kind, b, re_idx, im_idx, _integrate(ext, sca, gsca)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _band_midpoints(limits):
    limits = np.asarray(limits, dtype=np.float64)
    return 0.5 * (limits[:-1] + limits[1:])


def _build_tasks(template):
    tasks = []
    specs = {
        "sw": (_band_midpoints(template["LFL_SW_bands"].values),
               template["refindex_real_sw"].values, template["refindex_im_sw"].values),
        "lw": (_band_midpoints(template["LFL_LW_bands"].values),
               template["refindex_real_lw"].values, template["refindex_im_lw"].values),
    }
    for kind, (mids, re_grid, im_grid) in specs.items():
        n_band, n_re = re_grid.shape
        n_im = im_grid.shape[1]
        for b in range(n_band):
            for re_idx in range(n_re):
                for im_idx in range(n_im):
                    tasks.append((kind, b, re_idx, im_idx, float(mids[b]),
                                  float(re_grid[b, re_idx]), float(im_grid[b, im_idx])))
    return tasks, {k: (v[1].shape[0], v[2].shape[1], v[1].shape[1]) for k, v in specs.items()}


def _radius_grid(medians):
    widest = max(CANONICAL_SIGMA_G.values())
    span = np.exp(N_SIGMA * np.log(widest))
    return np.geomspace(medians.min() / span, medians.max() * span, N_GRID)


def _accumulate(results, shapes, modes, n_radius):
    store = {kind: {mode: {q: np.zeros((shapes[kind][0], 1, shapes[kind][1], shapes[kind][2], n_radius))
                           for q in ("ext", "abs", "asm")}
                    for mode in modes}
             for kind in shapes}
    for kind, b, re_idx, im_idx, per_mode in results:
        for mode, (ext, absn, asm) in per_mode.items():
            if mode not in modes:
                continue
            store[kind][mode]["ext"][b, 0, im_idx, re_idx, :] = ext
            store[kind][mode]["abs"][b, 0, im_idx, re_idx, :] = absn
            store[kind][mode]["asm"][b, 0, im_idx, re_idx, :] = asm
    return store


def _build_dataset(template, mode, store, radius, sigma_g, scaled_c):
    sw_dims = ("sw_band", "mode", "refindex_im", "refindex_real", "radii_number")
    lw_dims = ("lw_band", "mode", "refindex_im", "refindex_real", "radii_number")
    c = scaled_c
    r3 = (radius ** 3)[None, None, None, None, :]

    def scaled(mie):
        return mie * c / r3

    sw = store["sw"][mode]
    lw = store["lw"][mode]
    data = {
        "extpsw_mie": (sw_dims, sw["ext"]),
        "extpsw_scaled": (sw_dims, scaled(sw["ext"])),
        "abspsw_mie": (sw_dims, sw["abs"]),
        "abspsw_scaled": (sw_dims, scaled(sw["abs"])),
        "asmpsw": (sw_dims, sw["asm"]),
        "extplw_mie": (lw_dims, lw["ext"]),
        "extplw_scaled": (lw_dims, scaled(lw["ext"])),
        "absplw_mie": (lw_dims, lw["abs"]),
        "absplw_scaled": (lw_dims, scaled(lw["abs"])),
        "asmplw": (lw_dims, lw["asm"]),
    }
    # carry axes / metadata verbatim, except the corrected sigmag
    for name in ("refindex_real_sw", "refindex_im_sw", "refindex_real_lw", "refindex_im_lw",
                 "particle_radius", "LFL_SW_bands", "LFL_LW_bands", "opticsmethod",
                 "dgnum", "dgnumlo", "dgnumhi", "rhcrystal", "rhdeliques"):
        data[name] = (template[name].dims, template[name].values)

    ds = xr.Dataset(data)
    ds["sigmag"] = ((), np.float64(sigma_g))
    for name in template.data_vars:
        if name in ds and template[name].attrs:
            ds[name].attrs.update(template[name].attrs)
    ds["sigmag"].attrs.update(template["sigmag"].attrs)
    monodisperse = mode == "mono"
    basis = ("monodisperse single-particle Mie (sigma_g=1.0)" if monodisperse
             else "number-averaged over lognormal mode (sigma_g=%.2f)" % sigma_g)
    ds.attrs.update({
        "history": "%s generate_lut.py: %s homogeneous-sphere Mie" %
                   (datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "monodisperse" if monodisperse else "number-averaged"),
        "source": "mie_sphere.py (homogeneous sphere)" +
                  ("" if monodisperse else " + mie_lognormal lognormal-mode integration"),
        "mie_basis": basis,
        "wavelength_convention": "band midpoint of LFL_SW_bands / LFL_LW_bands",
        "predecessor": "c000002.v2 (extpsw_mie ~2.5-3.6x low; sigmag narrowed for modes 1,3)",
        "note": "dgnum carried from c000002.v2 template; sigmag set per mode/bin",
    })
    return ds


def _check(template, medians, radius_grid):
    """Spot-check generator integration against mie_lognormal at a few points."""
    _init_worker(radius_grid, medians, CANONICAL_SIGMA_G)
    mids = _band_midpoints(template["LFL_SW_bands"].values)
    re_grid = template["refindex_real_sw"].values
    im_grid = template["refindex_im_sw"].values
    print("validate generator vs mie_lognormal (number-averaged), mode1 sigma_g=1.8:")
    worst = 0.0
    for b in (4, 9):
        wl = float(mids[b])
        for re_idx in (1, 4):
            n_re = float(re_grid[b, re_idx])
            n_im = float(im_grid[b, 0])
            ext, sca, gsca = _mie_curve(wl, n_re, n_im)
            out = _integrate(ext, sca, gsca)[1]  # mode 1
            for r_target in (0.1, 1.0, 5.0):
                j = int(np.argmin(np.abs(medians - r_target)))
                ref = mode_averaged_cross_sections_um2(n_re, n_im, float(medians[j]), 1.8, wl)["ext"]
                got = out[0][j]
                rel = abs(got - ref) / ref
                worst = max(worst, rel)
                print("  band%2d wl=%.3f n=%.3f r=%.3f  gen=%.5g ref=%.5g rel=%.2e"
                      % (b + 1, wl, n_re, medians[j], got, ref, rel))
    print("worst rel err vs reference: %.2e" % worst)
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--optics-dir", default=os.path.expanduser("~/Data/Optics/SARB"))
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--template-suffix", default="c000002.v2")
    parser.add_argument("--out-suffix", default="c000003.v2")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--check", action="store_true", help="validate vs mie_lognormal, no write")
    parser.add_argument("--monodisperse", action="store_true",
                        help="single-particle Mie at each radius (sigma_g=1.0); write one mam4_mono table")
    args = parser.parse_args(argv)

    template_path = os.path.join(args.optics_dir, "mam4_mode1_larc_%s.nc" % args.template_suffix)
    template = xr.open_dataset(template_path)
    medians = np.asarray(template["particle_radius"].values, dtype=np.float64)
    radius_grid = _radius_grid(medians)

    if args.check:
        _check(template, medians, radius_grid)
        return 0

    tasks, shapes = _build_tasks(template)
    modes = ["mono"] if args.monodisperse else list(args.modes)
    print("nodes: %d  radius-grid: %d  monodisperse: %s  modes: %s  processes: %d"
          % (len(tasks), radius_grid.size, args.monodisperse, modes, args.processes))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.processes, initializer=_init_worker,
                  initargs=(radius_grid, medians, CANONICAL_SIGMA_G, args.monodisperse)) as pool:
        results = []
        for k, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=8)):
            results.append(res)
            if (k + 1) % 200 == 0:
                print("  %d / %d nodes" % (k + 1, len(tasks)), flush=True)

    store = _accumulate(results, shapes, set(modes), medians.size)
    if args.monodisperse:
        ds = _build_dataset(template, "mono", store, medians, 1.0, SCALED_C[1])
        out_path = os.path.join(args.optics_dir, "mam4_mono_larc_%s.nc" % args.out_suffix)
        ds.to_netcdf(out_path)
        col = float(ds["extpsw_mie"].values[4, 0, 0, 1, 200] / (np.pi * medians[200] ** 2))
        print("wrote %s  (monodisperse sigma_g=1.0, sample Q_eff=%.3f)" % (out_path, col))
        return 0
    for mode in args.modes:
        ds = _build_dataset(template, mode, store, medians, CANONICAL_SIGMA_G[mode], SCALED_C[mode])
        out_path = os.path.join(args.optics_dir, "mam4_mode%d_larc_%s.nc" % (mode, args.out_suffix))
        ds.to_netcdf(out_path)
        col = float(ds["extpsw_mie"].values[4, 0, 0, 1, 200] / (np.pi * medians[200] ** 2))
        print("wrote %s  (mode%d sigma_g=%.2f, sample Q_eff=%.3f)"
              % (out_path, mode, CANONICAL_SIGMA_G[mode], col))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
