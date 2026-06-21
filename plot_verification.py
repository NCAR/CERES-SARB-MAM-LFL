"""Physics figures for the MAM internal-mixing / radiative-property chain.

Each figure shows the physics that `verify_physics.py` confirms: the modal/bin
size structure, hygroscopic growth, refractive-index mixing, Mie optics, the
mode-integrated cross section the optical depth requires, and the resulting AOD.
Parameter-space panels use NSF NCAR brand styling (after ../DAVINCI); the global
AOD maps use the sister-repo cartopy style. Every figure is written as PDF and
PNG (300 dpi).

    python plot_verification.py --outdir ~/Plots/verification
"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_config import load_config
from mode_physics import kohler_wet_radius_um, mix_mode_state
from mie_sphere import mie_efficiencies, mie_cross_sections_um2
from mie_lognormal import mode_averaged_cross_sections_um2
import verify_physics as vp

# --------------------------------------------------------------------------- #
# NSF NCAR brand style (mirrors ../DAVINCI/davinci_monet/plots/style.py)
# --------------------------------------------------------------------------- #
NCAR = {
    "space": "#011837", "dark_blue": "#00357A", "ncar_blue": "#0A5DDA",
    "aqua": "#00A2B4", "orange": "#FF8C00", "yellow": "#FFDD31",
    "gray": "#58595B", "red": "#D62839", "green": "#2E8B57", "purple": "#7B68EE",
}
PALETTE = [NCAR[c] for c in ("ncar_blue", "aqua", "orange", "purple", "green", "red", "dark_blue")]
STATUS_COLOR = {"PASS": NCAR["green"], "FAIL": NCAR["red"], "FINDING": NCAR["ncar_blue"], "WARN": NCAR["gray"]}
SW05_UM = 0.54625


def apply_style():
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="deep")
    except ImportError:
        pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 15,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "-",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  %s.{pdf,png}" % name)


# --------------------------------------------------------------------------- #
# Size structure: lognormal modes + bin-resolved dust / sea salt
# --------------------------------------------------------------------------- #
def fig_size_distributions(outdir, config):
    modes = config["Schemes"]["MAM4"]["modes"]
    r = np.geomspace(2e-3, 30.0, 800)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, color in zip(("a1", "a2", "a3", "a4"),
                          (NCAR["ncar_blue"], NCAR["aqua"], NCAR["orange"], NCAR["purple"])):
        rg = float(modes[key]["dry_radius_um"]); sg = float(modes[key]["sigma_g"])
        ls = np.log(sg)
        pdf = np.exp(-((np.log(r) - np.log(rg)) ** 2) / (2 * ls ** 2)) / (ls * np.sqrt(2 * np.pi))
        ax.plot(r, pdf / pdf.max(), color=color,
                label="%s  $r_g$=%.3g $\\mu$m, $\\sigma_g$=%.1f" % (key, rg, sg))
    du = [float(modes["du%d" % i]["dry_radius_um"]) for i in range(1, 6)]
    ss = [float(modes["ss%d" % i]["dry_radius_um"]) for i in range(1, 6)]
    ax.vlines(du, 0, 0.85, color=NCAR["red"], lw=2.2, label="dust bins (monodisperse)")
    ax.vlines(ss, 0, 0.7, color=NCAR["green"], lw=2.2, ls=(0, (4, 2)), label="sea-salt bins (monodisperse)")
    ax.set_xscale("log")
    ax.set_xlabel("dry radius ($\\mu$m)"); ax.set_ylabel("normalized $dN/d\\ln r$")
    ax.set_title("MAM4 size structure: internally-mixed lognormal modes\nand bin-resolved dust / sea salt")
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.set_xlim(2e-3, 30); ax.set_ylim(0, 1.08)
    save(fig, outdir, "size_distributions")


# --------------------------------------------------------------------------- #
# Hygroscopicity b(RH)
# --------------------------------------------------------------------------- #
def fig_hygroscopicity(outdir):
    rh = np.linspace(0.0, 1.0, 101)
    species = [
        ("sulfate", [2.42848, -3.85261, 1.88159], NCAR["ncar_blue"]),
        ("sea salt", [4.83257, -6.92329, 3.27805], NCAR["aqua"]),
        ("organic / dust", [0.14, 0.0, 0.0], NCAR["orange"]),
        ("black carbon", [0.01, 0.0, 0.0], NCAR["gray"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for label, (b0, b1, b2), c in species:
        ax.plot(rh, b0 + b1 * rh + b2 * rh ** 2, color=c, label=label)
    ax.set_xlabel("relative humidity"); ax.set_ylabel("hygroscopicity  $b(RH)$")
    ax.set_title("Hygroscopicity $b(RH)=b_0+b_1\\,RH+b_2\\,RH^2$\n"
                 "mixed mode $\\kappa$ = volume-weighted mean")
    ax.legend(frameon=True); ax.set_ylim(bottom=0)
    save(fig, outdir, "hygroscopicity")


# --------------------------------------------------------------------------- #
# Köhler hygroscopic growth
# --------------------------------------------------------------------------- #
def fig_kohler(outdir):
    rh = np.linspace(0.05, 0.985, 160).astype(np.float32)
    T = np.full_like(rh, 290.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for label, B, c in [("black carbon  $\\kappa$=0.01", 0.01, NCAR["gray"]),
                        ("dust  0.14", 0.14, NCAR["orange"]),
                        ("sulfate  0.6", 0.6, NCAR["ncar_blue"]),
                        ("sea salt  1.2", 1.2, NCAR["aqua"])]:
        w = kohler_wet_radius_um(0.055, np.full_like(rh, B), rh, T)
        ax1.plot(rh, w / 0.055, color=c, label=label)
    ax1.set_xlabel("relative humidity"); ax1.set_ylabel("growth factor  $r_w/r_d$")
    ax1.set_title("(a) growth factor  ($r_d=0.055\\ \\mu$m)")
    ax1.legend(frameon=True)
    for label, rd, B, c in [("sulfate, $r_d$=0.055", 0.055, 0.6, NCAR["ncar_blue"]),
                           ("dust, $r_d$=0.73", 0.73, 0.14, NCAR["orange"]),
                           ("sea salt, $r_d$=1.0", 1.0, 1.2, NCAR["aqua"])]:
        w = kohler_wet_radius_um(rd, np.full_like(rh, B), rh, T)
        ax2.plot(rh, w, color=c, label=label)
    ax2.set_yscale("log")
    ax2.set_xlabel("relative humidity"); ax2.set_ylabel("wet radius  $r_w$ ($\\mu$m)")
    ax2.set_title("(b) wet radius by species / size")
    ax2.legend(frameon=True)
    fig.suptitle("Köhler equilibrium wet radius", fontweight="bold")
    save(fig, outdir, "kohler_growth")


# --------------------------------------------------------------------------- #
# Refractive-index volume mixing (water dilution)
# --------------------------------------------------------------------------- #
def fig_refractive_mixing(outdir):
    rh = np.linspace(0.0, 0.97, 90).astype(np.float32)
    info = {"BC": {"density": 1.0, "hygroscopicity": [0.3, 0.0, 0.0]}}
    refr = {"BC": (1.95, 0.79), "WAT": (1.34, 0.0)}
    q = {"BC": np.full(rh.shape, 5e-9, np.float32)}
    st = mix_mode_state(info, q, refr, rh, np.full_like(rh, 290.0), dry_radius_um=0.055)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(rh, st["n_re"], color=NCAR["ncar_blue"])
    ax1.axhline(1.95, color=NCAR["gray"], ls="--", lw=1.2, label="dry component (1.95)")
    ax1.axhline(1.34, color=NCAR["aqua"], ls=":", lw=1.6, label="water (1.34)")
    ax1.set_xlabel("relative humidity"); ax1.set_ylabel("real index  $n_{re}$")
    ax1.set_title("(a) real index diluted toward water"); ax1.legend(frameon=True)
    ax2.plot(rh, st["n_im"], color=NCAR["red"])
    ax2.axhline(0.79, color=NCAR["gray"], ls="--", lw=1.2, label="dry component (0.79)")
    ax2.set_xlabel("relative humidity"); ax2.set_ylabel("imaginary index  $n_{im}$")
    ax2.set_title("(b) absorption diluted by water"); ax2.legend(frameon=True)
    fig.suptitle("Volume-weighted refractive-index mixing (with water uptake)", fontweight="bold")
    save(fig, outdir, "refractive_mixing")


# --------------------------------------------------------------------------- #
# Mie efficiencies vs size parameter
# --------------------------------------------------------------------------- #
def fig_mie_efficiency(outdir):
    radii = np.geomspace(0.005, 30.0, 240)
    x = 2 * np.pi * radii / 0.55
    qext = np.array([mie_efficiencies(1.5, 0.0, float(r), 0.55)["q_ext"] for r in radii])
    eff_abs = [mie_efficiencies(1.5, 0.01, float(r), 0.55) for r in radii]
    qext_a = np.array([e["q_ext"] for e in eff_abs])
    qabs_a = np.array([e["q_abs"] for e in eff_abs])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.semilogx(x, qext, color=NCAR["ncar_blue"], label="$Q_{ext}$  (non-absorbing, $n$=1.5)")
    ax.semilogx(x, qabs_a, color=NCAR["red"], label="$Q_{abs}$  (absorbing, $n$=1.5+0.01$i$)")
    ax.axhline(2.0, color=NCAR["gray"], ls="--", lw=1.2, label="geometric limit  $Q_{ext}\\to2$")
    ax.set_xlabel("size parameter  $x = 2\\pi r/\\lambda$"); ax.set_ylabel("efficiency  $Q$")
    ax.set_title("Homogeneous-sphere Mie efficiencies\nRayleigh rise, resonances, geometric limit")
    ax.legend(frameon=True, loc="upper left"); ax.set_ylim(0, 4.6)
    save(fig, outdir, "mie_efficiency")


# --------------------------------------------------------------------------- #
# Mode-integrated cross section (the quantity tau = sigma * N requires)
# --------------------------------------------------------------------------- #
def fig_mode_integrated(outdir, optics_dir):
    new = xr.open_dataset(os.path.join(optics_dir, "mam4_mode1_larc_c000003.v2.nc"))
    n_real = np.asarray(new["refindex_real_sw"].values[4], dtype=np.float64)
    i_re = int(np.argmin(np.abs(n_real - 1.5)))
    n_re = float(n_real[i_re])
    rad = np.asarray(new["particle_radius"].values, dtype=np.float64)
    sel = np.flatnonzero((rad >= 0.05) & (rad <= 10.0))
    sel = sel[np.unique(np.linspace(0, sel.size - 1, 38).astype(int))]
    r = rad[sel]
    lut = np.asarray(new["extpsw_mie"].values[4, 0, 0, i_re], dtype=np.float64)[sel]
    mono = np.array([mie_cross_sections_um2(n_re, 0.0, float(x), SW05_UM)["ext"] for x in r])
    avg18 = np.array([mode_averaged_cross_sections_um2(n_re, 0.0, float(x), 1.8, SW05_UM)["ext"] for x in r])
    avg16 = np.array([mode_averaged_cross_sections_um2(n_re, 0.0, float(x), 1.6, SW05_UM)["ext"] for x in r])
    new.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.loglog(r, mono, color=NCAR["gray"], ls=":", label="monodisperse Mie")
    ax1.loglog(r, avg16, color=NCAR["aqua"], label="mode integral  $\\sigma_g$=1.6")
    ax1.loglog(r, avg18, color=NCAR["green"], label="mode integral  $\\sigma_g$=1.8")
    ax1.loglog(r, lut, color=NCAR["ncar_blue"], ls="none", marker="o", ms=4, label="SARB LUT")
    ax1.set_xlabel("median wet radius ($\\mu$m)"); ax1.set_ylabel("per-particle ext ($\\mu$m$^2$)")
    ax1.set_title("(a) cross section at SW05 ($n$=1.5)"); ax1.legend(frameon=True, fontsize=9)

    ax2.semilogx(r, mono / (np.pi * r ** 2), color=NCAR["gray"], ls=":", label="monodisperse")
    ax2.semilogx(r, avg16 / (np.pi * r ** 2), color=NCAR["aqua"], label="$\\sigma_g$=1.6")
    ax2.semilogx(r, avg18 / (np.pi * r ** 2), color=NCAR["green"], label="$\\sigma_g$=1.8")
    ax2.axhline(2.0, color=NCAR["gray"], ls="--", lw=1.0)
    for sg, c in ((1.6, NCAR["aqua"]), (1.8, NCAR["green"])):
        ax2.axhline(2.0 * np.exp(2 * np.log(sg) ** 2), color=c, ls=":", lw=1.0)
    ax2.set_xlabel("median wet radius ($\\mu$m)"); ax2.set_ylabel("effective $Q_{ext}$ = ext/$\\pi r^2$")
    ax2.set_title("(b) mode integral $\\approx 2\\,e^{2\\ln^2\\sigma_g}$"); ax2.legend(frameon=True, fontsize=9)
    fig.suptitle("Number-averaged (mode-integrated) cross section — the quantity $\\tau=\\sigma N$ requires",
                 fontweight="bold")
    save(fig, outdir, "mode_integrated_optics")


# --------------------------------------------------------------------------- #
# Spectral band dependence of the radiative properties (MEE, SSA, asymmetry)
# --------------------------------------------------------------------------- #
_SPECIES_FILE = {"SU": "SU.v1_3", "OC": "OC.v1_3", "BC": "BC.v1_3",
                 "DU": "DU.v15_3", "SS": "SS.v3_3", "NI": "NI.v2_5"}
_SPECIES_CACHE = {}


def _species_index(optics_dir, typ, wl_um):
    if typ not in _SPECIES_CACHE:
        ds = xr.open_dataset(os.path.join(optics_dir, "..", "MERRA2", "optics_%s.nc" % _SPECIES_FILE[typ]))
        lam = np.asarray(ds["lambda"].values, dtype=np.float64) * 1e6
        nr = np.asarray(ds["refreal"].values, dtype=np.float64)
        ni = np.abs(np.asarray(ds["refimag"].values, dtype=np.float64))
        if nr.ndim == 3:
            nr, ni = nr[0, 0], ni[0, 0]
        _SPECIES_CACHE[typ] = (lam, nr, ni)
        ds.close()
    lam, nr, ni = _SPECIES_CACHE[typ]
    return float(np.interp(wl_um, lam, nr)), float(np.interp(wl_um, lam, ni))


def _band_midpoints(optics_dir):
    ds = xr.open_dataset(os.path.join(optics_dir, "LFL_bands.nc"))
    sw = np.asarray(ds["LFL_SW_bands"].values, dtype=np.float64)
    lw = np.asarray(ds["LFL_LW_bands"].values, dtype=np.float64)
    ds.close()
    return np.concatenate([0.5 * (sw[:-1] + sw[1:]), 0.5 * (lw[:-1] + lw[1:])])


def fig_spectral_radiative(outdir, optics_dir):
    mids = _band_midpoints(optics_dir)
    # (type, label, density g/cm3, representative radius um, sigma_g, color)
    cases = [
        ("SU", "sulfate", 1.7, 0.10, 1.8, NCAR["ncar_blue"]),
        ("OC", "organic", 1.8, 0.10, 1.8, NCAR["green"]),
        ("BC", "black carbon", 1.0, 0.05, 1.6, NCAR["gray"]),
        ("DU", "dust", 2.6, 1.40, 1.0, NCAR["orange"]),
        ("SS", "sea salt", 2.2, 1.00, 1.0, NCAR["aqua"]),
    ]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))
    for typ, label, rho, r, sg, color in cases:
        mee, ssa, asm = [], [], []
        mean_vol_um3 = (4.0 / 3.0) * np.pi * r ** 3 * np.exp(4.5 * np.log(sg) ** 2)
        mass_g = rho * mean_vol_um3 * 1e-12            # g/cm3 * (um3 -> cm3) -> g per particle
        for wl in mids:
            nr, ni = _species_index(optics_dir, typ, float(wl))
            cs = (mie_cross_sections_um2(nr, ni, r, float(wl)) if sg <= 1.0
                  else mode_averaged_cross_sections_um2(nr, ni, r, sg, float(wl)))
            mee.append((cs["ext"] * 1e-12) / mass_g)   # um2 -> m2, per g  => m2/g
            ssa.append(cs["sca"] / cs["ext"] if cs["ext"] > 0 else np.nan)
            asm.append(cs["asymmetry"])
        ax1.plot(mids, mee, color=color, marker="o", ms=3, label=label)
        ax2.plot(mids, ssa, color=color, marker="o", ms=3, label=label)
        ax3.plot(mids, asm, color=color, marker="o", ms=3, label=label)
    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log")
        ax.axvspan(4.0, mids.max() * 1.1, color=NCAR["light_gray"] if "light_gray" in NCAR else "0.92", alpha=0.5)
        ax.axvline(4.0, color=NCAR["gray"], ls=":", lw=1.0)
        ax.set_xlabel("wavelength ($\\mu$m)")
    ax1.set_yscale("log"); ax1.set_ylabel("mass ext. efficiency (m$^2$/g)"); ax1.set_title("(a) extinction")
    ax2.set_ylabel("single-scattering albedo"); ax2.set_title("(b) SSA"); ax2.set_ylim(0, 1.02)
    ax3.set_ylabel("asymmetry parameter $g$"); ax3.set_title("(c) asymmetry"); ax3.set_ylim(0, 1.0)
    ax1.legend(frameon=True, fontsize=9, loc="lower left")
    fig.suptitle("Spectral radiative properties across the SW (white) and LW (shaded) bands",
                 fontweight="bold")
    save(fig, outdir, "spectral_radiative_properties")


# --------------------------------------------------------------------------- #
# AOD by component (where the optical depth comes from)
# --------------------------------------------------------------------------- #
def fig_aod_components(outdir):
    comp = [("a1", 0.02440), ("a2", 0.00000), ("a3", 0.00453), ("a4", 0.00074),
            ("du1", 0.00963), ("du2", 0.02010), ("du3", 0.01210), ("du4", 0.00123), ("du5", 0.00013),
            ("ss1", 0.00108), ("ss2", 0.02044), ("ss3", 0.07324), ("ss4", 0.02087), ("ss5", 0.00072)]
    names = [c[0] for c in comp]; vals = [c[1] for c in comp]
    colors = [NCAR["purple"] if n[0] == "a" else NCAR["orange"] if n[0] == "d" else NCAR["aqua"] for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(np.arange(len(names)), vals, color=colors)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("global-mean column AOD (SW05)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NCAR["purple"], NCAR["orange"], NCAR["aqua"])]
    ax.legend(handles, ["internal modes", "dust bins", "sea-salt bins"], frameon=True)
    ax.set_title("Column AOD by component  (total 0.189, GEOS-IT 2008-07-01T00)", fontweight="bold")
    save(fig, outdir, "aod_components")


# --------------------------------------------------------------------------- #
# Global AOD maps: MAM total vs reference (shared scale)
# --------------------------------------------------------------------------- #
def fig_aod_maps(outdir):
    try:
        import cartopy.crs as ccrs
        from cartopy.util import add_cyclic_point
    except Exception as exc:
        print("  skipping maps (cartopy import failed): %s" % exc)
        return
    D = os.path.expanduser("~/Data/GEOSIT_MAM/2008/07")
    PRE = "GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294"
    panels = [
        (os.path.join(D, "%s.AER_SW05.2008-07-01T0000.V01.nc4" % PRE), "MAM (bin-resolved)"),
        (os.path.expanduser("~/Data/GEOSIT/2008/07/"
         "GEOS.it.asm.aer_inst_3hr_glo_L288x180_v24.GEOS5294.AER_SW05.2008-07-01T0000.V01.nc4"),
         "alpha_4 reference"),
    ]
    if not all(os.path.exists(p) for p, _ in panels):
        print("  skipping maps (slice/reference missing)")
        return
    levels = np.arange(0, 1.0001, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), subplot_kw={"projection": ccrs.PlateCarree()})
    cf = None
    for ax, (path, title) in zip(axes, panels):
        ds = xr.open_dataset(path)
        col = ds["Extinction_Column_Optical_Depth"]
        if "time" in col.dims:
            col = col.isel(time=0)
        lat = col["lat"].values; lon = col["lon"].values
        vals = col.values
        w = np.cos(np.deg2rad(lat))
        mean = float((vals * w[:, None]).sum() / (w.sum() * vals.shape[1]))
        vals_c, lon_c = add_cyclic_point(vals, coord=lon)
        ax.set_facecolor("gray")
        cf = ax.contourf(*np.meshgrid(lon_c, lat), vals_c, levels, cmap="turbo",
                         extend="max", transform=ccrs.PlateCarree())
        try:
            ax.coastlines(linewidth=0.5)
        except Exception:
            pass
        ax.set_title("%s    area-mean %.3f" % (title, mean))
        ds.close()
    cbar = fig.colorbar(cf, ax=axes, orientation="horizontal", pad=0.06, shrink=0.6, aspect=40)
    cbar.set_label("column AOD (SW05)")
    fig.suptitle("Global column AOD — GEOS-IT 2008-07-01T00", fontweight="bold")
    save(fig, outdir, "aod_maps")


# --------------------------------------------------------------------------- #
# Verification scorecard (live verify_physics)
# --------------------------------------------------------------------------- #
def fig_scorecard(outdir, config):
    order = list(vp.STAGES)
    titles = {k: vp.STAGES[k][0] for k in order}
    tally = {k: {} for k in order}
    total = {"PASS": 0, "FAIL": 0, "FINDING": 0, "WARN": 0}
    for letter in order:
        for res in vp.STAGES[letter][1](config):
            tally[res.stage][res.status] = tally[res.stage].get(res.status, 0) + 1
            total[res.status] = total.get(res.status, 0) + 1
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ys = np.arange(len(order))[::-1]
    for y, letter in zip(ys, order):
        left = 0
        for status in ("PASS", "FINDING", "WARN", "FAIL"):
            n = tally[letter].get(status, 0)
            if n:
                ax.barh(y, n, left=left, color=STATUS_COLOR[status], edgecolor="white")
                ax.text(left + n / 2, y, str(n), ha="center", va="center", color="white", fontsize=9)
                left += n
        ax.text(-0.4, y, "%s  %s" % (letter, titles[letter]), ha="right", va="center", fontsize=10)
    ax.set_yticks([]); ax.set_xlabel("number of checks")
    ax.set_xlim(0, max(sum(t.values()) for t in tally.values()) + 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLOR[s]) for s in ("PASS", "FINDING")]
    ax.legend(handles, ["PASS", "FINDING"], ncol=2, loc="lower right", frameon=True)
    ax.set_title("Physics verification — verify_physics.py\n%d PASS · %d FAIL · %d FINDING"
                 % (total["PASS"], total["FAIL"], total["FINDING"]), fontweight="bold")
    fig.subplots_adjust(left=0.34)
    save(fig, outdir, "verification_scorecard")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default=os.path.expanduser("~/Plots/verification"))
    parser.add_argument("--aerosol", default="aerosol.yaml")
    parser.add_argument("--optics-dir", default=os.path.expanduser("~/Data/Optics/SARB"))
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    apply_style()
    config = load_config(args.aerosol)

    print("figures -> %s" % args.outdir)
    fig_size_distributions(args.outdir, config)
    fig_hygroscopicity(args.outdir)
    fig_kohler(args.outdir)
    fig_refractive_mixing(args.outdir)
    fig_mie_efficiency(args.outdir)
    fig_mode_integrated(args.outdir, args.optics_dir)
    fig_spectral_radiative(args.outdir, args.optics_dir)
    fig_aod_components(args.outdir)
    fig_aod_maps(args.outdir)
    fig_scorecard(args.outdir, config)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
