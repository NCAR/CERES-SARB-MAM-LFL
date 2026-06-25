"""Physics figures for the MAM internal-mixing / radiative-property chain.

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
from matplotlib.ticker import FuncFormatter, NullFormatter

from mode_config import load_config, resolved_allocations
from mode_physics import kohler_wet_radius_um, mix_mode_state, _mixed_hygroscopicity
from mode_optics import _mode_species, _species_info, _type_key
from mie_sphere import mie_efficiencies, mie_cross_sections_um2
from mie_lognormal import mode_averaged_cross_sections_um2

NCAR = {
    "space": "#011837", "dark_blue": "#00357A", "ncar_blue": "#0A5DDA",
    "aqua": "#00A2B4", "orange": "#FF8C00", "yellow": "#FFDD31",
    "gray": "#58595B", "red": "#D62839", "green": "#2E8B57", "purple": "#7B68EE",
}
PALETTE = [NCAR[c] for c in ("ncar_blue", "aqua", "orange", "purple", "green", "red", "dark_blue")]

# Canonical color per internal mode (used wherever lines/markers are modes).
MODE_COLOR = {
    "Accumulation": NCAR["ncar_blue"],
    "Aitken": NCAR["aqua"],
    "Coarse": NCAR["purple"],
    "Primary Carbon": "#3F3F3F",  # dark grey, matching the Black Carbon species color
}
# Semantic color per aerosol species / family (used wherever they are species).
SPECIES_COLOR = {
    "Sulfate": "#2166AC",       # blue
    "Black Carbon": "#3F3F3F",  # dark grey
    "Organic": "#5AAE61",       # green
    "Dust": "#CD853F",          # desert brown
    "Sea Salt": "#006D77",      # teal
    "Nitrate": "#9467BD",       # violet
    "Water": "#74A9CF",         # light blue
}
SW05_UM = 0.54625


# --------------------------------------------------------------------------- #
# Prescribed modal mixtures: per-mode composition (for pies) and volume-weighted
# hygroscopicity (for the Köhler / growth curves). Built from REP_MASS x the
# prescribed allocation -- the same representative composition the optics
# figures use -- so the curves and pies describe the actual modal mixtures.
# --------------------------------------------------------------------------- #
_TYPE_FAMILY = {"SU": "Sulfate", "NI": "Nitrate", "POM": "Organic", "SOA": "Organic",
                "BC": "Black Carbon", "DU": "Dust", "SS": "Sea Salt"}
_INTERNAL_MODES = [("a1", "Accumulation"), ("a2", "Aitken"),
                   ("a3", "Coarse"), ("a4", "Primary Carbon")]
_PIE_FAMILY_ORDER = ["Sulfate", "Organic", "Black Carbon", "Nitrate", "Dust", "Sea Salt"]


def _mode_mix_q(config, mode):
    """Representative dry-mass dict (REP_MASS x allocation) for one mode, keeping
    only the species that carry mass -- the prescribed internal mixture."""
    alloc = resolved_allocations(config, "MAM4")
    q = {s: _REP_MASS.get(s, 0.0) * float(alloc[s].get(mode, 0.0))
         for s in _mode_species(config, "MAM4", mode)}
    return {s: m for s, m in q.items() if m > 0.0}


def _mode_mass_composition(config, mode):
    """Prescribed dry-mass fractions of a mode aggregated by species family."""
    fam = {}
    for s, m in _mode_mix_q(config, mode).items():
        key = _TYPE_FAMILY[_type_key(s)]
        fam[key] = fam.get(key, 0.0) + m
    total = sum(fam.values())
    return {f: v / total for f, v in fam.items()} if total > 0.0 else {}


def _mode_mixed_B(config, mode, rh):
    """Volume-weighted hygroscopic growth coefficient B(RH) of a mode's
    prescribed mixture (same rule as the production mix_mode_state)."""
    q = {s: np.array([m], dtype=np.float32) for s, m in _mode_mix_q(config, mode).items()}
    info = _species_info(config, list(q))
    return _mixed_hygroscopicity(info, q, np.asarray(rh, dtype=np.float32))


def _species_B(config, tkey, rh):
    """Pure-species B(RH) from a Type's hygroscopicity polynomial."""
    b = list(config["Types"][tkey]["hygroscopicity"]) + [0.0, 0.0]
    rh = np.asarray(rh, dtype=np.float64)
    return float(b[0]) + float(b[1]) * rh + float(b[2]) * rh ** 2


def _draw_mode_pies(fig, pie_axes, config, modes=_INTERNAL_MODES,
                    legend_title="Prescribed Mode Composition (Dry Mass)"):
    """Draw one prescribed dry-mass composition pie per mode into ``pie_axes``
    and add a shared species-family colour key beneath them."""
    seen = []
    for ax, (mkey, mname) in zip(pie_axes, modes):
        comp = _mode_mass_composition(config, mkey)
        ax.set(frame_on=False); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if not comp:
            continue
        fams = [f for f in _PIE_FAMILY_ORDER if f in comp]
        ax.pie([comp[f] for f in fams], colors=[SPECIES_COLOR[f] for f in fams],
               startangle=90, counterclock=False,
               wedgeprops=dict(linewidth=0, edgecolor="none"))
        ax.set_title(mname, fontsize=10, pad=1)
        ax.set_aspect("equal")
        seen += [f for f in fams if f not in seen]
    order = [f for f in _PIE_FAMILY_ORDER if f in seen]
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPECIES_COLOR[f]) for f in order]
    fig.legend(handles, order, loc="lower center", ncol=len(order) or 1, frameon=True,
               bbox_to_anchor=(0.5, -0.02), title=legend_title)


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


def _decimal(v, _pos):
    if v <= 0:
        return ""
    return ("%.10f" % v).rstrip("0").rstrip(".")


DECIMAL = FuncFormatter(_decimal)


def _logticks(ax, xticks=None, yticks=None):
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(DECIMAL)
        ax.xaxis.set_minor_formatter(NullFormatter())
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(DECIMAL)
        ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight", pad_inches=0.15, dpi=300)
    plt.close(fig)
    print("  %s.{pdf,png}" % name)


def add_coastlines(ax, linewidth=0.4):
    """Draw coastlines robustly. ``ax.coastlines()`` routes through cartopy's
    downloader, which fails in offline/locked-down environments even when the
    Natural Earth shapefiles are already cached; read the cached shapefile
    directly first, fall back to the downloader, then degrade gracefully."""
    try:
        import cartopy.crs as ccrs
        from cartopy import config as _cconfig
        from cartopy.io.shapereader import Reader
        from cartopy.feature import ShapelyFeature
        base = os.path.join(_cconfig["data_dir"], "shapefiles",
                            "natural_earth", "physical")
        for res in ("110m", "50m"):
            shp = os.path.join(base, "ne_%s_coastline.shp" % res)
            if os.path.exists(shp):
                feat = ShapelyFeature(Reader(shp).geometries(), ccrs.PlateCarree(),
                                      edgecolor="k", facecolor="none", linewidth=linewidth)
                ax.add_feature(feat)
                return
        ax.coastlines(linewidth=linewidth)
    except Exception as exc:
        print("  (coastlines unavailable: %s; drawing maps without them)" % exc)


# --------------------------------------------------------------------------- #
def fig_size_distributions(outdir, config):
    modes = config["Schemes"]["MAM4"]["modes"]
    r = np.geomspace(2e-3, 30.0, 800)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key in ("a1", "a2", "a3", "a4"):
        name = modes[key]["name"]
        rg = float(modes[key]["dry_radius_um"]); sg = float(modes[key]["sigma_g"])
        ls = np.log(sg)
        pdf = np.exp(-((np.log(r) - np.log(rg)) ** 2) / (2 * ls ** 2)) / (ls * np.sqrt(2 * np.pi))
        lw = 1.6 if name == "Primary Carbon" else 2.0  # thinner so it doesn't hide Accumulation
        ax.plot(r, pdf / pdf.max(), color=MODE_COLOR[name], lw=lw, label=name)
    du = [float(modes["du%d" % i]["dry_radius_um"]) for i in range(1, 6)]
    ss = [float(modes["ss%d" % i]["dry_radius_um"]) for i in range(1, 6)]
    ax.vlines(du, 0, 0.85, color=SPECIES_COLOR["Dust"], lw=2.2, label="Dust Bins")
    ax.vlines(ss, 0, 0.7, color=SPECIES_COLOR["Sea Salt"], lw=2.2, ls=(0, (4, 2)), label="Sea-Salt Bins")
    ax.set_xscale("log")
    _logticks(ax, xticks=[0.01, 0.1, 1, 10])
    ax.set_xlim(2e-3, 30); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Dry Radius (μm)"); ax.set_ylabel("$dN/d\\ln r$ (Peak-Normalized)")
    ax.set_title("Size Distributions")
    ax.legend(frameon=True, loc="upper right")
    save(fig, outdir, "size_distributions")


def fig_hygroscopicity(outdir, config):
    rh = np.linspace(0.0, 1.0, 101)
    fig = plt.figure(figsize=(8.5, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.3], hspace=0.55, wspace=0.4)
    ax = fig.add_subplot(gs[0, :])
    pie_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for mkey, mname in _INTERNAL_MODES:
        ax.plot(rh, _mode_mixed_B(config, mkey, rh), color=MODE_COLOR[mname], label=mname)
    for tkey, fam in [("SU", "Sulfate"), ("SS", "Sea Salt"),
                      ("POM", "Organic"), ("BC", "Black Carbon")]:
        ax.plot(rh, _species_B(config, tkey, rh), color=SPECIES_COLOR[fam],
                lw=1.0, ls=(0, (4, 2)), alpha=0.55, label="%s (Pure)" % fam)
    ax.set_xlabel("Relative Humidity"); ax.set_ylabel("Hygroscopic Growth Coefficient $B$")
    ax.set_title("Hygroscopicity of the Prescribed Modal Mixtures")
    ax.legend(frameon=True, fontsize=8, ncol=2); ax.set_ylim(bottom=0)
    _draw_mode_pies(fig, pie_axes, config)
    save(fig, outdir, "hygroscopicity")


def fig_kohler(outdir, config):
    rh = np.linspace(0.05, 0.985, 160).astype(np.float32)
    T = np.full_like(rh, 290.0)
    modes = config["Schemes"]["MAM4"]["modes"]
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.3], hspace=0.5, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0:2]); ax2 = fig.add_subplot(gs[0, 2:4])
    pie_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    # Primary curves: the prescribed modal mixtures (a1-a4).
    for mkey, mname in _INTERNAL_MODES:
        r_d = float(modes[mkey]["dry_radius_um"])
        B = _mode_mixed_B(config, mkey, rh)
        w = kohler_wet_radius_um(r_d, B, rh, T)
        ax1.plot(rh, w / r_d, color=MODE_COLOR[mname], label=mname)
        ax2.plot(rh, w, color=MODE_COLOR[mname], label=mname)
    # Faint pure end-member reference (growth factor at an accumulation-size r_d).
    for tkey, fam in [("SU", "Sulfate"), ("SS", "Sea Salt"), ("BC", "Black Carbon")]:
        w = kohler_wet_radius_um(0.055, _species_B(config, tkey, rh), rh, T)
        ax1.plot(rh, w / 0.055, color=SPECIES_COLOR[fam], lw=1.0, ls=(0, (4, 2)),
                 alpha=0.55, label="%s (Pure)" % fam)
    ax1.set_xlabel("Relative Humidity"); ax1.set_ylabel("$r_w / r_d$")
    ax1.set_title("Growth Factor")
    ax1.legend(frameon=True, fontsize=8, ncol=2, loc="upper left")
    ax2.set_yscale("log"); _logticks(ax2, yticks=[0.01, 0.1, 1])
    ax2.set_xlabel("Relative Humidity"); ax2.set_ylabel("$r_w$ (μm)")
    ax2.set_title("Wet Radius"); ax2.legend(frameon=True, fontsize=8, title="Mode")
    _draw_mode_pies(fig, pie_axes, config)
    fig.suptitle("Köhler Growth of the Prescribed Modal Mixtures", fontweight="bold")
    save(fig, outdir, "kohler_growth")


def fig_refractive_mixing(outdir):
    rh = np.linspace(0.0, 0.97, 90).astype(np.float32)
    info = {"BC": {"density": 1.0, "hygroscopicity": [0.3, 0.0, 0.0]}}
    refr = {"BC": (1.95, 0.79), "WAT": (1.34, 0.0)}
    q = {"BC": np.full(rh.shape, 5e-9, np.float32)}
    st = mix_mode_state(info, q, refr, rh, np.full_like(rh, 290.0), dry_radius_um=0.055)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(rh, st["n_re"], color=NCAR["ncar_blue"], label="Mixture")
    ax1.axhline(1.95, color=SPECIES_COLOR["Black Carbon"], ls="--", lw=1.2, label="Dry BC")
    ax1.axhline(1.34, color=SPECIES_COLOR["Water"], ls=":", lw=1.6, label="Water")
    ax1.set_xlabel("Relative Humidity"); ax1.set_ylabel("$n_r$")
    ax1.set_title("Real Part"); ax1.legend(frameon=True, loc="lower left")
    ax2.plot(rh, st["n_im"], color=NCAR["ncar_blue"], label="Mixture")
    ax2.axhline(0.79, color=SPECIES_COLOR["Black Carbon"], ls="--", lw=1.2, label="Dry BC")
    ax2.set_xlabel("Relative Humidity"); ax2.set_ylabel("$n_i$")
    ax2.set_title("Imaginary Part"); ax2.legend(frameon=True, loc="lower left")
    fig.suptitle("Refractive-Index Mixing", fontweight="bold")
    save(fig, outdir, "refractive_mixing")


def fig_mie_efficiency(outdir):
    radii = np.geomspace(0.005, 30.0, 240)
    x = 2 * np.pi * radii / 0.55
    qext = np.array([mie_efficiencies(1.5, 0.0, float(r), 0.55)["q_ext"] for r in radii])
    qabs = np.array([mie_efficiencies(1.5, 0.01, float(r), 0.55)["q_abs"] for r in radii])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(x, qext, color=NCAR["ncar_blue"], label="$Q_\\mathrm{ext}$")
    ax.plot(x, qabs, color=NCAR["red"], label="$Q_\\mathrm{abs}$")
    ax.axhline(2.0, color=NCAR["gray"], ls="--", lw=1.2, label="$Q=2$ (Geometric Limit)")
    ax.set_xscale("log")
    _logticks(ax, xticks=[0.1, 1, 10, 100])
    ax.set_xlabel("Size Parameter, $x$"); ax.set_ylabel("$Q$")
    ax.set_title("Mie Efficiencies")
    ax.legend(frameon=True, loc="upper left"); ax.set_ylim(0, 4.6)
    save(fig, outdir, "mie_efficiency")


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
    ax1.loglog(r, mono, color=NCAR["gray"], ls=":", label="Monodisperse")
    ax1.loglog(r, avg16, color=NCAR["aqua"], label="$\\sigma_g = 1.6$")
    ax1.loglog(r, avg18, color=NCAR["green"], label="$\\sigma_g = 1.8$")
    ax1.loglog(r, lut, color=NCAR["ncar_blue"], ls="none", marker="o", ms=4, label="LUT")
    _logticks(ax1, xticks=[0.1, 1, 10], yticks=[0.01, 1, 100])
    ax1.set_xlabel("Radius (μm)"); ax1.set_ylabel("$\\sigma_\\mathrm{ext}$ (μm$^2$)")
    ax1.set_title("Cross Section"); ax1.legend(frameon=True)

    ax2.semilogx(r, mono / (np.pi * r ** 2), color=NCAR["gray"], ls=":", label="Monodisperse")
    ax2.semilogx(r, avg16 / (np.pi * r ** 2), color=NCAR["aqua"], label="$\\sigma_g = 1.6$")
    ax2.semilogx(r, avg18 / (np.pi * r ** 2), color=NCAR["green"], label="$\\sigma_g = 1.8$")
    ax2.axhline(2.0, color=NCAR["gray"], ls="--", lw=1.2, label="$Q=2$ (Geometric Limit)")
    _logticks(ax2, xticks=[0.1, 1, 10])
    ax2.set_xlabel("Radius (μm)"); ax2.set_ylabel("$Q_\\mathrm{ext}$")
    ax2.set_title("Effective Efficiency"); ax2.legend(frameon=True, loc="lower left")
    fig.suptitle("Mode-Integrated Cross Section", fontweight="bold")
    save(fig, outdir, "mode_integrated_optics")


# --------------------------------------------------------------------------- #
# Spectral band dependence: internally-mixed modes vs external dust/sea-salt
# --------------------------------------------------------------------------- #
_REP_MASS = {"SO4": 2.948e-6, "OCPHILIC": 4.369e-6, "OCPHOBIC": 9.967e-7,
             "BCPHILIC": 3.989e-7, "BCPHOBIC": 1.47e-7,
             "NO3AN1": 6.013e-7, "NO3AN2": 9.862e-7, "NO3AN3": 2.941e-9}
_TYPE_FILE = {"SU": "SU.v1_3", "POM": "OC.v1_3", "SOA": "OC.v1_3", "BC": "BC.v1_3",
              "DU": "DU.v15_3", "SS": "SS.v3_3", "NI": "NI.v2_5"}
_IDX_CACHE = {}
_REP_RH = 0.70


def _type_index(optics_dir, type_key, wl_um):
    if type_key not in _IDX_CACHE:
        ds = xr.open_dataset(os.path.join(optics_dir, "..", "MERRA2", "optics_%s.nc" % _TYPE_FILE[type_key]))
        lam = np.asarray(ds["lambda"].values, dtype=np.float64) * 1e6
        nr = np.asarray(ds["refreal"].values, dtype=np.float64)
        ni = np.abs(np.asarray(ds["refimag"].values, dtype=np.float64))
        if nr.ndim == 3:
            nr, ni = nr[0, 0], ni[0, 0]
        _IDX_CACHE[type_key] = (lam, nr, ni)
        ds.close()
    lam, nr, ni = _IDX_CACHE[type_key]
    return float(np.interp(wl_um, lam, nr)), float(np.interp(wl_um, lam, ni))


def _water_index(optics_dir, wl_um):
    if "WAT" not in _IDX_CACHE:
        ds = xr.open_dataset(os.path.join(optics_dir, "..", "MERRA2", "optics_WAT.nc"))
        _IDX_CACHE["WAT"] = (np.asarray(ds["wavelength1"].values, dtype=np.float64),
                             np.asarray(ds["watern"].values, dtype=np.float64),
                             np.asarray(ds["wateri"].values, dtype=np.float64))
        ds.close()
    wl, nr, ni = _IDX_CACHE["WAT"]
    return float(np.interp(wl_um, wl, nr)), float(np.interp(wl_um, wl, ni))


def _band_midpoints(optics_dir):
    ds = xr.open_dataset(os.path.join(optics_dir, "LFL_bands.nc"))
    sw = np.asarray(ds["LFL_SW_bands"].values, dtype=np.float64)
    lw = np.asarray(ds["LFL_LW_bands"].values, dtype=np.float64)
    ds.close()
    return np.concatenate([0.5 * (sw[:-1] + sw[1:]), 0.5 * (lw[:-1] + lw[1:])])


def fig_spectral_radiative(outdir, optics_dir, config):
    mids = _band_midpoints(optics_dir)
    modes = config["Schemes"]["MAM4"]["modes"]
    alloc = resolved_allocations(config, "MAM4")
    comps = [
        ("a1", modes["a1"]["name"], MODE_COLOR[modes["a1"]["name"]], "-", False),
        ("a2", modes["a2"]["name"], MODE_COLOR[modes["a2"]["name"]], "-", False),
        ("a3", modes["a3"]["name"], MODE_COLOR[modes["a3"]["name"]], "-", False),
        ("a4", modes["a4"]["name"], MODE_COLOR[modes["a4"]["name"]], "-", False),
        ("du2", "Dust", SPECIES_COLOR["Dust"], "--", True),
        ("ss3", "Sea Salt", SPECIES_COLOR["Sea Salt"], "--", True),
    ]
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 12, height_ratios=[3.0, 1.3], hspace=0.5, wspace=1.4)
    ax1 = fig.add_subplot(gs[0, 0:4]); ax2 = fig.add_subplot(gs[0, 4:8]); ax3 = fig.add_subplot(gs[0, 8:12])
    pie_axes = [fig.add_subplot(gs[1, 3 * i:3 * i + 3]) for i in range(4)]
    rh = np.array([_REP_RH], dtype=np.float32)
    temp = np.array([283.0], dtype=np.float32)
    for key, label, color, lsty, external in comps:
        species = _mode_species(config, "MAM4", key)
        info = _species_info(config, species)
        r_d = float(modes[key]["dry_radius_um"]); sg = float(modes[key]["sigma_g"])
        if external:
            q = {s: np.array([1.0], dtype=np.float32) for s in species}
        else:
            q = {s: np.array([_REP_MASS.get(s, 0.0) * float(alloc[s].get(key, 0.0))], dtype=np.float32)
                 for s in species}
        rho_eff = (sum(float(q[s][0]) for s in species)
                   / max(sum(float(q[s][0]) / info[s]["density"] for s in species), 1e-30))
        mass_g = rho_eff * (4.0 / 3.0) * np.pi * r_d ** 3 * np.exp(4.5 * np.log(sg) ** 2) * 1e-12
        mee, ssa, asm = [], [], []
        for wl in mids:
            refractive = {s: _type_index(optics_dir, _type_key(s), float(wl)) for s in species}
            refractive["WAT"] = _water_index(optics_dir, float(wl))
            st = mix_mode_state(info, q, refractive, rh, temp, dry_radius_um=r_d)
            nr, ni, rw = float(st["n_re"][0]), float(st["n_im"][0]), float(st["r_w_um"][0])
            cs = (mie_cross_sections_um2(nr, ni, rw, float(wl)) if sg <= 1.0
                  else mode_averaged_cross_sections_um2(nr, ni, rw, sg, float(wl)))
            mee.append((cs["ext"] * 1e-12) / mass_g)
            ssa.append(cs["sca"] / cs["ext"] if cs["ext"] > 0 else np.nan)
            asm.append(cs["asymmetry"])
        kw = dict(color=color, ls=lsty, marker="o", ms=3, label=label)
        ax1.plot(mids, mee, **kw); ax2.plot(mids, ssa, **kw); ax3.plot(mids, asm, **kw)
    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log")
        ax.axvspan(4.0, mids.max() * 1.1, color="0.92")
        _logticks(ax, xticks=[0.3, 1, 3, 10, 30])
        ax.set_xlim(mids.min() * 0.9, mids.max() * 1.1)
        ax.set_xlabel("$\\lambda$ (μm)")
    ax1.set_yscale("log"); _logticks(ax1, xticks=[0.3, 1, 3, 10, 30], yticks=[0.001, 0.01, 0.1, 1, 10])
    ax1.set_ylabel("$k_\\mathrm{ext}$ (m$^2$ g$^{-1}$)"); ax1.set_title("Extinction")
    ax2.set_ylabel("$\\omega_0$"); ax2.set_title("Single-Scattering Albedo"); ax2.set_ylim(0, 1.02)
    ax3.set_ylabel("$g$"); ax3.set_title("Asymmetry"); ax3.set_ylim(0, 1.0)
    ax3.text(0.02, 0.96, "SW", transform=ax3.transAxes, ha="left", va="top",
             color="0.45", fontsize=10, fontweight="bold")
    ax3.text(0.985, 0.96, "LW", transform=ax3.transAxes, ha="right", va="top",
             color="0.45", fontsize=10, fontweight="bold")
    ax1.legend(frameon=True, loc="lower left", title="Modes (Solid) / Species (Dashed)")
    _draw_mode_pies(fig, pie_axes, config)
    fig.suptitle("Spectral Radiative Properties (RH = 70%, 283 K)", fontweight="bold")
    save(fig, outdir, "spectral_radiative_properties")


# --------------------------------------------------------------------------- #
# Internal-mixture response: hydration (vs RH) and composition (vs species mix)
# --------------------------------------------------------------------------- #
def _mixture_optics_point(optics_dir, info, species, q, r_d, sg, rh, temp, wl, n_quad=128):
    """Run the production mixing -> Köhler -> Mie chain for one (composition, RH)
    point and return intensive radiative properties at wavelength ``wl``."""
    refractive = {s: _type_index(optics_dir, _type_key(s), wl) for s in species}
    refractive["WAT"] = _water_index(optics_dir, wl)
    rh_a = np.array([rh], dtype=np.float32)
    t_a = np.array([temp], dtype=np.float32)
    st = mix_mode_state(info, q, refractive, rh_a, t_a, dry_radius_um=r_d)
    nr, ni, rw = float(st["n_re"][0]), float(st["n_im"][0]), float(st["r_w_um"][0])
    cs = (mode_averaged_cross_sections_um2(nr, ni, rw, sg, wl, n_quad=n_quad) if sg > 1.0
          else mie_cross_sections_um2(nr, ni, rw, wl))
    masses = [float(q[s][0]) for s in species]
    rho_eff = sum(masses) / max(sum(m / info[s]["density"] for s, m in zip(species, masses)), 1e-30)
    mass_g = rho_eff * (4.0 / 3.0) * np.pi * r_d ** 3 * np.exp(4.5 * np.log(sg) ** 2) * 1e-12
    ext = cs["ext"]
    return {
        "k_ext": ext * 1e-12 / mass_g if mass_g > 0 else np.nan,
        "ssa": cs["sca"] / ext if ext > 0 else np.nan,
        "g": cs["asymmetry"],
        "f_water": max(0.0, 1.0 - (r_d / rw) ** 3),
        "gf": rw / r_d, "n_re": nr, "n_im": ni,
    }


def fig_hydration_radiative(outdir, optics_dir, config):
    modes = config["Schemes"]["MAM4"]["modes"]
    alloc = resolved_allocations(config, "MAM4")
    rh = np.linspace(0.0, 0.98, 40)
    temp = 283.0
    comps = [("a1", MODE_COLOR["Accumulation"]), ("a2", MODE_COLOR["Aitken"]),
             ("a3", MODE_COLOR["Coarse"]), ("a4", MODE_COLOR["Primary Carbon"])]
    fig = plt.figure(figsize=(11, 10.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[3.0, 3.0, 1.35], hspace=0.33, wspace=0.5)
    axU = fig.add_subplot(gs[0, 0:2]); axE = fig.add_subplot(gs[0, 2:4])
    axS = fig.add_subplot(gs[1, 0:2]); axG = fig.add_subplot(gs[1, 2:4])
    pie_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]
    for key, color in comps:
        species = _mode_species(config, "MAM4", key)
        info = _species_info(config, species)
        r_d = float(modes[key]["dry_radius_um"]); sg = float(modes[key]["sigma_g"])
        q = {s: np.array([_REP_MASS.get(s, 0.0) * float(alloc[s].get(key, 0.0))], dtype=np.float32)
             for s in species}
        fw, ke, ss, gg = [], [], [], []
        for r in rh:
            p = _mixture_optics_point(optics_dir, info, species, q, r_d, sg, float(r), temp, SW05_UM)
            fw.append(p["f_water"]); ke.append(p["k_ext"]); ss.append(p["ssa"]); gg.append(p["g"])
        kw = dict(color=color, label=modes[key]["name"], marker="o", ms=3)
        axU.plot(rh, fw, **kw); axE.plot(rh, ke, **kw)
        axS.plot(rh, ss, **kw); axG.plot(rh, gg, **kw)
    for ax in (axU, axE, axS, axG):
        ax.set_xlabel("Relative Humidity"); ax.set_xlim(0, 1)
    axU.set_ylabel("Water Volume Fraction"); axU.set_title("Water Uptake"); axU.set_ylim(0, 1)
    axE.set_yscale("log"); _logticks(axE, yticks=[0.1, 1, 10])
    axE.set_ylabel("$k_\\mathrm{ext}$ (m$^2$ g$^{-1}$)"); axE.set_title("Extinction")
    axS.set_ylabel("$\\omega_0$"); axS.set_title("Single-Scattering Albedo"); axS.set_ylim(0, 1.02)
    axG.set_ylabel("$g$"); axG.set_title("Asymmetry"); axG.set_ylim(0, 1.02)
    axU.legend(frameon=True, title="Mode")
    _draw_mode_pies(fig, pie_axes, config)
    fig.suptitle("Hydration Response at 0.55 μm", fontweight="bold")
    save(fig, outdir, "hydration_radiative")


def fig_composition_radiative(outdir, optics_dir, config):
    modes = config["Schemes"]["MAM4"]["modes"]
    r_d = float(modes["a1"]["dry_radius_um"]); sg = float(modes["a1"]["sigma_g"])
    rh, temp = 0.70, 283.0
    f = np.linspace(0.0, 1.0, 31)
    guests = [("BCPHILIC", "Black Carbon", SPECIES_COLOR["Black Carbon"]),
              ("OCPHILIC", "Organic", SPECIES_COLOR["Organic"]),
              ("NO3AN1", "Nitrate", SPECIES_COLOR["Nitrate"])]
    fig, (axE, axS, axG) = plt.subplots(1, 3, figsize=(14, 4.4))
    for guest, label, color in guests:
        species = ["SO4", guest]
        info = _species_info(config, species)
        ke, ss, gg = [], [], []
        for frac in f:
            q = {"SO4": np.array([1.0 - frac], dtype=np.float32),
                 guest: np.array([frac], dtype=np.float32)}
            p = _mixture_optics_point(optics_dir, info, species, q, r_d, sg, rh, temp, SW05_UM)
            ke.append(p["k_ext"]); ss.append(p["ssa"]); gg.append(p["g"])
        kw = dict(color=color, label=label, marker="o", ms=3)
        axE.plot(f, ke, **kw); axS.plot(f, ss, **kw); axG.plot(f, gg, **kw)
    for ax in (axE, axS, axG):
        ax.set_xlabel("Guest Mass Fraction"); ax.set_xlim(0, 1)
    axE.set_ylabel("$k_\\mathrm{ext}$ (m$^2$ g$^{-1}$)"); axE.set_title("Extinction")
    axS.set_ylabel("$\\omega_0$"); axS.set_title("Single-Scattering Albedo"); axS.set_ylim(0, 1.02)
    axG.set_ylabel("$g$"); axG.set_title("Asymmetry")
    axS.legend(frameon=True, title="Guest in Sulfate")
    fig.suptitle(f"Composition Dependence ({modes['a1']['name']} Mode, RH = 70%, 0.55 μm)", fontweight="bold")
    save(fig, outdir, "composition_radiative")


# --------------------------------------------------------------------------- #
def fig_aod_components(outdir, config):
    if not os.path.exists(_mam_component_path("a1")):
        print("  skipping aod_components (MAM component files missing)")
        return
    modes = config["Schemes"]["MAM4"]["modes"]
    names = MAM_COMPONENTS
    vals = [_area_mean(_column_aod(_mam_component_path(c))) for c in names]
    labels = [modes[c]["name"] if c.startswith("a") else c.upper() for c in names]
    fam = {"a": MODE_COLOR["Accumulation"], "d": SPECIES_COLOR["Dust"], "s": SPECIES_COLOR["Sea Salt"]}
    colors = [fam[n[0]] for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(np.arange(len(names)), vals, color=colors)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("MAM4 Component"); ax.set_ylabel("Column AOD $\\tau$")
    handles = [plt.Rectangle((0, 0), 1, 1, color=fam[k]) for k in ("a", "d", "s")]
    ax.legend(handles, ["Internal Modes", "Dust Bins", "Sea-Salt Bins"], frameon=True)
    ax.set_title("Column AOD by Component", fontweight="bold")
    save(fig, outdir, "aod_components")


MAM_COMPONENTS = ["a1", "a2", "a3", "a4",
                  "du1", "du2", "du3", "du4", "du5",
                  "ss1", "ss2", "ss3", "ss4", "ss5"]


def _aod_paths():
    D = os.path.expanduser("~/Data/GEOSIT_MAM/2008/07")
    PRE = "GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294"
    mam = os.path.join(D, "%s.AER_SW05.2008-07-01T0000.V01.nc4" % PRE)
    ref = os.path.expanduser("~/Data/GEOSIT/2008/07/"
        "GEOS.it.asm.aer_inst_3hr_glo_L288x180_v24.GEOS5294.AER_SW05.2008-07-01T0000.V01.nc4")
    return mam, ref


def _mam_component_path(comp):
    D = os.path.expanduser("~/Data/GEOSIT_MAM/2008/07")
    PRE = "GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294"
    return os.path.join(D, "%s.MAM4_%s_SW05.2008-07-01T0000.V01.nc4" % (PRE, comp))


def _mam_total_column():
    """MAM total column AOD = sum of the 14 per-component fields (always
    reflects the current GEOSIT_MAM files, including the capped sea-salt bins)."""
    total = None
    for comp in MAM_COMPONENTS:
        col = _column_aod(_mam_component_path(comp))
        total = col if total is None else total + col
    return total


def _column_aod(path):
    ds = xr.open_dataset(path)
    col = ds["Extinction_Column_Optical_Depth"]
    if "time" in col.dims:
        col = col.isel(time=0)
    col = col.load()
    ds.close()
    return col


def _area_mean(da):
    return float(da.weighted(np.cos(np.deg2rad(da["lat"]))).mean())


def _coarsen_to_reference(col, ref):
    """Coarsen a native L576x361 column field onto the reference L288x180 grid
    with the *same* operator as species_optics.py: a latitude 1/4-1/2-1/4
    adjacent-midpoint stencil and longitude stride-2 decimation. Returns a
    DataArray carrying the reference's lat/lon so differences align exactly
    (apples-to-apples with the reference's own coarsening)."""
    v = np.asarray(col.values, dtype=np.float64)
    v_mid = 0.5 * (v[:-1, :] + v[1:, :])            # 361 -> 360 lat midpoints
    v_sub = 0.5 * (v_mid[:-1:2, :] + v_mid[1::2, :])  # 360 -> 180 lat pairs
    v_sub = v_sub[:, ::2]                            # 576 -> 288 lon (decimate)
    if v_sub.shape != (ref["lat"].size, ref["lon"].size):
        raise ValueError("coarsened MAM %s != reference grid %s"
                         % (v_sub.shape, (ref["lat"].size, ref["lon"].size)))
    return xr.DataArray(v_sub, dims=["lat", "lon"],
                        coords={"lat": ref["lat"], "lon": ref["lon"]})


def fig_aod_maps(outdir):
    try:
        import cartopy.crs as ccrs
        from cartopy.util import add_cyclic_point
    except Exception as exc:
        print("  skipping maps (cartopy import failed): %s" % exc)
        return
    mam_p, ref_p = _aod_paths()
    if not (os.path.exists(mam_p) and os.path.exists(ref_p)):
        print("  skipping maps (slice/reference missing)")
        return
    mam, ref = _mam_total_column(), _column_aod(ref_p)
    diff = _coarsen_to_reference(mam, ref) - ref
    m_mam, m_ref, m_diff = _area_mean(mam), _area_mean(ref), _area_mean(diff)

    proj = ccrs.PlateCarree()
    levels = np.arange(0, 1.0001, 0.05)
    dlev = np.linspace(-0.5, 0.5, 21)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), subplot_kw={"projection": proj})
    base = None
    for ax, (da, title, mean) in zip(axes[:2],
                                     [(mam, "MAM", m_mam), (ref, "Reference", m_ref)]):
        vals_c, lon_c = add_cyclic_point(da.values, coord=da["lon"].values)
        ax.set_facecolor("0.6")
        base = ax.contourf(*np.meshgrid(lon_c, da["lat"].values), vals_c, levels,
                           cmap="turbo", extend="max", transform=proj)
        add_coastlines(ax)
        ax.set_title("%s   $\\overline{\\tau}=%.3f$" % (title, mean))
    vals_d, lon_d = add_cyclic_point(diff.values, coord=diff["lon"].values)
    axes[2].set_facecolor("0.6")
    cfd = axes[2].contourf(*np.meshgrid(lon_d, diff["lat"].values), vals_d, dlev,
                           cmap="RdBu_r", extend="both", transform=proj)
    add_coastlines(axes[2])
    axes[2].set_title("MAM $-$ Reference   $\\Delta\\overline{\\tau}=%+.3f$" % m_diff)
    cb1 = fig.colorbar(base, ax=axes[:2], orientation="horizontal", pad=0.05, shrink=0.6, aspect=40)
    cb1.set_label("Column AOD $\\tau$")
    cb2 = fig.colorbar(cfd, ax=axes[2], orientation="horizontal", pad=0.05, shrink=0.85, aspect=40)
    cb2.set_label("$\\Delta$ Column AOD")
    fig.suptitle("Column AOD: MAM vs Reference (GEOS-IT 2008-07-01 00Z, 0.55 μm)", fontweight="bold")
    save(fig, outdir, "aod_maps")


def fig_aod_zonal(outdir):
    mam_p, ref_p = _aod_paths()
    if not (os.path.exists(mam_p) and os.path.exists(ref_p)):
        print("  skipping zonal (slice/reference missing)")
        return
    mam, ref = _mam_total_column(), _column_aod(ref_p)
    zm_mam = _coarsen_to_reference(mam, ref).mean("lon")
    zm_ref = ref.mean("lon")
    lat = ref["lat"].values
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.fill_betweenx(lat, zm_ref.values, zm_mam.values,
                     where=(zm_mam.values >= zm_ref.values), color=NCAR["red"], alpha=0.18)
    ax.fill_betweenx(lat, zm_ref.values, zm_mam.values,
                     where=(zm_mam.values < zm_ref.values), color=NCAR["ncar_blue"], alpha=0.18)
    ax.plot(zm_mam.values, lat, color=NCAR["ncar_blue"], label="MAM (%.3f)" % _area_mean(mam))
    ax.plot(zm_ref.values, lat, color=NCAR["gray"], label="Reference (%.3f)" % _area_mean(ref))
    ax.set_ylim(-90, 90); ax.set_yticks(np.arange(-90, 91, 30)); ax.set_xlim(left=0)
    ax.set_xlabel("Column AOD"); ax.set_ylabel("Latitude (°N)")
    shade = [plt.Rectangle((0, 0), 1, 1, color=NCAR["red"], alpha=0.18),
             plt.Rectangle((0, 0), 1, 1, color=NCAR["ncar_blue"], alpha=0.18)]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + shade, labels + ["MAM > Reference", "MAM < Reference"],
              frameon=True, title="Global Mean", loc="upper right")
    ax.set_title("Zonal-Mean AOD (0.55 μm)", fontweight="bold")
    save(fig, outdir, "aod_zonal")


# --- multi-band spatial maps (per-band AER_MAM4 run outputs) -----------------
# Spectral midpoints (um) of the produced LFL bands, from LFL_SW/LW_bands.
BAND_WAVELENGTH_UM = {
    "SW02": 0.340, "SW05": 0.546, "SW09": 0.966, "SW12": 2.202, "LW06": 9.645,
}
BAND_PLOT_ORDER = ["SW02", "SW05", "SW09", "SW12", "LW06"]
BAND_FILE_PREFIX = "GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294"
BAND_STAMP = "2008-07-01T0000"
# Wavelength pair for the Angstrom-exponent map (visible -> NIR).
ANGSTROM_PAIR = ("SW02", "SW09")


def _band_path(run_dir, band):
    return os.path.join(os.path.expanduser(run_dir),
                        "%s.AER_MAM4_%s.%s.V01.nc4" % (BAND_FILE_PREFIX, band, BAND_STAMP))


def _available_bands(run_dir):
    return [b for b in BAND_PLOT_ORDER if os.path.exists(_band_path(run_dir, b))]


def _col_from_ds(ds):
    col = ds["Extinction_Column_Optical_Depth"]
    if "time" in col.dims:
        col = col.isel(time=0)
    return col.load()


def _column_ssa(ds):
    """Column single-scattering albedo = sum(scattering)/sum(extinction) over
    levels, masked where the column extinction is negligible."""
    ext = ds["Extinction_Layer_Optical_Depth"]
    sca = ds["Scattering_Layer_Optical_Depth"]
    if "time" in ext.dims:
        ext = ext.isel(time=0)
        sca = sca.isel(time=0)
    ext_c = ext.sum("lev")
    sca_c = sca.sum("lev")
    return xr.where(ext_c > 1.0e-4, sca_c / ext_c, np.nan).load()


def _map_panels(panels, levels, cmap, extend, cbar_label, suptitle, outdir, name,
                cbar_ticks=None):
    """Render (title, DataArray[lat,lon]) panels as a shared-colorbar map grid."""
    try:
        from cartopy.util import add_cyclic_point
        import cartopy.crs as ccrs
    except Exception as exc:
        print("  skipping %s (cartopy import failed): %s" % (name, exc))
        return
    proj = ccrs.PlateCarree()
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.2 * nrows + 0.7),
                             subplot_kw={"projection": proj}, squeeze=False)
    axes_flat = list(axes.ravel())
    mappable = None
    for ax, (title, da) in zip(axes_flat, panels):
        vals, lon = add_cyclic_point(da.values, coord=da["lon"].values)
        ax.set_facecolor("0.6")
        mappable = ax.contourf(*np.meshgrid(lon, da["lat"].values), vals, levels,
                               cmap=cmap, extend=extend, transform=proj)
        add_coastlines(ax)
        ax.set_title(title)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(mappable, ax=axes_flat[:n], orientation="horizontal",
                      pad=0.04, shrink=0.55, aspect=45)
    cb.set_label(cbar_label)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    if min(levels) > 0:
        cb.ax.xaxis.set_major_formatter(DECIMAL)
    fig.suptitle(suptitle, fontweight="bold")
    save(fig, outdir, name)


def fig_band_aod_maps(outdir, run_dir):
    bands = _available_bands(run_dir)
    if not bands:
        print("  skipping band_aod_maps (no AER_MAM4 band files in %s)" % run_dir)
        return
    panels = []
    for b in bands:
        ds = xr.open_dataset(_band_path(run_dir, b))
        col = _col_from_ds(ds)
        panels.append(("%s  %.3f $\\mu$m   $\\overline{\\tau}=%.3f$"
                       % (b, BAND_WAVELENGTH_UM[b], _area_mean(col)), col))
        ds.close()
    _map_panels(panels, np.arange(0, 1.0001, 0.05), "turbo", "max", "Column AOD $\\tau$",
                "Column AOD by Band (MAM, GEOS-IT 2008-07-01 00Z, SW05-anchored)",
                outdir, "band_aod_maps",
                cbar_ticks=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0])


def fig_band_ssa_maps(outdir, run_dir):
    bands = _available_bands(run_dir)
    if not bands:
        print("  skipping band_ssa_maps (no AER_MAM4 band files in %s)" % run_dir)
        return
    panels = []
    for b in bands:
        ds = xr.open_dataset(_band_path(run_dir, b))
        ssa = _column_ssa(ds)
        panels.append(("%s  %.3f $\\mu$m   $\\overline{\\omega}=%.3f$"
                       % (b, BAND_WAVELENGTH_UM[b], _area_mean(ssa)), ssa))
        ds.close()
    # Non-uniform levels: most aerosol is weakly absorbing (omega~0.95-1) so the
    # interesting structure lives near the top of the range; LW is far lower.
    levels = [0.70, 0.80, 0.86, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
    _map_panels(panels, levels, "viridis", "min", "Column Single-Scattering Albedo $\\omega_0$",
                "Column SSA by Band (MAM, GEOS-IT 2008-07-01 00Z)",
                outdir, "band_ssa_maps", cbar_ticks=levels)


def fig_band_angstrom_map(outdir, run_dir):
    b1, b2 = ANGSTROM_PAIR
    if not (os.path.exists(_band_path(run_dir, b1)) and os.path.exists(_band_path(run_dir, b2))):
        print("  skipping band_angstrom_map (need %s and %s)" % (b1, b2))
        return
    ds1 = xr.open_dataset(_band_path(run_dir, b1))
    ds2 = xr.open_dataset(_band_path(run_dir, b2))
    t1 = _col_from_ds(ds1)
    t2 = _col_from_ds(ds2)
    ds1.close()
    ds2.close()
    w1, w2 = BAND_WAVELENGTH_UM[b1], BAND_WAVELENGTH_UM[b2]
    valid = (t1 > 1.0e-3) & (t2 > 1.0e-3)
    ang = xr.where(valid, -np.log(t2 / t1) / np.log(w2 / w1), np.nan)
    _map_panels([("$\\alpha$(%s/%s: %.2f, %.2f $\\mu$m)   $\\overline{\\alpha}=%.2f$"
                  % (b1, b2, w1, w2, _area_mean(ang)), ang)],
                np.arange(-0.5, 2.51, 0.25), "Spectral_r", "both",
                "Ångström Exponent $\\alpha$",
                "Ångström Exponent %.2f$-$%.2f $\\mu$m (MAM, GEOS-IT 2008-07-01 00Z)"
                % (w1, w2), outdir, "band_angstrom_map")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default=os.path.expanduser("~/Plots/verification"))
    parser.add_argument("--aerosol", default="aerosol.yaml")
    parser.add_argument("--optics-dir", default=os.path.expanduser("~/Data/Optics/SARB"))
    parser.add_argument("--band-run-dir", default=os.path.expanduser("~/Data/GEOSIT_MAM/2008/07"),
                        help="directory holding the per-band AER_MAM4_<BAND> run outputs")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    apply_style()
    config = load_config(args.aerosol)

    print("figures -> %s" % args.outdir)
    fig_size_distributions(args.outdir, config)
    fig_hygroscopicity(args.outdir, config)
    fig_kohler(args.outdir, config)
    fig_refractive_mixing(args.outdir)
    fig_mie_efficiency(args.outdir)
    fig_mode_integrated(args.outdir, args.optics_dir)
    fig_spectral_radiative(args.outdir, args.optics_dir, config)
    fig_hydration_radiative(args.outdir, args.optics_dir, config)
    fig_composition_radiative(args.outdir, args.optics_dir, config)
    fig_aod_components(args.outdir, config)
    fig_aod_maps(args.outdir)
    fig_aod_zonal(args.outdir)
    fig_band_aod_maps(args.outdir, args.band_run_dir)
    fig_band_ssa_maps(args.outdir, args.band_run_dir)
    fig_band_angstrom_map(args.outdir, args.band_run_dir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
