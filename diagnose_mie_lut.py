import argparse
import os
from types import SimpleNamespace

import numpy as np
import xarray as xr

from mie_sphere import mie_cross_sections_um2
from mode_config import load_config, resolved_allocations
from mode_optics import (
    _band_label,
    _band_wavelength_um,
    _build_path,
    _date_strings,
    _has_band_limits,
    _mode_species,
    _mode_table_path,
    _refractive_indices,
    _select_source_timestep,
    _source_spec,
    _species_info,
    _timestamp_from_date_str,
)
from mode_physics import mix_mode_state
from source_fields import open_source_fields


def _fmt(value):
    return "%.6g" % float(value)


def _lower_index(value, table):
    table = np.asarray(table, dtype=np.float64)
    clipped = np.clip(float(value), table[0], table[-1])
    return int(np.searchsorted(table, clipped, side="right") - 1)


def _relative_delta(test, reference):
    reference = float(reference)
    if reference == 0.0:
        return np.nan
    return (float(test) - reference) / reference


def _date_str(args):
    return next(_date_strings(args.time, args.time))


def _all_mode_species(config, scheme, mode):
    species = []
    for name in _mode_species(config, scheme, mode):
        if name not in species:
            species.append(name)
    return species


def _sample_flat_indices(state, mass, count):
    radius = np.asarray(state["r_w_um"], dtype=np.float64).ravel()
    dry_volume = np.asarray(state["dry_volume"], dtype=np.float64).ravel()
    mass_flat = np.asarray(mass, dtype=np.float64).ravel()
    mask = np.isfinite(radius) & np.isfinite(dry_volume) & (dry_volume > 0.0)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError("no dry-volume-positive points to sample")

    selected = []
    quantiles = np.linspace(0.05, 0.95, max(1, int(count)))
    radius_candidates = radius[candidates]
    for quantile in quantiles:
        target = np.quantile(radius_candidates, quantile)
        local = int(np.argmin(np.abs(radius_candidates - target)))
        selected.append(int(candidates[local]))

    selected.append(int(candidates[np.argmax(mass_flat[candidates])]))
    selected.append(int(candidates[np.argmax(radius_candidates)]))

    deduped = []
    for index in selected:
        if index not in deduped:
            deduped.append(index)
    return deduped


def _manual_points(args):
    if args.radius_um is None and args.n_real is None and args.n_imag is None:
        return []
    if args.radius_um is None or args.n_real is None or args.n_imag is None:
        raise SystemExit("--radius-um, --n-real, and --n-imag must be provided together")
    if not (len(args.radius_um) == len(args.n_real) == len(args.n_imag)):
        raise SystemExit("--radius-um, --n-real, and --n-imag must have the same length")
    return list(zip(args.radius_um, args.n_real, args.n_imag))


def _print_comparison(label, radius, n_real, n_imag, wavelength_um, ds_table):
    n_real_values = ds_table["n_real"].values
    n_imag_values = ds_table["n_imag"].values
    radius_values = ds_table["radius"].values
    i_re = _lower_index(n_real, n_real_values)
    i_im = _lower_index(abs(n_imag), n_imag_values)
    i_radius = _lower_index(radius, radius_values)

    lut_ext = float(ds_table["ext"].isel(n_real=i_re, n_imag=i_im, radius=i_radius))
    lut_abs = float(ds_table["abs"].isel(n_real=i_re, n_imag=i_im, radius=i_radius))
    lut_radius = float(radius_values[i_radius])
    lut_n_real = float(n_real_values[i_re])
    lut_n_imag = float(n_imag_values[i_im])

    mie_at_state = mie_cross_sections_um2(n_real, n_imag, radius, wavelength_um)
    mie_at_lut = mie_cross_sections_um2(lut_n_real, lut_n_imag, lut_radius, wavelength_um)

    print("\n[%s]" % label)
    print("  state radius=%s n_real=%s n_imag_abs=%s wavelength_um=%s" % (_fmt(radius), _fmt(n_real), _fmt(abs(n_imag)), _fmt(wavelength_um)))
    print("  lut   radius=%s n_real=%s n_imag=%s indices=(%d,%d,%d)" % (_fmt(lut_radius), _fmt(lut_n_real), _fmt(lut_n_imag), i_re, i_im, i_radius))
    print("  ext_um2 lut=%s mie_at_lut=%s rel=%s mie_at_state=%s rel_state_vs_lut=%s" % (
        _fmt(lut_ext),
        _fmt(mie_at_lut["ext"]),
        _fmt(_relative_delta(mie_at_lut["ext"], lut_ext)),
        _fmt(mie_at_state["ext"]),
        _fmt(_relative_delta(mie_at_state["ext"], lut_ext)),
    ))
    print("  abs_um2 lut=%s mie_at_lut=%s rel=%s mie_at_state=%s rel_state_vs_lut=%s" % (
        _fmt(lut_abs),
        _fmt(mie_at_lut["abs"]),
        _fmt(_relative_delta(mie_at_lut["abs"], lut_abs)),
        _fmt(mie_at_state["abs"]),
        _fmt(_relative_delta(mie_at_state["abs"], lut_abs)),
    ))
    print("  q_ext state=%s q_sca state=%s terms=%d" % (_fmt(mie_at_state["q_ext"]), _fmt(mie_at_state["q_sca"]), mie_at_state["series_terms"]))


def _load_state(config, args, band_label, wavelength_um, date_str):
    source_key = str(args.source).upper()
    source_spec = _source_spec(config, source_key)
    species_names = _all_mode_species(config, args.scheme, args.mode)
    input_path = _build_path(args.datadir, source_spec["input_pattern"], date_str, "", band_label)
    fields = open_source_fields(input_path, source_spec, species_names)
    fields = _select_source_timestep(fields, _timestamp_from_date_str(date_str))

    allocations = resolved_allocations(config, args.scheme)
    q = {
        species: np.asarray(fields.species[species], dtype=np.float32)
        * float(allocations[species].get(args.mode, 0.0))
        for species in species_names
    }
    mass = np.zeros_like(fields.delp, dtype=np.float32)
    for values in q.values():
        mass += values

    mode_spec = config["Schemes"][args.scheme]["modes"][args.mode]
    state = mix_mode_state(
        _species_info(config, species_names),
        q,
        _refractive_indices(config, species_names, wavelength_um),
        fields.rh,
        fields.temperature,
        mode_spec["dry_radius_um"],
    )
    return input_path, state, mass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare SARB mode LUT cross sections with independent Mie calculations")
    parser.add_argument("--aerosol", default="aerosol_ceres.yaml")
    parser.add_argument("--source", default="geosit")
    parser.add_argument("--scheme", default="MAM4")
    parser.add_argument("--mode", default="a3")
    parser.add_argument("--band", default="sw05")
    parser.add_argument("--wvl", type=float, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--radius-um", type=float, action="append", default=None)
    parser.add_argument("--n-real", type=float, action="append", default=None)
    parser.add_argument("--n-imag", type=float, action="append", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.aerosol)
    if args.mode not in config["Schemes"][args.scheme]["modes"]:
        raise SystemExit("unknown mode %s" % args.mode)

    band_args = SimpleNamespace(wvl=args.wvl, band=args.band)
    band_label = _band_label(band_args)
    mode_spec = config["Schemes"][args.scheme]["modes"][args.mode]
    table_path = _mode_table_path(mode_spec, band_args)

    with xr.open_dataset(table_path) as ds_table:
        ds_bands = None
        try:
            if args.wvl is not None or _has_band_limits(ds_table):
                wavelength_um = _band_wavelength_um(band_args, ds_table)
            else:
                ds_bands = xr.open_dataset(os.path.expandvars(config["filename_bands"]))
                wavelength_um = _band_wavelength_um(band_args, ds_bands)

            print("config: %s" % args.aerosol)
            print("table: %s" % table_path)
            print("mode: %s band: %s wavelength_um: %s" % (args.mode, band_label, _fmt(wavelength_um)))

            manual = _manual_points(args)
            for point_index, (radius, n_real, n_imag) in enumerate(manual, start=1):
                _print_comparison("manual_%d" % point_index, radius, n_real, n_imag, wavelength_um, ds_table)

            if args.time is not None:
                date_str = _date_str(args)
                input_path, state, mass = _load_state(config, args, band_label, wavelength_um, date_str)
                print("input: %s" % input_path)
                flat_indices = _sample_flat_indices(state, mass, args.sample_count)
                radius = np.asarray(state["r_w_um"]).ravel()
                n_real = np.asarray(state["n_re"]).ravel()
                n_imag = np.asarray(state["n_im"]).ravel()
                for sample_index, flat_index in enumerate(flat_indices, start=1):
                    _print_comparison(
                        "state_%d_flat_%d" % (sample_index, flat_index),
                        radius[flat_index],
                        n_real[flat_index],
                        n_imag[flat_index],
                        wavelength_um,
                        ds_table,
                    )
        finally:
            if ds_bands is not None:
                ds_bands.close()


if __name__ == "__main__":
    main()
