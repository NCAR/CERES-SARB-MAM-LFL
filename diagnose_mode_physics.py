import argparse
import os
from types import SimpleNamespace

import numpy as np
import xarray as xr

from mode_config import load_config, resolved_allocations
from mode_optics import (
    COL,
    _band_label,
    _band_wavelength_um,
    _build_path,
    _date_strings,
    _has_band_limits,
    _mode_species,
    _mode_table_path,
    _select_source_timestep,
    _source_spec,
    _species_info,
    _timestamp_from_date_str,
    _refractive_indices,
)
from mode_physics import (
    derive_number_mixing_ratio,
    layer_optical_depth,
    lognormal_volume_factor,
    lookup_mode_optics,
    mix_mode_state,
)
from source_fields import open_source_fields


G = 9.8


def _fmt(value):
    if value is None:
        return "n/a"
    return "%.6g" % float(value)


def _stats(values, mask=None):
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask)]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None, None, None
    return float(array.min()), float(array.mean()), float(array.max())


def _print_stats(label, values, mask=None):
    lo, mean, hi = _stats(values, mask=mask)
    print("    %-24s min=%s mean=%s max=%s" % (label, _fmt(lo), _fmt(mean), _fmt(hi)))


def _field_column(field, dims):
    array = np.asarray(field, dtype=np.float64)
    dim_names = list(dims)
    lev_axis = dim_names.index("lev")
    column = array.sum(axis=lev_axis)
    dim_names.pop(lev_axis)
    if "time" in dim_names and column.shape[dim_names.index("time")] == 1:
        time_axis = dim_names.index("time")
        column = np.take(column, 0, axis=time_axis)
        dim_names.pop(time_axis)
    return column, tuple(dim_names)


def _area_weighted_mean_2d(field, dims, lat):
    array = np.asarray(field, dtype=np.float64)
    dim_names = list(dims)
    if "time" in dim_names and array.shape[dim_names.index("time")] == 1:
        time_axis = dim_names.index("time")
        array = np.take(array, 0, axis=time_axis)
        dim_names.pop(time_axis)
    if tuple(dim_names) != ("lat", "lon"):
        raise ValueError("expected lat/lon field, got dims %s" % (dim_names,))
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return float((array * weights[:, None]).sum() / (weights.sum() * array.shape[1]))


def _area_weighted_column_mean(layer, dims, lat):
    column, column_dims = _field_column(layer, dims)
    return _area_weighted_mean_2d(column, column_dims, lat)


def _column_stats(layer, dims, lat):
    column, column_dims = _field_column(layer, dims)
    return (
        _area_weighted_mean_2d(column, column_dims, lat),
        float(np.nanmin(column)),
        float(np.nanmean(column)),
        float(np.nanmax(column)),
    )


def _file_column_mean(path):
    with xr.open_dataset(path) as ds:
        column = ds[COL]
        if "time" in column.dims and column.sizes["time"] == 1:
            column = column.isel(time=0)
        lat = column["lat"].values
        return _area_weighted_mean_2d(column.values, column.dims, lat)


def _table_clip_report(label, values, table_values):
    values = np.asarray(values, dtype=np.float64)
    table_values = np.asarray(table_values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        print("    %-24s no finite values" % label)
        return
    below = float((values[finite] < table_values.min()).mean())
    above = float((values[finite] > table_values.max()).mean())
    print(
        "    %-24s input=[%s,%s] table=[%s,%s] below=%.3g above=%.3g"
        % (
            label,
            _fmt(values[finite].min()),
            _fmt(values[finite].max()),
            _fmt(table_values.min()),
            _fmt(table_values.max()),
            below,
            above,
        )
    )


def _date_str(args):
    return next(_date_strings(args.time, args.time))


def _band_bounds(config, args):
    if args.wvl is not None:
        return None
    path = os.path.expandvars(config["filename_bands"])
    label = str(args.band).lower()
    family = label[:2]
    index = int(label[2:]) - 1
    variable = "LFL_SW_bands" if family == "sw" else "LFL_LW_bands"
    with xr.open_dataset(path) as ds:
        values = np.asarray(ds[variable].values, dtype=np.float64).ravel()
    return path, variable, values[index], values[index + 1]


def _diagnose_mode(config, source_key, source_spec, scheme, mode, band_label, date_str, args, fields):
    mode_spec = config["Schemes"][scheme]["modes"][mode]
    species_names = _mode_species(config, scheme, mode)
    allocations = resolved_allocations(config, scheme)
    q = {
        species: np.asarray(fields.species[species], dtype=np.float32)
        * float(allocations[species].get(mode, 0.0))
        for species in species_names
    }
    species_info = _species_info(config, species_names)

    table_path = _mode_table_path(mode_spec, args)
    print("\n[%s] %s" % (mode, mode_spec.get("name", "")))
    print("  table: %s" % table_path)

    with xr.open_dataset(table_path) as ds_table:
        if args.wvl is not None or _has_band_limits(ds_table):
            wavelength_um = _band_wavelength_um(args, ds_table)
        else:
            with xr.open_dataset(os.path.expandvars(config["filename_bands"])) as ds_bands:
                wavelength_um = _band_wavelength_um(args, ds_bands)

        print("  wavelength_um: %s" % _fmt(wavelength_um))
        print(
            "  table ranges: radius=[%s,%s] n_real=[%s,%s] n_imag=[%s,%s]"
            % (
                _fmt(ds_table["radius"].min()),
                _fmt(ds_table["radius"].max()),
                _fmt(ds_table["n_real"].min()),
                _fmt(ds_table["n_real"].max()),
                _fmt(ds_table["n_imag"].min()),
                _fmt(ds_table["n_imag"].max()),
            )
        )
        _print_stats("table ext um2", ds_table["ext"].values)
        _print_stats("table abs um2", ds_table["abs"].values)

        refractive = _refractive_indices(config, species_names, wavelength_um)
        state = mix_mode_state(
            species_info,
            q,
            refractive,
            fields.rh,
            fields.temperature,
            mode_spec["dry_radius_um"],
        )
        dry_mask = state["dry_volume"] > 0.0
        number = derive_number_mixing_ratio(
            state["dry_volume"],
            mode_spec["dry_radius_um"],
            mode_spec["sigma_g"],
        )
        cross_ext, cross_abs, asm = lookup_mode_optics(
            state["n_re"],
            state["n_im"],
            state["r_w_um"],
            ds_table,
        )
        tau_ext = layer_optical_depth(fields.delp, number, cross_ext)
        tau_abs = layer_optical_depth(fields.delp, number, cross_abs)

        _table_clip_report("lookup n_real", state["n_re"], ds_table["n_real"].values)
        _table_clip_report("lookup n_imag_abs", np.abs(state["n_im"]), ds_table["n_imag"].values)
        _table_clip_report("lookup radius_um", state["r_w_um"], ds_table["radius"].values)

    lat = np.asarray(fields.coords["lat"])
    dims = tuple(fields.dims)
    mass = np.zeros_like(fields.delp, dtype=np.float32)
    for species, values in q.items():
        mass += values

    dry_mass_col = _area_weighted_column_mean(mass * fields.delp / G, dims, lat)
    dry_volume_col = _area_weighted_column_mean(state["dry_volume"] * fields.delp / G, dims, lat)
    number_col = _area_weighted_column_mean(number * fields.delp / G, dims, lat)
    tau_mean, tau_min, tau_simple, tau_max = _column_stats(tau_ext, dims, lat)
    abs_mean, _abs_min, _abs_simple, _abs_max = _column_stats(tau_abs, dims, lat)

    print("  configured dry_radius_um=%s sigma_g=%s" % (_fmt(mode_spec["dry_radius_um"]), _fmt(mode_spec["sigma_g"])))
    print("  particle dry volume m3: %s" % _fmt(lognormal_volume_factor(mode_spec["dry_radius_um"], mode_spec["sigma_g"])))
    print("  column dry mass kg/m2: %s" % _fmt(dry_mass_col))
    print("  column dry volume m3/m2: %s" % _fmt(dry_volume_col))
    print("  column particle number m-2: %s" % _fmt(number_col))
    if dry_mass_col > 0.0:
        print("  effective ext m2/kg: %s" % _fmt(tau_mean / dry_mass_col))
    print("  computed column AOD: area=%s simple=%s min=%s max=%s" % (_fmt(tau_mean), _fmt(tau_simple), _fmt(tau_min), _fmt(tau_max)))
    print("  computed column ABS: area=%s" % _fmt(abs_mean))

    print("  species dry mass columns:")
    for species in species_names:
        species_mass_col = _area_weighted_column_mean(q[species] * fields.delp / G, dims, lat)
        print("    %-12s allocation=%s kg/m2=%s" % (species, _fmt(allocations[species].get(mode, 0.0)), _fmt(species_mass_col)))

    _print_stats("dry volume m3/kg", state["dry_volume"], mask=dry_mask)
    _print_stats("number kg-1", number, mask=dry_mask)
    _print_stats("wet radius um", state["r_w_um"], mask=dry_mask)
    _print_stats("n_real selected", state["n_re"], mask=dry_mask)
    _print_stats("n_imag selected", state["n_im"], mask=dry_mask)
    _print_stats("cross ext um2", cross_ext, mask=dry_mask)
    _print_stats("cross abs um2", cross_abs, mask=dry_mask)
    _print_stats("asm", asm, mask=dry_mask)

    label = "%s_%s" % (scheme, mode)
    output_path = _build_path(args.outdir, source_spec["output_pattern"], date_str, label, band_label)
    if os.path.exists(output_path):
        print("  existing output AOD area mean: %s" % _fmt(_file_column_mean(output_path)))
    else:
        print("  existing output missing: %s" % output_path)

    return tau_mean


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose one mode-optics physics chain")
    parser.add_argument("--aerosol", default="aerosol_ceres.yaml")
    parser.add_argument("--source", default="geosit")
    parser.add_argument("--scheme", default="MAM4")
    parser.add_argument("--band", default=None)
    parser.add_argument("--wvl", type=float, default=None)
    parser.add_argument("--time", required=True)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--reference-total", default=None)
    parser.add_argument("--mode", action="append", default=None)
    args = parser.parse_args(argv)
    if args.band is None and args.wvl is None:
        args.band = "sw05"

    config = load_config(args.aerosol)
    source_key = str(args.source).upper()
    source_spec = _source_spec(config, source_key)
    band_label = _band_label(SimpleNamespace(wvl=args.wvl, band=args.band))
    date_str = _date_str(args)
    input_path = _build_path(args.datadir, source_spec["input_pattern"], date_str, "", band_label)

    print("config: %s" % args.aerosol)
    print("source: %s" % source_key)
    print("input: %s" % input_path)
    print("band label: %s" % band_label)
    bounds = _band_bounds(config, args)
    if bounds is not None:
        path, variable, lower, upper = bounds
        print("band bounds: %s %s=[%s,%s] midpoint=%s" % (path, variable, _fmt(lower), _fmt(upper), _fmt(0.5 * (lower + upper))))

    all_species = []
    modes = args.mode or list(config["Schemes"][args.scheme]["modes"])
    for mode in modes:
        if mode not in config["Schemes"][args.scheme]["modes"]:
            raise SystemExit("unknown mode %s" % mode)
        for species in _mode_species(config, args.scheme, mode):
            if species not in all_species:
                all_species.append(species)
    fields = open_source_fields(input_path, source_spec, all_species)
    fields = _select_source_timestep(fields, _timestamp_from_date_str(date_str))
    print("field dims: %s" % (fields.dims,))
    print("field shape: %s" % (np.asarray(fields.rh).shape,))

    total = 0.0
    for mode in modes:
        total += _diagnose_mode(config, source_key, source_spec, args.scheme, mode, band_label, date_str, args, fields)

    print("\n[total]")
    print("  computed mode-sum AOD area mean: %s" % _fmt(total))
    if args.reference_total:
        print("  reference total AOD area mean: %s" % _fmt(_file_column_mean(args.reference_total)))
        if total > 0.0:
            print("  reference/computed ratio: %s" % _fmt(_file_column_mean(args.reference_total) / total))


if __name__ == "__main__":
    main()
