import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from mode_config import load_config, resolved_allocations
from mode_physics import (
    derive_number_mixing_ratio,
    layer_optical_depth,
    lookup_mode_optics,
    mix_mode_state,
)
from source_fields import open_source_fields
from utils import fill_date_hour_template


DELP = "DELP"
EXT = "Extinction_Layer_Optical_Depth"
SCA = "Scattering_Layer_Optical_Depth"
ASM = "Layer_Asymmetry_Parameter"
COL = "Extinction_Column_Optical_Depth"

SUPPORTED_LAYER_DIMS = (
    ("lev", "lat", "lon"),
    ("time", "lev", "lat", "lon"),
)


def _require_dataarray(name, value):
    if not isinstance(value, xr.DataArray):
        raise ValueError("%s must be an xarray DataArray" % name)


def _has_any(condition):
    return bool(condition.any().compute().item())


def _validate_layer_inputs(delp, tau_ext, tau_sca, asm):
    arrays = (
        ("delp", delp),
        ("tau_ext", tau_ext),
        ("tau_sca", tau_sca),
        ("asm", asm),
    )
    for name, value in arrays:
        _require_dataarray(name, value)

    if "lev" not in tau_ext.dims:
        raise ValueError("tau_ext must include lev dimension")

    expected_dims = tau_ext.dims
    if expected_dims not in SUPPORTED_LAYER_DIMS:
        raise ValueError("unsupported dims %s for layer variables" % (expected_dims,))

    expected_shape = tau_ext.shape
    for name, value in arrays:
        if value.dims != expected_dims:
            raise ValueError("%s dims must match tau_ext dims %s" % (name, expected_dims))
        if value.shape != expected_shape:
            raise ValueError("%s shape must match tau_ext shape %s" % (name, expected_shape))

    try:
        xr.align(delp, tau_ext, tau_sca, asm, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("input coords must match tau_ext coords") from exc

    if _has_any(tau_ext < 0.0):
        raise ValueError("tau_ext contains negative values")


def build_mode_output_dataset(delp, tau_ext, tau_sca, asm, attrs):
    _validate_layer_inputs(delp, tau_ext, tau_sca, asm)

    tau_ext = tau_ext.astype(np.float32)
    tau_sca = tau_sca.astype(np.float32)
    asm = asm.astype(np.float32)

    tau_sca = xr.where(tau_sca < 0.0, 0.0, tau_sca)
    tau_sca = xr.where(tau_sca > tau_ext, tau_ext, tau_sca).astype(np.float32)
    asm = asm.clip(min=-1.0, max=1.0).astype(np.float32)
    column_ext = tau_ext.sum(dim="lev").astype(np.float32)

    return xr.Dataset(
        {
            DELP: delp.astype(np.float32),
            EXT: tau_ext,
            SCA: tau_sca,
            ASM: asm,
            COL: column_ext,
        },
        coords=tau_ext.coords,
        attrs=dict(attrs or {}),
    )


def _band_label(args):
    if args.wvl is not None:
        wvl = float(args.wvl)
        if wvl.is_integer():
            return "%dNM" % int(wvl)
        return "%gNM" % wvl
    return str(args.band).upper()


def _date_strings(start, end):
    start_time = pd.to_datetime(start)
    end_time = pd.to_datetime(end)
    for timestamp in pd.date_range(start_time, end_time, freq="3h"):
        yield timestamp.strftime("%Y-%m-%b-%d-%j-%H")


def _source_spec(config, source):
    return config["Sources"][str(source).upper()]


def _mode_species(config, scheme, mode):
    allocations = resolved_allocations(config, scheme)
    return [
        species
        for species, weights in allocations.items()
        if float(weights.get(mode, 0.0)) > 0.0
    ]


def _build_path(root, pattern, date_str, label, band_label):
    expanded = os.path.expandvars(str(pattern))
    filled = fill_date_hour_template(expanded, date_str)
    filled = filled.replace("{label}", label).replace("{band}", band_label)
    if os.path.isabs(filled):
        return filled
    root = os.path.expandvars(root or "")
    if root:
        return os.path.join(root, filled)
    return filled


def _mode_table_path(mode_spec, args):
    path = os.path.expandvars(mode_spec["filename_sarb"])
    if args.wvl is not None:
        replacement = "%dnm_larc" % int(float(args.wvl))
    else:
        replacement = "%s_larc" % _band_label(args)
    return path.replace("larc", replacement, 1)


def _band_wavelength_um(args, ds_table):
    if args.wvl is not None:
        return float(args.wvl) / 1000.0

    label = str(args.band).lower()
    family = label[:2]
    try:
        index = int(label[2:]) - 1
    except ValueError as exc:
        raise ValueError("band must look like sw01 or lw01") from exc

    if family == "sw":
        variable = "LFL_SW_bands"
    elif family == "lw":
        variable = "LFL_LW_bands"
    else:
        raise ValueError("band must start with sw or lw")

    bands = np.asarray(ds_table[variable].values, dtype=np.float64)
    if bands.ndim == 1:
        values = bands[index : index + 2]
    else:
        values = np.asarray(bands[index]).squeeze()
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("%s band %s does not define two bounds" % (variable, label))
    return float(0.5 * (values[0] + values[-1]))


def _type_key(species_name):
    name = str(species_name).upper()
    if name == "SO4":
        return "SU"
    if name.startswith("NO3"):
        return "NI"
    if name.startswith("OC"):
        return "POM"
    if name.startswith("BC"):
        return "BC"
    if name.startswith("DU"):
        return "DU"
    if name.startswith("SS"):
        return "SS"
    return name


def _hygroscopicity_values(values):
    next_values = []
    for value in values or []:
        next_values.append(0.0 if value is None else float(value))
    while len(next_values) < 3:
        next_values.append(0.0)
    return next_values[:3]


def _species_info(config, species_names):
    types = config["Types"]
    info = {}
    for species in species_names:
        type_name = _type_key(species)
        type_info = types[type_name]
        info[species] = {
            "density": float(type_info["density"]),
            "hygroscopicity": _hygroscopicity_values(type_info.get("hygroscopicity")),
        }
    return info


def _nearest_scalar(ds, wavelength_name, variable_name, target):
    wavelength = ds[wavelength_name]
    values = np.asarray(wavelength.values, dtype=np.float64).squeeze()
    if values.size == 0:
        raise ValueError("%s has no wavelength values" % wavelength_name)
    index = int(np.argmin(np.abs(values.ravel() - float(target))))
    dim = wavelength.dims[-1] if wavelength.dims else wavelength_name
    variable = ds[variable_name]
    if dim in variable.dims:
        variable = variable.isel({dim: index})
    elif wavelength_name in variable.dims:
        variable = variable.isel({wavelength_name: index})
    selected = np.asarray(variable.values, dtype=np.float64).squeeze()
    if selected.size == 0:
        raise ValueError("%s has no selected value" % variable_name)
    return float(selected.reshape(-1)[0])


def _type_refractive_index(type_name, type_info, wavelength_um):
    path = os.path.expandvars(type_info["filename"])
    ds = xr.open_dataset(path)
    try:
        if str(type_name).upper() == "WAT":
            return (
                _nearest_scalar(ds, "wavelength1", "watern", wavelength_um),
                _nearest_scalar(ds, "wavelength1", "wateri", wavelength_um),
            )
        wavelength_m = float(wavelength_um) * 1.0e-6
        return (
            _nearest_scalar(ds, "lambda", "refreal", wavelength_m),
            _nearest_scalar(ds, "lambda", "refimag", wavelength_m),
        )
    finally:
        ds.close()


def _refractive_indices(config, species_names, wavelength_um):
    types = config["Types"]
    cache = {}
    refractive = {}
    for species in species_names:
        type_name = _type_key(species)
        if type_name not in cache:
            cache[type_name] = _type_refractive_index(type_name, types[type_name], wavelength_um)
        refractive[species] = cache[type_name]
    if "WAT" in types:
        refractive["WAT"] = _type_refractive_index("WAT", types["WAT"], wavelength_um)
    return refractive


def _dataarray_from_fields(values, fields):
    dims = tuple(fields.dims)
    coords = {}
    for name, coord in fields.coords.items():
        coord_dims = getattr(coord, "dims", ())
        if all(dim in dims for dim in coord_dims):
            coords[name] = coord
    return xr.DataArray(np.asarray(values, dtype=np.float32), dims=dims, coords=coords)


def compute_mode_dataset(config, source_key, source_spec, scheme, mode, band_label, args, fields):
    mode_spec = config["Schemes"][scheme]["modes"][mode]
    species_names = _mode_species(config, scheme, mode)
    allocations = resolved_allocations(config, scheme)

    q = {}
    for species in species_names:
        allocation = float(allocations[species].get(mode, 0.0))
        q[species] = np.asarray(fields.species[species], dtype=np.float32) * allocation

    table_path = _mode_table_path(mode_spec, args)
    ds_table = xr.open_dataset(table_path)
    try:
        wavelength_um = _band_wavelength_um(args, ds_table)
        species_info = _species_info(config, species_names)
        refractive = _refractive_indices(config, species_names, wavelength_um)
        state = mix_mode_state(
            species_info,
            q,
            refractive,
            fields.rh,
            fields.temperature,
            mode_spec["dry_radius_um"],
        )
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
        cross_sca = np.maximum(cross_ext - cross_abs, 0.0).astype(np.float32)
        tau_ext = layer_optical_depth(fields.delp, number, cross_ext)
        tau_sca = layer_optical_depth(fields.delp, number, cross_sca)
    finally:
        ds_table.close()

    attrs = {
        "source": source_key,
        "scheme": scheme,
        "mode": mode,
        "band": band_label,
        "wavelength_um": float(wavelength_um),
    }
    delp_da = _dataarray_from_fields(fields.delp, fields)
    tau_ext_da = _dataarray_from_fields(tau_ext, fields)
    tau_sca_da = _dataarray_from_fields(tau_sca, fields)
    asm_da = _dataarray_from_fields(asm, fields)
    return build_mode_output_dataset(delp_da, tau_ext_da, tau_sca_da, asm_da, attrs)


def run(args):
    config = load_config(args.aerosol)
    source_key = str(args.source).upper()
    source_spec = _source_spec(config, source_key)
    band_label = _band_label(args)
    label = "%s_%s" % (args.scheme, args.mode)
    species_names = _mode_species(config, args.scheme, args.mode)

    for date_str in _date_strings(args.start, args.end):
        input_path = _build_path(
            args.datadir,
            source_spec["input_pattern"],
            date_str,
            label,
            band_label,
        )
        fields = open_source_fields(input_path, source_spec, species_names)
        ds = compute_mode_dataset(
            config,
            source_key,
            source_spec,
            args.scheme,
            args.mode,
            band_label,
            args,
            fields,
        )
        output_path = _build_path(
            args.outdir,
            source_spec["output_pattern"],
            date_str,
            label,
            band_label,
        )
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        ds.to_netcdf(output_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute native-grid internally mixed mode optics")
    parser.add_argument("--source", choices=["geosit", "merra2"], required=True)
    parser.add_argument("--scheme", default="MAM4")
    parser.add_argument("--mode", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--band", default=None)
    group.add_argument("--wvl", type=float, default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--aerosol", default="aerosol.yaml")
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--external-vis", default=None)
    args = parser.parse_args(argv)
    if args.band is None and args.wvl is None:
        args.band = "sw01"
    try:
        return run(args)
    except FileNotFoundError as exc:
        raise SystemExit("mode_optics computation failed: %s" % exc) from exc


if __name__ == "__main__":
    sys.exit(main())
