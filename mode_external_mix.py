import argparse

import numpy as np
import xarray as xr


TAU_THRESH = 1e-5

DELP = "DELP"
EXT = "Extinction_Layer_Optical_Depth"
SCA = "Scattering_Layer_Optical_Depth"
ASM = "Layer_Asymmetry_Parameter"
COL = "Extinction_Column_Optical_Depth"

REQUIRED_VARIABLES = (DELP, EXT, SCA, ASM, COL)
LAYER_VARIABLES = (EXT, SCA, ASM)
SUPPORTED_LAYER_DIMS = (
    ("lev", "lat", "lon"),
    ("time", "lev", "lat", "lon"),
)


def _as_list(datasets):
    datasets = list(datasets)
    if not datasets:
        raise ValueError("datasets must contain at least one dataset")
    return datasets


def _require_variables(ds, index):
    missing = [name for name in REQUIRED_VARIABLES if name not in ds]
    if missing:
        raise ValueError("dataset %d missing required variable %s" % (index, missing[0]))


def _expected_layer_dims(dims):
    dims = tuple(dims)
    if dims not in SUPPORTED_LAYER_DIMS:
        raise ValueError("unsupported dims %s for layer variables" % (dims,))
    return dims


def _validate_dims_shape(ds, index, expected_dims, expected_shape):
    for name in LAYER_VARIABLES:
        da = ds[name]
        _expected_layer_dims(da.dims)
        if da.dims != expected_dims:
            raise ValueError("dataset %d variable %s dims must match %s" % (index, name, expected_dims))
        if da.shape != expected_shape:
            raise ValueError("dataset %d variable %s shape must match %s" % (index, name, expected_shape))

    delp = ds[DELP]
    if delp.dims != expected_dims:
        raise ValueError("dataset %d variable %s dims must match %s" % (index, DELP, expected_dims))
    if delp.shape != expected_shape:
        raise ValueError("dataset %d variable %s shape must match %s" % (index, DELP, expected_shape))


def _validate_ext_coords(ds, index, base_ext):
    ext = ds[EXT]
    for coord in base_ext.dims:
        base_has_coord = coord in base_ext.coords
        has_coord = coord in ext.coords
        if base_has_coord != has_coord:
            raise ValueError("dataset %d coord %s must match first dataset" % (index, coord))
        if not base_has_coord:
            continue
        if not np.array_equal(base_ext.coords[coord].values, ext.coords[coord].values):
            raise ValueError("dataset %d coord %s must match first dataset" % (index, coord))


def _validate_delp_values(ds, index, base_delp):
    if index == 0:
        return
    if not np.allclose(ds[DELP].values, base_delp.values):
        raise ValueError("dataset %d variable %s values must match first dataset" % (index, DELP))


def _has_any(condition):
    return bool(condition.any().compute().item())


def _validate_optical_depths(ext, sca, index):
    if _has_any(ext < 0.0):
        raise ValueError("dataset %d variable %s contains negative values" % (index, EXT))
    if _has_any(sca < 0.0):
        raise ValueError("dataset %d variable %s contains negative values" % (index, SCA))
    if _has_any(sca > ext):
        raise ValueError("dataset %d variable %s exceeds %s" % (index, SCA, EXT))


def mix_mode_datasets(datasets):
    datasets = _as_list(datasets)
    for index, ds in enumerate(datasets):
        if not isinstance(ds, xr.Dataset):
            raise ValueError("datasets must contain xarray Dataset objects")
        _require_variables(ds, index)

    first = datasets[0]
    base_ext = first[EXT]
    base_delp = first[DELP]
    expected_dims = _expected_layer_dims(base_ext.dims)
    expected_shape = base_ext.shape

    total_ext = xr.zeros_like(base_ext, dtype=np.float32)
    total_sca = xr.zeros_like(base_ext, dtype=np.float32)
    weighted_asm = xr.zeros_like(base_ext, dtype=np.float32)

    for index, ds in enumerate(datasets):
        _validate_dims_shape(ds, index, expected_dims, expected_shape)
        _validate_ext_coords(ds, index, base_ext)
        _validate_delp_values(ds, index, base_delp)
        ext = ds[EXT].astype(np.float32)
        sca = ds[SCA].astype(np.float32)
        asm = ds[ASM].astype(np.float32)

        _validate_optical_depths(ext, sca, index)

        total_ext = total_ext + ext
        total_sca = total_sca + sca
        weighted_asm = weighted_asm + sca * asm

    total_ext = total_ext.astype(np.float32)
    total_sca = total_sca.astype(np.float32)
    layer_asm = xr.where(
        total_sca > TAU_THRESH,
        weighted_asm / total_sca,
        0.0,
    ).astype(np.float32)
    column_ext = total_ext.sum(dim="lev").astype(np.float32)

    return xr.Dataset(
        {
            DELP: first[DELP].astype(np.float32),
            EXT: total_ext,
            SCA: total_sca,
            ASM: layer_asm,
            COL: column_ext,
        },
        coords=first.coords,
        attrs=first.attrs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="output mixed AER NetCDF file")
    parser.add_argument("mode_files", nargs="+", help="corrected mode NetCDF files")
    args = parser.parse_args(argv)

    opened = []
    try:
        for filename in args.mode_files:
            opened.append(xr.open_dataset(filename))
        mixed = mix_mode_datasets(opened).load()
        mixed.to_netcdf(args.output)
    finally:
        for ds in opened:
            ds.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
