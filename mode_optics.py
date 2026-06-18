import argparse
import sys

import numpy as np
import xarray as xr


DELP = "DELP"
EXT = "Extinction_Layer_Optical_Depth"
SCA = "Scattering_Layer_Optical_Depth"
ASM = "Layer_Asymmetry_Parameter"
COL = "Extinction_Column_Optical_Depth"


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
    raise SystemExit("mode_optics computation is added in Task 8")


if __name__ == "__main__":
    sys.exit(main())
