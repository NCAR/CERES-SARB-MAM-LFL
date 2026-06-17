import numpy as np
import xarray as xr


def compute_vis_factor(
    external_column,
    internal_column,
    min_aod=1.0e-8,
    min_factor=0.25,
    max_factor=4.0,
):
    if not isinstance(external_column, xr.DataArray) or not isinstance(internal_column, xr.DataArray):
        raise ValueError("external_column and internal_column must be 2D xarray DataArrays")
    if len(external_column.dims) != 2 or len(internal_column.dims) != 2:
        raise ValueError("external_column and internal_column must be 2D xarray DataArrays")
    if external_column.dims != internal_column.dims:
        raise ValueError("external_column and internal_column dims must match")

    external_column, internal_column = xr.align(external_column, internal_column, join="exact", copy=False)

    finite = np.isfinite(external_column) & np.isfinite(internal_column)
    tiny = finite & (external_column <= min_aod) & (internal_column <= min_aod)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = external_column / internal_column

    clipped = raw.clip(min=min_factor, max=max_factor)
    factor = xr.where((~finite) | tiny, 1.0, clipped).astype(np.float32)
    capped = finite & (~tiny) & ((raw < min_factor) | (raw > max_factor))
    stats = {
        "capped": int(capped.sum().item()),
        "skipped": int(tiny.sum().item()),
    }
    return factor, stats


def apply_column_factor(layer_field, factor):
    if "lev" not in layer_field.dims:
        raise ValueError("layer_field must include lev dimension")
    expected_factor_dims = tuple(dim for dim in layer_field.dims if dim != "lev")
    if factor.dims != expected_factor_dims:
        raise ValueError("factor dims must match layer_field dims except lev")

    layer_field, factor = xr.align(layer_field, factor, join="exact", copy=False)
    return (layer_field * factor).astype(np.float32)
