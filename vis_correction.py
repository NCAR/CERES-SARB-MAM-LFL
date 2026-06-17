import numpy as np
import xarray as xr


def compute_vis_factor(
    external_column,
    internal_column,
    min_aod=1.0e-8,
    min_factor=0.25,
    max_factor=4.0,
):
    external_column, internal_column = xr.align(external_column, internal_column, join="exact", copy=False)

    tiny = (external_column <= min_aod) & (internal_column <= min_aod)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = external_column / internal_column
    raw = xr.where(internal_column > 0.0, raw, np.inf)

    clipped = raw.clip(min=min_factor, max=max_factor)
    factor = xr.where(tiny, 1.0, clipped).astype(np.float32)
    capped = ((raw < min_factor) | (raw > max_factor)) & (~tiny)
    stats = {
        "capped": int(capped.sum().item()),
        "skipped": int(tiny.sum().item()),
    }
    return factor, stats


def apply_column_factor(layer_field, factor):
    if "lev" not in layer_field.dims:
        raise ValueError("layer_field must include lev dimension")

    layer_field, factor = xr.align(layer_field, factor, join="exact", copy=False)
    return (layer_field * factor).astype(np.float32)
