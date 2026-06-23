from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class SourceFields:
    dataset: xr.Dataset
    rh: np.ndarray
    temperature: np.ndarray
    delp: np.ndarray
    species: dict
    coords: dict
    dims: tuple


def _as_float32_array(data_array):
    return np.asarray(data_array.values, dtype=np.float32)


def _require_var(ds, varname):
    if varname not in ds:
        raise KeyError(varname)
    return ds[varname]


def _validate_like(reference_da, candidate_da, name):
    if candidate_da.dims != reference_da.dims or candidate_da.shape != reference_da.shape:
        raise ValueError(
            "%s dims %s shape %s do not match %s dims %s shape %s"
            % (
                name,
                candidate_da.dims,
                candidate_da.shape,
                reference_da.name,
                reference_da.dims,
                reference_da.shape,
            )
        )


def _validate_reference_dims(reference_da, source_spec, name):
    dims_spec = source_spec.get("dims")
    if not dims_spec:
        return
    expected = tuple(dims_spec[key] for key in ("time", "lev", "lat", "lon") if key in dims_spec)
    if reference_da.dims != expected:
        raise ValueError("%s dims %s do not match expected dims %s" % (name, reference_da.dims, expected))


def _required_field_name(source_spec, field_name):
    fields = source_spec.get("fields", {})
    if field_name not in fields:
        raise KeyError(field_name)
    return fields[field_name]


DRY_AIR_GAS_CONSTANT = 287.05  # J kg-1 K-1
# GEOS-IT 72-level model top. PS - sum(DELP) only recovers this to ~+/-20 Pa
# (float32 rounding over 72 summed layers), and that scatter blows up T in the
# near-vacuum top layers, so anchor on the known fixed value instead.
MODEL_TOP_PA = 1.0
# Physical guard mirroring the Kohler solver's own clamp (mode_physics); keeps
# the residual top-of-atmosphere ill-conditioning out of the stored field.
TEMPERATURE_BOUNDS_K = (180.0, 330.0)


def _lev_axis(reference_da, source_spec):
    lev_name = source_spec.get("dims", {}).get("lev", "lev")
    if lev_name not in reference_da.dims:
        raise ValueError("lev dim %r not found in %s" % (lev_name, reference_da.name))
    return reference_da.dims.index(lev_name)


def _derive_temperature_from_density(ds, source_spec, reference_da, rh, delp):
    """Reconstruct air temperature from density and the hydrostatic pressure grid.

    GEOS-IT ``aer_inst`` carries no air temperature, only air density (AIRDENS)
    and the layer pressure thickness (DELP). The mid-layer pressure follows from
    a top-of-atmosphere anchor ``PTOP = PS - sum(DELP)`` plus the running DELP
    integral, and the ideal-gas law gives ``T = P / (R_d * rho)``. This is far
    more faithful for the Kohler growth than a constant fallback. Returns None
    when the required fields are absent so the caller can fall back.
    """
    fields = source_spec.get("fields", {})
    density_var = fields.get("airdens", "AIRDENS")
    if density_var not in ds:
        return None

    density_da = _require_var(ds, density_var)
    _validate_like(reference_da, density_da, density_var)
    density = _as_float32_array(density_da).astype(np.float64)

    axis = _lev_axis(reference_da, source_spec)
    delp64 = delp.astype(np.float64)
    # Mid-layer pressure from the fixed model top down: PTOP + cumsum(DELP) - DELP/2.
    # Layers are ordered top -> bottom (lev positive = down).
    p_mid = MODEL_TOP_PA + np.cumsum(delp64, axis=axis) - 0.5 * delp64
    with np.errstate(divide="ignore", invalid="ignore"):
        temperature = p_mid / (density * DRY_AIR_GAS_CONSTANT)
    lo, hi = TEMPERATURE_BOUNDS_K
    temperature = np.where(np.isfinite(temperature), temperature, lo)
    return np.clip(temperature, lo, hi).astype(np.float32)


def read_source_fields_from_dataset(ds, source_spec, species_names):
    rh_var = _required_field_name(source_spec, "rh")
    delp_var = _required_field_name(source_spec, "delp")

    rh_da = _require_var(ds, rh_var)
    _validate_reference_dims(rh_da, source_spec, rh_var)
    delp_da = _require_var(ds, delp_var)
    _validate_like(rh_da, delp_da, delp_var)

    rh = _as_float32_array(rh_da)
    delp = _as_float32_array(delp_da)

    temperature_var = source_spec.get("fields", {}).get("temperature")
    if temperature_var in ds:
        temperature_da = _require_var(ds, temperature_var)
        _validate_like(rh_da, temperature_da, temperature_var)
        temperature = _as_float32_array(temperature_da)
    else:
        temperature = _derive_temperature_from_density(ds, source_spec, rh_da, rh, delp)
        if temperature is None:
            temperature = np.full_like(rh, 273.15, dtype=np.float32)

    species_mapping = source_spec.get("species", {})
    species = {}
    for species_name in species_names:
        varname = species_mapping.get(species_name)
        if varname is None:
            species[species_name] = np.zeros_like(rh, dtype=np.float32)
            continue
        species_da = _require_var(ds, varname)
        _validate_like(rh_da, species_da, varname)
        species[species_name] = _as_float32_array(species_da)

    coords = {name: ds.coords[name].copy(deep=True) for name in ds.coords}
    dims = tuple(rh_da.dims)

    return SourceFields(
        dataset=ds,
        rh=rh,
        temperature=temperature,
        delp=delp,
        species=species,
        coords=coords,
        dims=dims,
    )


def open_source_fields(path, source_spec, species_names):
    with xr.open_dataset(path) as ds:
        loaded = ds.load()
    return read_source_fields_from_dataset(loaded, source_spec, species_names)
