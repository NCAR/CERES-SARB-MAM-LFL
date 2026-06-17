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


def _required_field_name(source_spec, field_name):
    fields = source_spec.get("fields", {})
    if field_name not in fields:
        raise KeyError(field_name)
    return fields[field_name]


def read_source_fields_from_dataset(ds, source_spec, species_names):
    rh_var = _required_field_name(source_spec, "rh")
    delp_var = _required_field_name(source_spec, "delp")

    rh_da = _require_var(ds, rh_var)
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
