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


def _read_required_variable(ds, varname):
    if varname not in ds:
        raise KeyError(varname)
    return _as_float32_array(ds[varname])


def _required_field_name(source_spec, field_name):
    fields = source_spec.get("fields", {})
    if field_name not in fields:
        raise KeyError(field_name)
    return fields[field_name]


def read_source_fields_from_dataset(ds, source_spec, species_names):
    rh_var = _required_field_name(source_spec, "rh")
    delp_var = _required_field_name(source_spec, "delp")

    rh = _read_required_variable(ds, rh_var)
    delp = _read_required_variable(ds, delp_var)

    temperature_var = source_spec.get("fields", {}).get("temperature")
    if temperature_var in ds:
        temperature = _as_float32_array(ds[temperature_var])
    else:
        temperature = np.full_like(rh, 273.15, dtype=np.float32)

    species_mapping = source_spec.get("species", {})
    species = {}
    for species_name in species_names:
        varname = species_mapping.get(species_name)
        if varname is None:
            species[species_name] = np.zeros_like(rh, dtype=np.float32)
            continue
        species[species_name] = _read_required_variable(ds, varname)

    coords = {name: ds.coords[name] for name in ds.coords}
    dims = tuple(ds[rh_var].dims)

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
    ds = xr.open_dataset(path)
    return read_source_fields_from_dataset(ds, source_spec, species_names)
