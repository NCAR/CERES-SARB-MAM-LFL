import argparse
import os

import numpy as np
import xarray as xr
import yaml

from mode_optics import COL


G = 9.8
SPECIES = (
    ["SO4", "OCPHOBIC", "OCPHILIC", "BCPHOBIC", "BCPHILIC"]
    + ["SS%03d" % index for index in range(1, 6)]
    + ["DU%03d" % index for index in range(1, 6)]
    + ["NO3AN%d" % index for index in range(1, 4)]
)


def _fmt(value):
    return "%.6g" % float(value)


def _type_key(species):
    if species == "SO4":
        return "SU"
    if species == "OCPHOBIC":
        return "OCPHO"
    if species == "OCPHILIC":
        return "OCPHI"
    if species == "BCPHOBIC":
        return "BCPHO"
    if species == "BCPHILIC":
        return "BCPHI"
    if species.startswith("SS"):
        return "SS"
    if species.startswith("DU"):
        return "DU"
    if species.startswith("NO3AN"):
        return "NI"
    raise KeyError(species)


def _size_index(species):
    if species.startswith("SS") or species.startswith("DU"):
        return int(species[-3:]) - 1
    if species.startswith("NO3AN"):
        return int(species[-1]) - 1
    if species == "OCPHILIC":
        return 1
    return 0


def _area_weighted_mean_2d(field, lat):
    array = np.asarray(field, dtype=np.float64)
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return float((array * weights[:, None]).sum() / (weights.sum() * array.shape[1]))


def _column_mean(layer, lat):
    array = np.asarray(layer, dtype=np.float64)
    if array.ndim == 4:
        array = array[0]
    column = array.sum(axis=0)
    return _area_weighted_mean_2d(column, lat)


def _file_column_mean(path):
    with xr.open_dataset(path) as ds:
        column = ds[COL]
        if "time" in column.dims and column.sizes["time"] == 1:
            column = column.isel(time=0)
        return _area_weighted_mean_2d(column.values, column["lat"].values)


def _open_optics(path):
    with xr.open_dataset(path) as ds:
        rh_fine = np.arange(0.0, 1.0, 0.01)
        return ds[["bext", "bsca", "g", "lambda"]].interp(rh=rh_fine).load()


def _band_index(ds_optics, bands_path, band):
    label = str(band).lower()
    band_number = int(label[2:4])
    family = label[:2]
    with xr.open_dataset(bands_path) as ds_bands:
        if family == "sw":
            bounds = ds_bands["LFL_SW_bands"].values
        elif family == "lw":
            bounds = ds_bands["LFL_LW_bands"].values
        else:
            raise ValueError("band must start with sw or lw")
    lower = float(bounds[band_number - 1])
    upper = float(bounds[band_number])
    target_um = 0.5 * (lower + upper)
    wavelengths_um = np.asarray(ds_optics["lambda"].values, dtype=np.float64) * 1.0e6
    index = int(np.argmin(np.abs(wavelengths_um - target_um)))
    return index, lower, upper, float(wavelengths_um[index])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose old external-species optical-depth formula")
    parser.add_argument("--aerosol", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--band", default="sw05")
    parser.add_argument("--optics-dir", default=None)
    parser.add_argument("--reference-total", default=None)
    args = parser.parse_args(argv)

    if args.optics_dir:
        os.environ["OPTICS_DIR"] = args.optics_dir

    with open(args.aerosol, "r") as stream:
        config = yaml.safe_load(stream)

    bands_path = os.path.expandvars(config["filename_bands"])
    optics_cache = {}

    with xr.open_dataset(args.input) as ds:
        rh = np.asarray(ds["RH"].values, dtype=np.float32)
        delp = np.asarray(ds["DELP"].values, dtype=np.float32)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)

        print("config: %s" % args.aerosol)
        print("input: %s" % args.input)
        print("band: %s" % args.band.upper())
        print("bands file: %s" % bands_path)

        total_tau = np.zeros_like(delp, dtype=np.float32)
        total_mass_col = 0.0
        total_aod = 0.0

        for species in SPECIES:
            if species not in ds:
                print("%-12s missing" % species)
                continue

            type_key = _type_key(species)
            optics_path = os.path.expandvars(config["Types"][type_key]["filename"])
            if optics_path not in optics_cache:
                optics_cache[optics_path] = _open_optics(optics_path)
            ds_optics = optics_cache[optics_path]
            idx_wvl, lower, upper, selected_um = _band_index(ds_optics, bands_path, args.band)

            q = np.asarray(ds[species].values, dtype=np.float32)
            if "PHO" in species:
                idx_rh = np.zeros(rh.size, dtype=np.int32)
            else:
                idx_rh = np.floor(rh.ravel() * 100.0).astype(np.int32)
                idx_rh[idx_rh > 99] = 99
                idx_rh[idx_rh < 0] = 0

            idx_size = _size_index(species)
            k_ext = ds_optics["bext"].values[idx_size, idx_rh, idx_wvl].reshape(rh.shape)
            tau = (delp * q * k_ext / G).astype(np.float32)
            aod = _column_mean(tau, lat)
            mass_col = _column_mean(q * delp / G, lat)
            total_tau += tau
            total_mass_col += mass_col
            total_aod += aod
            mee = aod / mass_col if mass_col > 0.0 else 0.0
            print(
                "%-12s AOD=%s mass_kg_m2=%s eff_m2_kg=%s optics=%s size=%d"
                % (species, _fmt(aod), _fmt(mass_col), _fmt(mee), os.path.basename(optics_path), idx_size + 1)
            )

        print("band bounds: [%s,%s] selected_optics_um=%s" % (_fmt(lower), _fmt(upper), _fmt(selected_um)))
        print("total external-species AOD: %s" % _fmt(total_aod))
        print("total dry mass kg/m2: %s" % _fmt(total_mass_col))
        if total_mass_col > 0.0:
            print("total effective m2/kg: %s" % _fmt(total_aod / total_mass_col))
        if args.reference_total:
            reference = _file_column_mean(args.reference_total)
            print("reference total AOD: %s" % _fmt(reference))
            print("reference/external ratio: %s" % _fmt(reference / total_aod))


if __name__ == "__main__":
    main()
