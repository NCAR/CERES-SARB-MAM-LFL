import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from source_fields import open_source_fields, read_source_fields_from_dataset


class TestSourceFields(unittest.TestCase):
    def native_dataset(self, include_temperature=True, include_delp=True):
        dims = ("time", "lev", "lat", "lon")
        shape = (1, 2, 2, 3)
        coords = {
            "time": np.array([0]),
            "lev": np.array([1000.0, 900.0], dtype=np.float32),
            "lat": np.array([-10.0, 10.0], dtype=np.float32),
            "lon": np.array([0.0, 120.0, 240.0], dtype=np.float32),
        }
        data_vars = {
            "RH": (dims, np.linspace(0.1, 0.9, num=np.prod(shape), dtype=np.float32).reshape(shape)),
            "SO4": (dims, np.full(shape, 1.0e-9, dtype=np.float32)),
            "BCPHILIC": (dims, np.full(shape, 2.0e-9, dtype=np.float32)),
        }
        if include_temperature:
            data_vars["T"] = (dims, np.full(shape, 280.0, dtype=np.float32))
        if include_delp:
            data_vars["DELP"] = (dims, np.full(shape, 100.0, dtype=np.float32))
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def source_spec(self):
        return {
            "fields": {
                "rh": "RH",
                "temperature": "T",
                "delp": "DELP",
            },
            "species": {
                "SO4": "SO4",
                "BCPHILIC": "BCPHILIC",
            },
        }

    def source_spec_with_dims(self):
        spec = self.source_spec()
        spec["dims"] = {
            "time": "time",
            "lev": "lev",
            "lat": "lat",
            "lon": "lon",
        }
        return spec

    def reordered_dataset(self):
        dims = ("time", "lat", "lev", "lon")
        shape = (1, 2, 2, 3)
        coords = {
            "time": np.array([0]),
            "lat": np.array([-10.0, 10.0], dtype=np.float32),
            "lev": np.array([1000.0, 900.0], dtype=np.float32),
            "lon": np.array([0.0, 120.0, 240.0], dtype=np.float32),
        }
        data_vars = {
            "RH": (dims, np.linspace(0.1, 0.9, num=np.prod(shape), dtype=np.float32).reshape(shape)),
            "T": (dims, np.full(shape, 280.0, dtype=np.float32)),
            "DELP": (dims, np.full(shape, 100.0, dtype=np.float32)),
            "SO4": (dims, np.full(shape, 1.0e-9, dtype=np.float32)),
        }
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def test_reads_native_fields_and_species(self):
        ds = self.native_dataset()

        fields = read_source_fields_from_dataset(
            ds,
            self.source_spec(),
            ["SO4", "BCPHILIC", "UNMAPPED"],
        )

        self.assertEqual(fields.rh.shape, (1, 2, 2, 3))
        self.assertEqual(fields.rh.dtype, np.float32)
        self.assertAlmostEqual(float(fields.temperature.mean()), 280.0)
        self.assertEqual(fields.delp.shape, (1, 2, 2, 3))
        self.assertIn("SO4", fields.species)
        self.assertIn("BCPHILIC", fields.species)
        np.testing.assert_allclose(fields.species["UNMAPPED"], np.zeros_like(fields.rh))
        self.assertEqual(fields.dims, ("time", "lev", "lat", "lon"))
        self.assertEqual(set(fields.coords), {"time", "lev", "lat", "lon"})

    def test_preserves_xarray_coordinate_metadata(self):
        ds = self.native_dataset()
        ds.coords["lat"].attrs["units"] = "degrees_north"

        fields = read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])

        self.assertIsInstance(fields.coords["lat"], xr.DataArray)
        self.assertEqual(fields.coords["lat"].dims, ("lat",))
        self.assertEqual(fields.coords["lat"].attrs["units"], "degrees_north")
        np.testing.assert_allclose(fields.coords["lat"].values, ds.coords["lat"].values)

    def test_copied_coords_do_not_mutate_original_attrs(self):
        ds = self.native_dataset()
        ds.coords["lat"].attrs["units"] = "degrees_north"

        fields = read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])
        fields.coords["lat"].attrs["units"] = "changed"

        self.assertEqual(ds.coords["lat"].attrs["units"], "degrees_north")

    def test_temperature_falls_back_to_freezing(self):
        ds = self.native_dataset(include_temperature=False)

        fields = read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])

        np.testing.assert_allclose(fields.temperature, np.full_like(fields.rh, 273.15))

    def test_missing_required_variable_raises(self):
        ds = self.native_dataset(include_delp=False)

        with self.assertRaisesRegex(KeyError, "DELP"):
            read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])

    def test_delp_shape_mismatch_raises_value_error(self):
        ds = self.native_dataset()
        ds["DELP"] = (("time", "lev", "lat"), np.full((1, 2, 2), 100.0, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "DELP"):
            read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])

    def test_species_reordered_dims_raises_value_error(self):
        ds = self.native_dataset()
        ds["SO4"] = (("time", "lev", "lon", "lat"), np.full((1, 2, 3, 2), 1.0e-9, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "SO4"):
            read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])

    def test_rh_reordered_dims_raise_when_source_spec_dims_provided(self):
        ds = self.reordered_dataset()

        with self.assertRaisesRegex(ValueError, "RH"):
            read_source_fields_from_dataset(ds, self.source_spec_with_dims(), ["SO4"])

    def test_mapped_missing_species_raises_key_error(self):
        ds = self.native_dataset().drop_vars("BCPHILIC")

        with self.assertRaisesRegex(KeyError, "BCPHILIC"):
            read_source_fields_from_dataset(ds, self.source_spec(), ["BCPHILIC"])

    def test_open_source_fields_loads_and_closes_dataset(self):
        class TrackingDataset:
            def __init__(self, dataset):
                self.dataset = dataset
                self.loaded = False
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.closed = True

            def load(self):
                self.loaded = True
                return self.dataset.load()

        tracking = TrackingDataset(self.native_dataset())

        with patch("source_fields.xr.open_dataset", return_value=tracking) as open_dataset:
            fields = open_source_fields("native.nc", self.source_spec(), ["SO4"])

        open_dataset.assert_called_once_with("native.nc")
        self.assertTrue(tracking.loaded)
        self.assertTrue(tracking.closed)
        self.assertEqual(fields.rh.shape, (1, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
