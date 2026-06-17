import unittest

import numpy as np
import xarray as xr

from source_fields import read_source_fields_from_dataset


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

    def test_temperature_falls_back_to_freezing(self):
        ds = self.native_dataset(include_temperature=False)

        fields = read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])

        np.testing.assert_allclose(fields.temperature, np.full_like(fields.rh, 273.15))

    def test_missing_required_variable_raises(self):
        ds = self.native_dataset(include_delp=False)

        with self.assertRaisesRegex(KeyError, "DELP"):
            read_source_fields_from_dataset(ds, self.source_spec(), ["SO4"])


if __name__ == "__main__":
    unittest.main()
