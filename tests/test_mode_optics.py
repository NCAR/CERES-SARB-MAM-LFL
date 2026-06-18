import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import xarray as xr

import mode_optics
from source_fields import SourceFields


class TestModeOpticsHelpers(unittest.TestCase):
    def test_type_key_maps_native_species_to_optical_types(self):
        cases = {
            "SO4": "SU",
            "NO3AN1": "NI",
            "OCPHILIC": "POM",
            "OCPHOBIC": "POM",
            "BCPHILIC": "BC",
            "BCPHOBIC": "BC",
            "DU001": "DU",
            "SS001": "SS",
        }

        for species, expected in cases.items():
            with self.subTest(species=species):
                self.assertEqual(mode_optics._type_key(species), expected)

    def test_mode_species_returns_positive_allocations_for_mode(self):
        config = {
            "Schemes": {
                "MAM4": {
                    "modes": {
                        "a1": {"dry_radius_um": 0.05, "sigma_g": 1.6},
                        "a2": {"dry_radius_um": 0.01, "sigma_g": 1.6},
                    },
                    "allocations": {
                        "SO4": {"a1": 0.9, "a2": 0.1},
                        "OCPHOBIC": {"a1": 0.0, "a2": 1.0},
                        "BCPHILIC": {"a1": 1.0},
                    },
                }
            }
        }

        species = mode_optics._mode_species(config, "MAM4", "a1")

        self.assertEqual(species, ["SO4", "BCPHILIC"])

    def test_build_path_fills_date_label_and_band(self):
        date_str = "2020-01-Jan-02-002-03"

        relative = mode_optics._build_path(
            "/root",
            "YYYY/MM/DD/{label}_{band}_HH.nc",
            date_str,
            "MAM4_a1",
            "SW01",
        )
        absolute = mode_optics._build_path(
            "/root",
            "/abs/YYYY/MM/DD/{label}_{band}_HH.nc",
            date_str,
            "MAM4_a1",
            "SW01",
        )

        self.assertEqual(relative, os.path.join("/root", "2020/01/02/MAM4_a1_SW01_03.nc"))
        self.assertEqual(absolute, "/abs/2020/01/02/MAM4_a1_SW01_03.nc")

    def test_band_label_uses_wavelength_or_uppercase_band(self):
        self.assertEqual(mode_optics._band_label(SimpleNamespace(wvl=550.0, band=None)), "550NM")
        self.assertEqual(mode_optics._band_label(SimpleNamespace(wvl=None, band="sw01")), "SW01")

    def test_band_wavelength_uses_wvl_or_lfl_band_midpoint(self):
        ds_table = xr.Dataset(
            {
                "LFL_SW_bands": (
                    ("band", "bound"),
                    np.array([[0.20, 0.90], [0.90, 1.20]], dtype=np.float32),
                ),
                "LFL_LW_bands": (
                    ("band", "bound"),
                    np.array([[5.0, 7.0], [7.0, 9.0]], dtype=np.float32),
                ),
            }
        )

        self.assertAlmostEqual(
            mode_optics._band_wavelength_um(SimpleNamespace(wvl=550.0, band=None), ds_table),
            0.55,
        )
        self.assertAlmostEqual(
            mode_optics._band_wavelength_um(SimpleNamespace(wvl=None, band="sw01"), ds_table),
            0.55,
        )
        self.assertAlmostEqual(
            mode_optics._band_wavelength_um(SimpleNamespace(wvl=None, band="lw01"), ds_table),
            6.0,
        )

    def test_mode_table_path_uses_raw_band_or_lowercase_wavelength_label(self):
        mode_spec = {"filename_sarb": "/tables/mam4_mode1_larc_c000002.nc"}

        band_path = mode_optics._mode_table_path(
            mode_spec,
            SimpleNamespace(wvl=None, band="sw01"),
        )
        wavelength_path = mode_optics._mode_table_path(
            mode_spec,
            SimpleNamespace(wvl=550.0, band=None),
        )

        self.assertEqual(band_path, "/tables/mam4_mode1_sw01_larc_c000002.nc")
        self.assertEqual(wavelength_path, "/tables/mam4_mode1_550nm_larc_c000002.nc")


class TestModeOpticsRun(unittest.TestCase):
    def _config(self):
        return {
            "Types": {
                "SU": {
                    "density": 1.7,
                    "hygroscopicity": [2.4, -3.8, 1.9],
                    "filename": "optics_SU.nc",
                },
                "WAT": {
                    "density": 1.0,
                    "hygroscopicity": [0.0, 0.0, 0.0],
                    "filename": "optics_WAT.nc",
                },
            },
            "Sources": {
                "TEST": {
                    "input_pattern": "input/YYYY/MM/native.YYYY-MM-DDTHH.nc",
                    "output_pattern": "output/YYYY/MM/{label}_{band}.YYYY-MM-DDTHH.nc",
                    "fields": {"rh": "RH", "temperature": "T", "delp": "DELP"},
                    "species": {"SO4": "SO4"},
                }
            },
            "Schemes": {
                "MAMX": {
                    "modes": {
                        "a1": {
                            "dry_radius_um": 0.05,
                            "sigma_g": 1.6,
                            "filename_sarb": "mode_larc.nc",
                        }
                    },
                    "allocations": {"SO4": {"a1": 1.0}},
                }
            },
        }

    def _fields(self):
        dims = ("time", "lev", "lat", "lon")
        shape = (1, 2, 1, 1)
        coords = {
            "time": xr.DataArray(np.array([0], dtype=np.int32), dims=("time",)),
            "lev": xr.DataArray(np.array([1000.0, 850.0], dtype=np.float32), dims=("lev",)),
            "lat": xr.DataArray(np.array([45.0], dtype=np.float32), dims=("lat",)),
            "lon": xr.DataArray(np.array([270.0], dtype=np.float32), dims=("lon",)),
        }
        return SourceFields(
            dataset=xr.Dataset(),
            rh=np.full(shape, 0.55, dtype=np.float32),
            temperature=np.full(shape, 280.0, dtype=np.float32),
            delp=np.full(shape, 100.0, dtype=np.float32),
            species={"SO4": np.full(shape, 1.0e-9, dtype=np.float32)},
            coords=coords,
            dims=dims,
        )

    def _table(self):
        n_real = np.array([1.3, 1.5, 1.7], dtype=np.float32)
        n_imag = np.array([0.0, 0.1], dtype=np.float32)
        radius = np.array([0.05, 0.10, 0.20], dtype=np.float32)
        ext = np.full((3, 2, 3), 4.0, dtype=np.float32)
        abs_ = np.full((3, 2, 3), 1.0, dtype=np.float32)
        asm = np.full((3, 2, 3), 0.6, dtype=np.float32)
        return xr.Dataset(
            {
                "LFL_SW_bands": (("band", "bound"), np.array([[0.2, 0.9]], dtype=np.float32)),
                "ext": (("n_real", "n_imag", "radius"), ext),
                "abs": (("n_real", "n_imag", "radius"), abs_),
                "asm": (("n_real", "n_imag", "radius"), asm),
            },
            coords={"n_real": n_real, "n_imag": n_imag, "radius": radius},
        )

    def _sulfate_refraction(self):
        return xr.Dataset(
            {
                "refreal": (("lambda",), np.array([1.45, 1.48], dtype=np.float32)),
                "refimag": (("lambda",), np.array([0.01, 0.02], dtype=np.float32)),
            },
            coords={"lambda": np.array([0.50e-6, 0.55e-6], dtype=np.float64)},
        )

    def _water_refraction(self):
        return xr.Dataset(
            {
                "watern": (("wavelength1",), np.array([1.33, 1.34], dtype=np.float32)),
                "wateri": (("wavelength1",), np.array([0.0, 0.0], dtype=np.float32)),
            },
            coords={"wavelength1": np.array([0.50, 0.55], dtype=np.float32)},
        )

    def test_run_writes_single_synthetic_mode_dataset(self):
        written = {}

        opened_paths = []

        def fake_open_dataset(path):
            opened_paths.append(path)
            if path == "mode_sw01_larc.nc":
                return self._table()
            if path == "optics_SU.nc":
                return self._sulfate_refraction()
            if path == "optics_WAT.nc":
                return self._water_refraction()
            raise AssertionError("unexpected open_dataset path %s" % path)

        def fake_to_netcdf(ds, path):
            written["path"] = path
            written["dataset"] = ds.copy(deep=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                source="test",
                scheme="MAMX",
                mode="a1",
                band="sw01",
                wvl=None,
                start="2020-01-01T00",
                end="2020-01-01T00",
                aerosol="unused.yaml",
                datadir=tmpdir,
                outdir=tmpdir,
                external_vis=None,
            )
            config = self._config()
            fields = self._fields()

            with mock.patch("mode_optics.load_config", return_value=config):
                with mock.patch("mode_optics.open_source_fields", return_value=fields) as open_fields:
                    with mock.patch("mode_optics.xr.open_dataset", side_effect=fake_open_dataset):
                        with mock.patch.object(xr.Dataset, "to_netcdf", fake_to_netcdf):
                            result = mode_optics.run(args)

            expected_input = os.path.join(tmpdir, "input/2020/01/native.2020-01-01T00.nc")
            expected_output = os.path.join(tmpdir, "output/2020/01/MAMX_a1_SW01.2020-01-01T00.nc")
            self.assertEqual(result, 0)
            open_fields.assert_called_once_with(
                expected_input,
                config["Sources"]["TEST"],
                ["SO4"],
            )
            self.assertEqual(written["path"], expected_output)

        ds = written["dataset"]
        self.assertIn("mode_sw01_larc.nc", opened_paths)
        self.assertNotIn("mode_SW01_larc.nc", opened_paths)
        self.assertEqual(
            set(ds.data_vars),
            {
                "DELP",
                "Extinction_Layer_Optical_Depth",
                "Scattering_Layer_Optical_Depth",
                "Layer_Asymmetry_Parameter",
                "Extinction_Column_Optical_Depth",
            },
        )
        ext = ds["Extinction_Layer_Optical_Depth"].values
        sca = ds["Scattering_Layer_Optical_Depth"].values
        self.assertTrue(np.all(ext >= 0.0))
        self.assertTrue(np.all(sca >= 0.0))
        self.assertTrue(np.all(sca <= ext))
        self.assertEqual(ds.attrs["source"], "TEST")
        self.assertEqual(ds.attrs["scheme"], "MAMX")
        self.assertEqual(ds.attrs["mode"], "a1")
        self.assertEqual(ds.attrs["band"], "SW01")

    def test_compute_mode_dataset_converts_absorption_cross_section_to_layer_tau(self):
        config = self._config()
        fields = self._fields()
        args = SimpleNamespace(wvl=None, band="sw01")
        shape = fields.rh.shape
        table = self._table()
        ext_cross = np.full(shape, 3.0, dtype=np.float32)
        abs_cross = np.full(shape, 1.0, dtype=np.float32)
        asm = np.full(shape, 0.5, dtype=np.float32)
        calls = []

        def fake_layer_optical_depth(delp, number, cross_section):
            calls.append(np.asarray(cross_section, dtype=np.float32).copy())
            return np.asarray(cross_section, dtype=np.float32) * 10.0

        with mock.patch("mode_optics.xr.open_dataset", return_value=table):
            with mock.patch(
                "mode_optics.mix_mode_state",
                return_value={
                    "dry_volume": np.ones(shape, dtype=np.float32),
                    "r_w_um": np.full(shape, 0.05, dtype=np.float32),
                    "n_re": np.full(shape, 1.5, dtype=np.float32),
                    "n_im": np.zeros(shape, dtype=np.float32),
                },
            ):
                with mock.patch("mode_optics.derive_number_mixing_ratio", return_value=np.ones(shape, dtype=np.float32)):
                    with mock.patch("mode_optics.lookup_mode_optics", return_value=(ext_cross, abs_cross, asm)):
                        with mock.patch("mode_optics._refractive_indices", return_value={"SO4": (1.5, 0.0)}):
                            with mock.patch("mode_optics.layer_optical_depth", side_effect=fake_layer_optical_depth):
                                ds = mode_optics.compute_mode_dataset(
                                    config,
                                    "TEST",
                                    config["Sources"]["TEST"],
                                    "MAMX",
                                    "a1",
                                    "SW01",
                                    args,
                                    fields,
                                )

        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0], ext_cross)
        np.testing.assert_allclose(calls[1], abs_cross)
        np.testing.assert_allclose(
            ds["Extinction_Layer_Optical_Depth"].values,
            np.full(shape, 30.0, dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["Scattering_Layer_Optical_Depth"].values,
            np.full(shape, 20.0, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
