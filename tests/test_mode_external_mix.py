import unittest

import numpy as np
import xarray as xr

from mode_external_mix import mix_mode_datasets


def mode_dataset(ext, sca, asm, delp=None):
    dims = ("time", "lev", "lat", "lon")
    ext = np.asarray(ext, dtype=np.float32)
    sca = np.asarray(sca, dtype=np.float32)
    asm = np.asarray(asm, dtype=np.float32)
    coords = {
        "time": np.arange(ext.shape[0], dtype=np.int32),
        "lev": np.linspace(1000.0, 850.0, ext.shape[1], dtype=np.float32),
        "lat": np.linspace(-45.0, 45.0, ext.shape[2], dtype=np.float32),
        "lon": np.linspace(0.0, 360.0, ext.shape[3], endpoint=False, dtype=np.float32),
    }
    if delp is None:
        delp = np.full(ext.shape, 100.0, dtype=np.float32)

    ext_array = xr.DataArray(ext, dims=dims, coords=coords)
    sca_array = xr.DataArray(sca, dims=dims, coords=coords)
    asm_array = xr.DataArray(asm, dims=dims, coords=coords)
    delp_array = xr.DataArray(np.asarray(delp, dtype=np.float32), dims=dims, coords=coords)

    return xr.Dataset(
        {
            "DELP": delp_array,
            "Extinction_Layer_Optical_Depth": ext_array,
            "Scattering_Layer_Optical_Depth": sca_array,
            "Layer_Asymmetry_Parameter": asm_array,
            "Extinction_Column_Optical_Depth": ext_array.sum("lev"),
        }
    )


class TestModeExternalMix(unittest.TestCase):
    def test_sums_extinction_and_scattering_and_scattering_weights_asymmetry(self):
        first = mode_dataset(
            ext=np.array([[[[0.2]], [[0.4]]]], dtype=np.float32),
            sca=np.array([[[[0.1]], [[0.3]]]], dtype=np.float32),
            asm=np.array([[[[0.5]], [[0.8]]]], dtype=np.float32),
        )
        second = mode_dataset(
            ext=np.array([[[[0.3]], [[0.1]]]], dtype=np.float32),
            sca=np.array([[[[0.2]], [[0.1]]]], dtype=np.float32),
            asm=np.array([[[[0.2]], [[0.4]]]], dtype=np.float32),
        )

        mixed = mix_mode_datasets([first, second])

        np.testing.assert_allclose(
            mixed["Extinction_Layer_Optical_Depth"].values,
            np.array([[[[0.5]], [[0.5]]]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            mixed["Scattering_Layer_Optical_Depth"].values,
            np.array([[[[0.3]], [[0.4]]]], dtype=np.float32),
        )
        expected_asm = np.array(
            [[
                [[(0.1 * 0.5 + 0.2 * 0.2) / 0.3]],
                [[(0.3 * 0.8 + 0.1 * 0.4) / 0.4]],
            ]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(mixed["Layer_Asymmetry_Parameter"].values, expected_asm)
        np.testing.assert_allclose(
            mixed["Extinction_Column_Optical_Depth"].values,
            np.array([[[1.0]]], dtype=np.float32),
        )
        self.assertEqual(mixed["Extinction_Layer_Optical_Depth"].dtype, np.float32)
        self.assertEqual(mixed["Scattering_Layer_Optical_Depth"].dtype, np.float32)
        self.assertEqual(mixed["Layer_Asymmetry_Parameter"].dtype, np.float32)
        self.assertEqual(mixed["Extinction_Column_Optical_Depth"].dtype, np.float32)
        self.assertEqual(mixed["Extinction_Layer_Optical_Depth"].dims, first["Extinction_Layer_Optical_Depth"].dims)
        np.testing.assert_allclose(mixed.coords["lev"].values, first.coords["lev"].values)

    def test_zero_scattering_returns_zero_asymmetry(self):
        first = mode_dataset(
            ext=np.array([[[[0.2]], [[0.4]]]], dtype=np.float32),
            sca=np.zeros((1, 2, 1, 1), dtype=np.float32),
            asm=np.array([[[[0.5]], [[0.8]]]], dtype=np.float32),
        )
        second = mode_dataset(
            ext=np.array([[[[0.3]], [[0.1]]]], dtype=np.float32),
            sca=np.zeros((1, 2, 1, 1), dtype=np.float32),
            asm=np.array([[[[0.2]], [[0.4]]]], dtype=np.float32),
        )

        mixed = mix_mode_datasets([first, second])

        np.testing.assert_allclose(
            mixed["Layer_Asymmetry_Parameter"].values,
            np.zeros((1, 2, 1, 1), dtype=np.float32),
        )

    def test_empty_input_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            mix_mode_datasets([])

    def test_mismatched_layer_dims_or_shapes_raise_value_error(self):
        first = mode_dataset(
            ext=np.ones((1, 2, 1, 1), dtype=np.float32),
            sca=np.ones((1, 2, 1, 1), dtype=np.float32),
            asm=np.ones((1, 2, 1, 1), dtype=np.float32),
        )
        mismatched_dims = first.rename({"lev": "level"})
        mismatched_shape = mode_dataset(
            ext=np.ones((1, 2, 1, 2), dtype=np.float32),
            sca=np.ones((1, 2, 1, 2), dtype=np.float32),
            asm=np.ones((1, 2, 1, 2), dtype=np.float32),
            delp=np.ones((1, 2, 1, 2), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "dims"):
            mix_mode_datasets([first, mismatched_dims])
        with self.assertRaisesRegex(ValueError, "shape"):
            mix_mode_datasets([first, mismatched_shape])

    def test_scattering_greater_than_extinction_raises_value_error(self):
        bad = mode_dataset(
            ext=np.array([[[[0.1]], [[0.2]]]], dtype=np.float32),
            sca=np.array([[[[0.2]], [[0.1]]]], dtype=np.float32),
            asm=np.zeros((1, 2, 1, 1), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "Scattering_Layer_Optical_Depth.*Extinction_Layer_Optical_Depth"):
            mix_mode_datasets([bad])


if __name__ == "__main__":
    unittest.main()
