import unittest

import numpy as np
import xarray as xr

from vis_correction import apply_column_factor, compute_vis_factor


class TestVisCorrection(unittest.TestCase):
    def test_compute_factor_caps_high_ratios_and_skips_tiny_columns(self):
        external = xr.DataArray(
            np.array([[2.0, 1.0, 0.0]], dtype=np.float32),
            dims=("lat", "lon"),
            coords={"lat": [45.0], "lon": [10.0, 20.0, 30.0]},
        )
        internal = xr.DataArray(
            np.array([[1.0, 0.1, 0.0]], dtype=np.float32),
            dims=("lat", "lon"),
            coords=external.coords,
        )

        factor, stats = compute_vis_factor(
            external,
            internal,
            min_aod=1.0e-8,
            min_factor=0.25,
            max_factor=4.0,
        )

        self.assertEqual(factor.dtype, np.float32)
        self.assertEqual(factor.dims, external.dims)
        np.testing.assert_allclose(factor.coords["lat"].values, external.coords["lat"].values)
        np.testing.assert_allclose(factor.coords["lon"].values, external.coords["lon"].values)
        np.testing.assert_allclose(factor.values, np.array([[2.0, 4.0, 1.0]], dtype=np.float32))
        self.assertEqual(stats, {"capped": 1, "skipped": 1})

    def test_compute_factor_handles_zero_internal_and_tiny_external_edges(self):
        external = xr.DataArray(
            np.array([[1.0, 1.0e-10, 0.0, 10.0]], dtype=np.float32),
            dims=("lat", "lon"),
            coords={"lat": [0.0], "lon": [0.0, 90.0, 180.0, 270.0]},
        )
        internal = xr.DataArray(
            np.array([[0.0, 1.0e-6, 0.0, 1.0]], dtype=np.float32),
            dims=("lat", "lon"),
            coords=external.coords,
        )

        factor, stats = compute_vis_factor(
            external,
            internal,
            min_aod=1.0e-8,
            min_factor=0.25,
            max_factor=4.0,
        )

        np.testing.assert_allclose(factor.values, np.array([[4.0, 0.25, 1.0, 4.0]], dtype=np.float32))
        self.assertEqual(stats["capped"], 3)
        self.assertEqual(stats["skipped"], 1)

    def test_compute_factor_skips_columns_equal_to_tiny_threshold(self):
        min_aod = 1.0e-8
        external = xr.DataArray(np.array([[min_aod]], dtype=np.float32), dims=("lat", "lon"))
        internal = xr.DataArray(np.array([[min_aod]], dtype=np.float32), dims=("lat", "lon"))

        factor, stats = compute_vis_factor(external, internal, min_aod=min_aod)

        self.assertAlmostEqual(float(factor.values[0, 0]), 1.0)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["capped"], 0)

    def test_apply_column_factor_scales_every_layer_and_preserves_column_scaling(self):
        layer = xr.DataArray(
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            dims=("lev", "lat", "lon"),
            coords={"lev": [1000.0, 850.0, 700.0], "lat": [-30.0, 30.0], "lon": [0.0, 180.0]},
        )
        factor = xr.DataArray(
            np.array([[2.0, 3.0], [0.5, 4.0]], dtype=np.float32),
            dims=("lat", "lon"),
            coords={"lat": layer.coords["lat"], "lon": layer.coords["lon"]},
        )

        scaled = apply_column_factor(layer, factor)

        expected = layer.values * factor.values[np.newaxis, :, :]
        np.testing.assert_allclose(scaled.values, expected)
        np.testing.assert_allclose(scaled.sum("lev").values, layer.sum("lev").values * factor.values)
        self.assertEqual(scaled.dtype, np.float32)

    def test_apply_column_factor_preserves_dataarray_dims_and_coords(self):
        layer = xr.DataArray(
            np.ones((2, 3, 2, 2), dtype=np.float32),
            dims=("time", "lev", "lat", "lon"),
            coords={
                "time": np.array([0, 1], dtype=np.int32),
                "lev": np.array([1000.0, 850.0, 700.0], dtype=np.float32),
                "lat": np.array([-45.0, 45.0], dtype=np.float32),
                "lon": np.array([0.0, 180.0], dtype=np.float32),
            },
        )
        factor = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            dims=("lat", "lon"),
            coords={"lat": layer.coords["lat"], "lon": layer.coords["lon"]},
        )

        scaled = apply_column_factor(layer, factor)

        self.assertEqual(scaled.dims, layer.dims)
        for coord in layer.coords:
            np.testing.assert_allclose(scaled.coords[coord].values, layer.coords[coord].values)

    def test_apply_column_factor_rejects_fields_without_lev(self):
        layer = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("lat", "lon"))
        factor = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("lat", "lon"))

        with self.assertRaisesRegex(ValueError, "lev"):
            apply_column_factor(layer, factor)


if __name__ == "__main__":
    unittest.main()
