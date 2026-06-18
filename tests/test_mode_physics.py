import unittest

import numpy as np
import xarray as xr

from mode_physics import (
    derive_number_mixing_ratio,
    kohler_wet_radius_um,
    layer_optical_depth,
    lookup_mode_optics,
    lognormal_volume_factor,
    mix_mode_state,
)


class TestModePhysics(unittest.TestCase):
    def test_lognormal_volume_factor_is_positive(self):
        factor = lognormal_volume_factor(dry_radius_um=0.05, sigma_g=1.6)

        self.assertGreater(factor, 0.0)

    def test_derive_number_positive_for_positive_volume(self):
        dry_volume = np.full((1, 2, 2, 2), 1.0e-12, dtype=np.float32)

        number = derive_number_mixing_ratio(dry_volume, dry_radius_um=0.05, sigma_g=1.6)

        self.assertEqual(number.shape, dry_volume.shape)
        self.assertEqual(number.dtype, np.float32)
        self.assertTrue(np.all(number > 0.0))

    def test_mix_mode_state_volume_weights_refractive_index(self):
        q = {
            "SO4": np.full((1, 1, 1, 1), 2.0, dtype=np.float32),
            "BC": np.full((1, 1, 1, 1), 1.0, dtype=np.float32),
        }
        species_info = {
            "SO4": {"density": 2.0, "hygroscopicity": [1.0, 0.0, 0.0]},
            "BC": {"density": 1.0, "hygroscopicity": [0.0, 0.0, 0.0]},
        }
        refractive = {"SO4": (1.4, 0.0), "BC": (1.8, 0.4)}
        rh = np.full((1, 1, 1, 1), 0.5, dtype=np.float32)
        temperature = np.full((1, 1, 1, 1), 280.0, dtype=np.float32)

        state = mix_mode_state(species_info, q, refractive, rh, temperature, dry_radius_um=0.05)

        self.assertEqual(set(state), {"dry_volume", "hygroscopicity", "r_w_um", "n_re", "n_im"})
        self.assertAlmostEqual(float(state["dry_volume"][0, 0, 0, 0]), 0.002, places=7)
        self.assertAlmostEqual(float(state["hygroscopicity"][0, 0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(state["n_re"][0, 0, 0, 0]), 1.6, places=5)
        self.assertAlmostEqual(float(state["n_im"][0, 0, 0, 0]), 0.2, places=5)
        self.assertGreaterEqual(float(state["r_w_um"][0, 0, 0, 0]), 0.05)
        for value in state.values():
            self.assertEqual(value.dtype, np.float32)

    def test_mix_mode_state_rejects_empty_q(self):
        with self.assertRaisesRegex(ValueError, "q"):
            mix_mode_state({}, {}, {}, np.array([0.5], dtype=np.float32), np.array([280.0], dtype=np.float32), 0.05)

    def test_kohler_wet_radius_solves_equation(self):
        rh = np.array([0.80], dtype=np.float32)
        temperature = np.array([280.0], dtype=np.float32)
        dry_radius_um = 0.05
        hygroscopicity = np.array([1.0], dtype=np.float32)

        wet = kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh, temperature)

        self.assertEqual(wet.dtype, np.float32)
        self.assertGreater(float(wet[0]), dry_radius_um)
        dry_m = dry_radius_um * 1.0e-6
        wet_m = float(wet[0]) * 1.0e-6
        a = 2.0 * 18.016 * 0.076 / (8.3143e3 * 1000.0 * float(temperature[0]))
        residual = np.log(float(rh[0])) - (
            a / wet_m - float(hygroscopicity[0]) * dry_m ** 3 / (wet_m ** 3 - dry_m ** 3)
        )
        self.assertAlmostEqual(float(residual), 0.0, places=4)

    def test_kohler_returns_dry_radius_for_dry_or_non_hygroscopic_cells(self):
        dry_radius_um = 0.05
        rh = np.array([0.0, 0.8, 0.8], dtype=np.float32)
        hygroscopicity = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        temperature = np.full(3, 280.0, dtype=np.float32)

        wet = kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh, temperature)

        np.testing.assert_allclose(wet, np.full(3, dry_radius_um, dtype=np.float32))

    def test_kohler_high_rh_returns_finite_wet_radius(self):
        wet = kohler_wet_radius_um(
            dry_radius_um=0.05,
            hygroscopicity=np.array([1.0], dtype=np.float32),
            rh=np.array([0.99], dtype=np.float32),
            temperature=np.array([280.0], dtype=np.float32),
        )

        self.assertTrue(np.isfinite(wet[0]))
        self.assertGreater(float(wet[0]), 0.05)

    def test_layer_optical_depth_computes_broadcast_tau(self):
        delp = np.array([[980.0, 1960.0]], dtype=np.float32)
        number = np.array([[2.0], [4.0]], dtype=np.float32)
        cross_section = np.array([[3.0, 5.0]], dtype=np.float32)

        tau = layer_optical_depth(delp, number, cross_section)

        expected = cross_section * 1.0e-12 * number * delp / 9.8
        self.assertEqual(tau.shape, (2, 2))
        self.assertEqual(tau.dtype, np.float32)
        np.testing.assert_allclose(tau, expected.astype(np.float32))

    def test_layer_optical_depth_zero_inputs_return_zero(self):
        delp = np.array([0.0, 980.0, 980.0], dtype=np.float32)
        number = np.array([2.0, 0.0, 2.0], dtype=np.float32)
        cross_section = np.array([3.0, 3.0, 0.0], dtype=np.float32)

        tau = layer_optical_depth(delp, number, cross_section)

        np.testing.assert_array_equal(tau, np.zeros(3, dtype=np.float32))

    def test_layer_optical_depth_rejects_negative_inputs(self):
        positive = np.array([1.0], dtype=np.float32)
        cases = (
            ("delp_pa", np.array([-1.0], dtype=np.float32), positive, positive),
            ("number_per_kg", positive, np.array([-1.0], dtype=np.float32), positive),
            ("cross_section_um2", positive, positive, np.array([-1.0], dtype=np.float32)),
        )

        for name, delp, number, cross_section in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    layer_optical_depth(delp, number, cross_section)

    def _lookup_table(self):
        n_real = np.array([1.3, 1.5, 1.7], dtype=np.float32)
        n_imag = np.array([0.0, 0.1], dtype=np.float32)
        radius = np.array([0.05, 0.10, 0.20], dtype=np.float32)
        shape = (len(n_real), len(n_imag), len(radius))
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        coords = {"n_real": n_real, "n_imag": n_imag, "radius": radius}
        return xr.Dataset(
            {
                "ext": (("n_real", "n_imag", "radius"), values),
                "abs": (("n_real", "n_imag", "radius"), values + 100.0),
                "asm": (("n_real", "n_imag", "radius"), values + 200.0),
            },
            coords=coords,
        )

    def test_lookup_mode_optics_uses_lower_bins_and_returns_float32(self):
        ds_table = self._lookup_table()
        n_re = np.array([[1.30, 1.59], [1.70, 1.49]], dtype=np.float32)
        n_im = np.array([[-0.05, 0.10], [0.0, -0.09]], dtype=np.float32)
        radius = np.array([[0.05, 0.15], [0.20, 0.10]], dtype=np.float32)

        ext, abs_, asm = lookup_mode_optics(n_re, n_im, radius, ds_table)

        expected_ext = np.array([[0.0, 10.0], [14.0, 1.0]], dtype=np.float32)
        self.assertEqual(ext.shape, n_re.shape)
        self.assertEqual(abs_.shape, n_re.shape)
        self.assertEqual(asm.shape, n_re.shape)
        self.assertEqual(ext.dtype, np.float32)
        self.assertEqual(abs_.dtype, np.float32)
        self.assertEqual(asm.dtype, np.float32)
        np.testing.assert_array_equal(ext, expected_ext)
        np.testing.assert_array_equal(abs_, expected_ext + 100.0)
        np.testing.assert_array_equal(asm, expected_ext + 200.0)

    def test_lookup_mode_optics_clips_to_table_and_uses_abs_imaginary_index(self):
        ds_table = self._lookup_table()
        n_re = np.array([1.0, 2.0], dtype=np.float32)
        n_im = np.array([-0.2, -0.03], dtype=np.float32)
        radius = np.array([0.001, 9.0], dtype=np.float32)

        ext, abs_, asm = lookup_mode_optics(n_re, n_im, radius, ds_table)

        expected_ext = np.array([3.0, 14.0], dtype=np.float32)
        np.testing.assert_array_equal(ext, expected_ext)
        np.testing.assert_array_equal(abs_, expected_ext + 100.0)
        np.testing.assert_array_equal(asm, expected_ext + 200.0)

    def test_lookup_mode_optics_missing_variable_raises_key_error(self):
        ds_table = self._lookup_table().drop_vars("asm")

        with self.assertRaisesRegex(KeyError, "asm"):
            lookup_mode_optics(
                np.array([1.3], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.05], dtype=np.float32),
                ds_table,
            )


if __name__ == "__main__":
    unittest.main()
