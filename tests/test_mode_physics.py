import unittest

import numpy as np

from mode_physics import (
    derive_number_mixing_ratio,
    kohler_wet_radius_um,
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


if __name__ == "__main__":
    unittest.main()
