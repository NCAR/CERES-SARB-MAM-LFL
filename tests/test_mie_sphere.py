import unittest

from mie_sphere import mie_cross_sections_um2, mie_efficiencies


class TestMieSphere(unittest.TestCase):
    def test_nonabsorbing_sphere_has_no_absorption(self):
        result = mie_cross_sections_um2(
            n_real=1.5,
            n_imag=0.0,
            radius_um=0.1,
            wavelength_um=0.55,
        )

        self.assertGreater(result["ext"], 0.0)
        self.assertGreater(result["sca"], 0.0)
        self.assertAlmostEqual(result["ext"], result["sca"], places=12)
        self.assertAlmostEqual(result["abs"], 0.0, places=12)

    def test_small_particle_matches_rayleigh_scattering_limit(self):
        n_real = 1.5
        radius_um = 0.005
        wavelength_um = 0.55

        result = mie_efficiencies(
            n_real=n_real,
            n_imag=0.0,
            radius_um=radius_um,
            wavelength_um=wavelength_um,
        )
        x = result["size_parameter"]
        m2 = n_real ** 2
        rayleigh_q_sca = (8.0 / 3.0) * x ** 4 * abs((m2 - 1.0) / (m2 + 2.0)) ** 2

        self.assertAlmostEqual(result["q_sca"], rayleigh_q_sca, delta=rayleigh_q_sca * 0.03)

    def test_absorbing_sphere_has_positive_absorption(self):
        result = mie_cross_sections_um2(
            n_real=1.5,
            n_imag=0.02,
            radius_um=0.1,
            wavelength_um=0.55,
        )

        self.assertGreater(result["ext"], result["sca"])
        self.assertGreater(result["abs"], 0.0)

    def test_rejects_nonpositive_size_inputs(self):
        with self.assertRaises(ValueError):
            mie_efficiencies(1.5, 0.0, 0.0, 0.55)
        with self.assertRaises(ValueError):
            mie_efficiencies(1.5, 0.0, 0.1, 0.0)


if __name__ == "__main__":
    unittest.main()
