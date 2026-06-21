import unittest
import numpy as np

from mode_config import (
    allocate_size_bins_to_modes,
    default_mam4_allocations,
    load_config,
    map_mam4_to_mam3,
    normalize_allocations,
    resolved_allocations,
)


class TestModeConfig(unittest.TestCase):
    def test_default_mam4_allocations(self):
        allocations = default_mam4_allocations()
        self.assertEqual(allocations["SO4"], {"a1": 0.90, "a2": 0.10, "a3": 0.0, "a4": 0.0})
        self.assertEqual(allocations["OCPHOBIC"], {"a4": 1.0})
        self.assertEqual(allocations["BCPHOBIC"], {"a4": 1.0})
        self.assertEqual(allocations["NO3"], {"a1": 0.70, "a3": 0.30})

    def test_mam3_maps_primary_carbon_to_accumulation(self):
        mapped = map_mam4_to_mam3({"BCPHOBIC": {"a4": 1.0}, "SO4": {"a1": 0.9, "a2": 0.1}})
        self.assertEqual(mapped["BCPHOBIC"], {"a1": 1.0})
        self.assertEqual(mapped["SO4"], {"a1": 0.9, "a2": 0.1})

    def test_normalize_allocations(self):
        normalized = normalize_allocations({"a1": 9.0, "a2": 1.0})
        self.assertAlmostEqual(normalized["a1"], 0.9)
        self.assertAlmostEqual(normalized["a2"], 0.1)

    def test_normalize_allocations_rejects_negative_weights(self):
        with self.assertRaises(ValueError):
            normalize_allocations({"a1": 2.0, "a2": -1.0})

    def test_size_bin_allocation_prefers_nearest_mode(self):
        bins = np.array([0.15, 0.60])
        modes = {
            "a1": {"dry_radius_um": 0.15, "sigma_g": 1.6},
            "a3": {"dry_radius_um": 0.60, "sigma_g": 1.8},
        }
        allocation = allocate_size_bins_to_modes(bins, modes)
        self.assertGreater(allocation[0]["a1"], 0.75)
        self.assertGreater(allocation[1]["a3"], 0.75)
        self.assertAlmostEqual(sum(allocation[0].values()), 1.0)
        self.assertAlmostEqual(sum(allocation[1].values()), 1.0)

    def test_resolved_allocations_adds_size_bins(self):
        config = {
            "Schemes": {
                "MAM4": {
                    "modes": {
                        "a1": {"dry_radius_um": 0.15, "sigma_g": 1.6},
                        "a3": {"dry_radius_um": 0.60, "sigma_g": 1.8},
                    },
                    "allocations": {"SO4": {"a1": 1.0}},
                    "size_bins": {
                        "DU": {"species": ["DU001", "DU002"], "radii_um": [0.15, 0.60]}
                    },
                }
            }
        }
        allocations = resolved_allocations(config, "MAM4")
        self.assertIn("SO4", allocations)
        self.assertIn("DU001", allocations)
        self.assertIn("DU002", allocations)
        self.assertGreater(allocations["DU001"]["a1"], 0.75)
        self.assertGreater(allocations["DU002"]["a3"], 0.75)

    def test_resolved_allocations_rejects_size_bin_length_mismatch(self):
        config = {
            "Schemes": {
                "MAM4": {
                    "modes": {
                        "a1": {"dry_radius_um": 0.15, "sigma_g": 1.6},
                        "a3": {"dry_radius_um": 0.60, "sigma_g": 1.8},
                    },
                    "size_bins": {
                        "DU": {"species": ["DU001"], "radii_um": [0.15, 0.60]}
                    },
                }
            }
        }
        with self.assertRaises(ValueError):
            resolved_allocations(config, "MAM4")


class TestYamlSchema(unittest.TestCase):
    REQUIRED_FIELDS = {"rh", "temperature", "delp", "ps"}
    REQUIRED_MODES = {"a1", "a2", "a3", "a4"}
    REQUIRED_SIZE_BINS = {"NO3AN", "SS", "DU"}

    def assert_native_grid_schema(self, config):
        self.assertIn("Sources", config)
        self.assertIn("Schemes", config)
        self.assertIn("GEOSIT", config["Sources"])
        self.assertIn("MERRA2", config["Sources"])

        for source_name in ("GEOSIT", "MERRA2"):
            source = config["Sources"][source_name]
            self.assertIn("fields", source)
            fields = source["fields"]
            self.assertLessEqual(self.REQUIRED_FIELDS, set(fields))

        self.assertIn("MAM4", config["Schemes"])
        scheme = config["Schemes"]["MAM4"]
        self.assertIn("allocations", scheme)
        self.assertIn("modes", scheme)
        self.assertIn("size_bins", scheme)
        self.assertLessEqual(self.REQUIRED_MODES, set(scheme["modes"]))
        self.assertLessEqual(self.REQUIRED_SIZE_BINS, set(scheme["size_bins"]))

        for bin_name in self.REQUIRED_SIZE_BINS:
            size_bin = scheme["size_bins"][bin_name]
            self.assertIn("species", size_bin)
            self.assertIn("radii_um", size_bin)
            self.assertEqual(len(size_bin["species"]), len(size_bin["radii_um"]))

    def config_strings(self, value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from self.config_strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from self.config_strings(item)

    def test_aerosol_yaml_has_sources_and_schemes(self):
        config = load_config("aerosol.yaml")
        self.assert_native_grid_schema(config)

    def test_aerosol_ceres_yaml_has_sources_and_schemes(self):
        config = load_config("aerosol_ceres.yaml")
        self.assert_native_grid_schema(config)

    def test_aerosol_ceres_yaml_has_required_paths(self):
        config = load_config("aerosol_ceres.yaml")
        sources = config["Sources"]
        modes = config["Schemes"]["MAM4"]["modes"]

        self.assertIn("/CERES_prd/GMAO/GEOSIT/", sources["GEOSIT"]["input_pattern"])
        self.assertIn("/CERES/sarb/dfillmor/GEOSIT-MAM/", sources["GEOSIT"]["output_pattern"])
        self.assertIn("/CERES/sarb/dfillmor/GEOSIT_alpha_4/", sources["GEOSIT"]["external_vis_pattern"])
        self.assertEqual(
            sources["MERRA2"]["input_pattern"],
            "/CERES_prd/GMAO/MERRA2/YYYY/MM/MERRA2_300.inst3_3d_aer_Nv.YYYYMMDD.nc4",
        )
        self.assertIn("/CERES/sarb/dfillmor", sources["MERRA2"]["output_pattern"])
        self.assertIn("/CERES/sarb/dfillmor", sources["MERRA2"]["external_vis_pattern"])
        for mode in modes.values():
            self.assertIn("/CERES/sarb/dfillmor", mode["filename_sarb"])

        config_strings = list(self.config_strings(config))
        self.assertTrue(any("/CERES/sarb/dfillmor" in value for value in config_strings))
        self.assertFalse(any("dfillmore" in value for value in config_strings))


if __name__ == "__main__":
    unittest.main()
