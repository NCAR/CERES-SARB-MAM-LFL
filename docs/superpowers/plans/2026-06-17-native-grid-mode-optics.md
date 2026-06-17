# Native-Grid Mode Optics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native-grid GEOSIT/MERRA2 mode-internal aerosol optics path with YAML-defined MAM mixing state, Kohler water uptake, VIS column preservation, and mode-summed AER output.

**Architecture:** Add small helper modules around the existing `microphysics.py` kernels. `mode_optics.py` reads source-native fields, allocates species to one mode, computes uncorrected mode optical depths, applies a VIS-derived column correction, and writes SARB-style mode files. `mode_external_mix.py` sums corrected mode files into scheme-level totals.

**Tech Stack:** Python, NumPy, xarray, PyYAML, SciPy lookup tables, standard-library `unittest`.

---

## File Structure

Create:

- `mode_config.py` — source/scheme config loading, default allocation, size-bin allocation.
- `source_fields.py` — source-native GEOSIT/MERRA2 field readers and pressure-thickness handling.
- `mode_physics.py` — mode dry volume, derived number, mixed state, lookup orchestration.
- `vis_correction.py` — total VIS column factor and layer scaling.
- `mode_optics.py` — CLI for per-mode per-band outputs.
- `mode_external_mix.py` — CLI and helper for externally mixing corrected modes into total AER.
- `tests/test_mode_config.py`
- `tests/test_source_fields.py`
- `tests/test_mode_physics.py`
- `tests/test_vis_correction.py`
- `tests/test_mode_external_mix.py`

Modify:

- `aerosol.yaml` — add `Sources`, `Schemes`, and allocation defaults.
- `aerosol_ceres.yaml` — same structure with CERES paths.
- `README.md` — add native-grid mode-optics usage.

---

### Task 1: YAML Allocation Helpers

**Files:**
- Create: `tests/test_mode_config.py`
- Create: `mode_config.py`

- [ ] **Step 1: Write failing allocation tests**

Create `tests/test_mode_config.py`:

```python
import unittest
import numpy as np

from mode_config import (
    allocate_size_bins_to_modes,
    default_mam4_allocations,
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m unittest tests.test_mode_config -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mode_config'`.

- [ ] **Step 3: Implement allocation helpers**

Create `mode_config.py`:

```python
import math
import yaml
import numpy as np


def load_config(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def normalize_allocations(weights):
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("allocation weights must have positive sum")
    return {mode: float(value) / total for mode, value in weights.items() if float(value) > 0.0}


def default_mam4_allocations():
    return {
        "SO4": {"a1": 0.90, "a2": 0.10, "a3": 0.0, "a4": 0.0},
        "OCPHILIC": {"a1": 1.0},
        "BCPHILIC": {"a1": 1.0},
        "OCPHOBIC": {"a4": 1.0},
        "BCPHOBIC": {"a4": 1.0},
        "NO3": {"a1": 0.70, "a3": 0.30},
        "POM": {"a1": 0.80, "a2": 0.20},
        "SOA": {"a1": 0.80, "a2": 0.20},
    }


def map_mam4_to_mam3(allocations):
    mapped = {}
    for species, weights in allocations.items():
        next_weights = {}
        for mode, value in weights.items():
            target = "a1" if mode == "a4" else mode
            next_weights[target] = next_weights.get(target, 0.0) + float(value)
        mapped[species] = normalize_allocations(next_weights)
    return mapped


def _lognormal_pdf(radius_um, median_um, sigma_g):
    radius = np.asarray(radius_um, dtype=np.float64)
    radius = np.clip(radius, 1.0e-12, None)
    log_sigma = math.log(float(sigma_g))
    prefactor = 1.0 / (radius * log_sigma * math.sqrt(2.0 * math.pi))
    exponent = -((np.log(radius) - math.log(float(median_um))) ** 2) / (2.0 * log_sigma ** 2)
    return prefactor * np.exp(exponent)


def allocate_size_bins_to_modes(bin_radii_um, mode_specs):
    allocation = []
    for radius in np.asarray(bin_radii_um, dtype=np.float64):
        weights = {}
        for mode, spec in mode_specs.items():
            weights[mode] = float(_lognormal_pdf(radius, spec["dry_radius_um"], spec["sigma_g"]))
        allocation.append(normalize_allocations(weights))
    return allocation


def resolved_allocations(config, scheme):
    scheme_info = config["Schemes"][scheme]
    allocations = {
        species: normalize_allocations(weights)
        for species, weights in scheme_info.get("allocations", {}).items()
    }
    mode_specs = scheme_info["modes"]
    for group in scheme_info.get("size_bins", {}).values():
        generated = allocate_size_bins_to_modes(group["radii_um"], mode_specs)
        for species, weights in zip(group["species"], generated):
            allocations[species] = weights
    return allocations
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m unittest tests.test_mode_config -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add mode_config.py tests/test_mode_config.py
git commit -m "Add mode allocation helpers"
```

---

### Task 2: YAML Schema Entries

**Files:**
- Modify: `aerosol.yaml`
- Modify: `aerosol_ceres.yaml`
- Test: `tests/test_mode_config.py`

- [ ] **Step 1: Add schema test**

Append to `tests/test_mode_config.py`:

```python
class TestYamlSchema(unittest.TestCase):
    def test_aerosol_yaml_has_sources_and_schemes(self):
        config = load_config("aerosol.yaml")
        self.assertIn("Sources", config)
        self.assertIn("Schemes", config)
        self.assertIn("GEOSIT", config["Sources"])
        self.assertIn("MERRA2", config["Sources"])
        self.assertIn("MAM4", config["Schemes"])
        self.assertIn("allocations", config["Schemes"]["MAM4"])
```

Also add `load_config` to the import list at the top of that file.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m unittest tests.test_mode_config -v
```

Expected: FAIL with `AssertionError: 'Sources' not found`.

- [ ] **Step 3: Add local YAML entries**

Append this block to `aerosol.yaml`, preserving existing `MAM4` and `Types` blocks:

```yaml

Sources:
    GEOSIT:
        input_pattern: GEOSIT/YYYY/MM/GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294.YYYY-MM-DDTHH00.V01.nc4
        output_pattern: GEOSIT/YYYY/MM/GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294.{label}_{band}.YYYY-MM-DDTHH00.V01.nc4
        external_vis_pattern: GEOSIT/YYYY/MM/GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294.AER_550NM.YYYY-MM-DDTHH00.V01.nc4
        dims:
            time: time
            lev: lev
            lat: lat
            lon: lon
        fields:
            rh: RH
            temperature: T
            delp: DELP
            ps: PS
        species:
            SO4: SO4
            OCPHOBIC: OCPHOBIC
            OCPHILIC: OCPHILIC
            BCPHOBIC: BCPHOBIC
            BCPHILIC: BCPHILIC
            NO3AN1: NO3AN1
            NO3AN2: NO3AN2
            NO3AN3: NO3AN3
            SS001: SS001
            SS002: SS002
            SS003: SS003
            SS004: SS004
            SS005: SS005
            DU001: DU001
            DU002: DU002
            DU003: DU003
            DU004: DU004
            DU005: DU005
    MERRA2:
        input_pattern: MERRA2/YYYY/MM/MERRA2_300.inst3_3d_aer_Nv.YYYYMMDD.nc4
        output_pattern: MERRA2/YYYY/MM/MERRA2_300.inst3_3d_aer_Nv.{label}_{band}.YYYYMMDDTHH.nc4
        external_vis_pattern: MERRA2/YYYY/MM/MERRA2_300.inst3_3d_aer_Nv.AER_550NM.YYYYMMDDTHH.nc4
        dims:
            time: time
            lev: lev
            lat: lat
            lon: lon
        fields:
            rh: RH
            temperature: T
            delp: DELP
            ps: PS
        species:
            SO4: SO4
            OCPHOBIC: OCPHOBIC
            OCPHILIC: OCPHILIC
            BCPHOBIC: BCPHOBIC
            BCPHILIC: BCPHILIC
            SS001: SS001
            SS002: SS002
            SS003: SS003
            SS004: SS004
            SS005: SS005
            DU001: DU001
            DU002: DU002
            DU003: DU003
            DU004: DU004
            DU005: DU005

Schemes:
    MAM4:
        modes:
            a1:
                name: Accumulation
                dry_radius_um: 0.055
                sigma_g: 1.8
                filename_sarb: ${HOME}/Data/Optics/SARB/mam4_mode1_larc_c000002.v2.nc
            a2:
                name: Aitken
                dry_radius_um: 0.012
                sigma_g: 1.6
                filename_sarb: ${HOME}/Data/Optics/SARB/mam4_mode2_larc_c000002.v2.nc
            a3:
                name: Coarse
                dry_radius_um: 0.40
                sigma_g: 1.8
                filename_sarb: ${HOME}/Data/Optics/SARB/mam4_mode3_larc_c000002.v2.nc
            a4:
                name: Primary Carbon
                dry_radius_um: 0.050
                sigma_g: 1.6
                filename_sarb: ${HOME}/Data/Optics/SARB/mam4_mode4_larc_c000002.v2.nc
        allocations:
            SO4: {a1: 0.90, a2: 0.10, a3: 0.00, a4: 0.00}
            OCPHILIC: {a1: 1.00}
            BCPHILIC: {a1: 1.00}
            OCPHOBIC: {a4: 1.00}
            BCPHOBIC: {a4: 1.00}
            NO3: {a1: 0.70, a3: 0.30}
            POM: {a1: 0.80, a2: 0.20}
            SOA: {a1: 0.80, a2: 0.20}
        size_bins:
            NO3AN:
                species: [NO3AN1, NO3AN2, NO3AN3]
                radii_um: [0.15, 0.50, 1.50]
            SS:
                species: [SS001, SS002, SS003, SS004, SS005]
                radii_um: [0.08, 0.30, 1.00, 3.00, 8.00]
            DU:
                species: [DU001, DU002, DU003, DU004, DU005]
                radii_um: [0.73, 1.40, 2.40, 4.50, 8.00]
```

- [ ] **Step 4: Add CERES YAML entries**

Append the same block to `aerosol_ceres.yaml`, replacing `${HOME}/Data/Optics` with `/CERES/sarb/dfillmor/Optics`, `GEOSIT/` with `GEOSIT_alpha_4/` in output and external VIS patterns, the MERRA2 input pattern with `/CERES_prd/GMAO/MERRA2/YYYY/MM/MERRA2_300.inst3_3d_aer_Nv.YYYYMMDD.nc4`, and the MERRA2 output pattern with `/CERES/sarb/dfillmor/MERRA2/YYYY/MM/MERRA2_300.inst3_3d_aer_Nv.{label}_{band}.YYYYMMDDTHH.nc4`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests.test_mode_config -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aerosol.yaml aerosol_ceres.yaml tests/test_mode_config.py
git commit -m "Add native-grid source and scheme config"
```

---

### Task 3: Source Field Adapter

**Files:**
- Create: `tests/test_source_fields.py`
- Create: `source_fields.py`

- [ ] **Step 1: Write failing source-field tests**

Create `tests/test_source_fields.py`:

```python
import unittest
import numpy as np
import xarray as xr

from source_fields import read_source_fields_from_dataset


class TestSourceFields(unittest.TestCase):
    def test_reads_native_fields_and_species(self):
        ds = xr.Dataset(
            {
                "RH": (("time", "lev", "lat", "lon"), np.full((1, 2, 2, 3), 0.5, dtype=np.float32)),
                "T": (("time", "lev", "lat", "lon"), np.full((1, 2, 2, 3), 280.0, dtype=np.float32)),
                "DELP": (("time", "lev", "lat", "lon"), np.full((1, 2, 2, 3), 100.0, dtype=np.float32)),
                "SO4": (("time", "lev", "lat", "lon"), np.full((1, 2, 2, 3), 1.0e-9, dtype=np.float32)),
                "BCPHILIC": (("time", "lev", "lat", "lon"), np.full((1, 2, 2, 3), 2.0e-10, dtype=np.float32)),
            },
            coords={"time": [0], "lev": [1, 2], "lat": [-1.0, 1.0], "lon": [0.0, 1.25, 2.5]},
        )
        spec = {"fields": {"rh": "RH", "temperature": "T", "delp": "DELP"}, "species": {"SO4": "SO4", "BCPHILIC": "BCPHILIC"}}
        fields = read_source_fields_from_dataset(ds, spec, ["SO4", "BCPHILIC"])
        self.assertEqual(fields.rh.shape, (1, 2, 2, 3))
        self.assertAlmostEqual(float(fields.temperature.mean()), 280.0)
        self.assertEqual(fields.delp.shape, (1, 2, 2, 3))
        self.assertIn("SO4", fields.species)
        self.assertIn("BCPHILIC", fields.species)

    def test_temperature_falls_back_to_freezing(self):
        ds = xr.Dataset(
            {
                "RH": (("time", "lev", "lat", "lon"), np.ones((1, 1, 1, 1), dtype=np.float32)),
                "DELP": (("time", "lev", "lat", "lon"), np.ones((1, 1, 1, 1), dtype=np.float32)),
                "SO4": (("time", "lev", "lat", "lon"), np.ones((1, 1, 1, 1), dtype=np.float32)),
            }
        )
        spec = {"fields": {"rh": "RH", "temperature": "T", "delp": "DELP"}, "species": {"SO4": "SO4"}}
        fields = read_source_fields_from_dataset(ds, spec, ["SO4"])
        self.assertAlmostEqual(float(fields.temperature[0, 0, 0, 0]), 273.15)

    def test_missing_required_variable_raises(self):
        ds = xr.Dataset({"RH": (("time", "lev", "lat", "lon"), np.ones((1, 1, 1, 1), dtype=np.float32))})
        spec = {"fields": {"rh": "RH", "temperature": "T", "delp": "DELP"}, "species": {"SO4": "SO4"}}
        with self.assertRaisesRegex(KeyError, "DELP"):
            read_source_fields_from_dataset(ds, spec, ["SO4"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m unittest tests.test_source_fields -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'source_fields'`.

- [ ] **Step 3: Implement source field adapter**

Create `source_fields.py`:

```python
from dataclasses import dataclass
import numpy as np
import xarray as xr


@dataclass
class SourceFields:
    dataset: xr.Dataset
    rh: np.ndarray
    temperature: np.ndarray
    delp: np.ndarray
    species: dict
    coords: dict
    dims: tuple


def _require_var(ds, name):
    if name not in ds:
        raise KeyError(name)
    return ds[name]


def read_source_fields_from_dataset(ds, source_spec, species_names):
    field_spec = source_spec["fields"]
    rh = _require_var(ds, field_spec["rh"]).values.astype(np.float32)
    temperature_name = field_spec.get("temperature")
    if temperature_name is not None and temperature_name in ds:
        temperature = ds[temperature_name].values.astype(np.float32)
    else:
        temperature = np.full_like(rh, 273.15, dtype=np.float32)
    delp = _require_var(ds, field_spec["delp"]).values.astype(np.float32)
    species = {}
    for species_name in species_names:
        source_name = source_spec["species"].get(species_name)
        if source_name is None:
            species[species_name] = np.zeros_like(rh, dtype=np.float32)
        else:
            species[species_name] = _require_var(ds, source_name).values.astype(np.float32)
    coords = {name: ds.coords[name] for name in ds.coords}
    return SourceFields(
        dataset=ds,
        rh=rh,
        temperature=temperature,
        delp=delp,
        species=species,
        coords=coords,
        dims=_require_var(ds, field_spec["rh"]).dims,
    )


def open_source_fields(path, source_spec, species_names):
    ds = xr.open_dataset(path)
    return read_source_fields_from_dataset(ds, source_spec, species_names)
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_source_fields -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add source_fields.py tests/test_source_fields.py
git commit -m "Add source-native field adapter"
```

---

### Task 4: Mode Physics Helpers

**Files:**
- Create: `tests/test_mode_physics.py`
- Create: `mode_physics.py`

- [ ] **Step 1: Write failing physics tests**

Create `tests/test_mode_physics.py`:

```python
import unittest
import numpy as np

from mode_physics import derive_number_mixing_ratio, kohler_wet_radius_um, mix_mode_state


class TestModePhysics(unittest.TestCase):
    def test_derive_number_positive_for_positive_volume(self):
        dry_volume = np.full((1, 2, 2, 2), 1.0e-12, dtype=np.float32)
        number = derive_number_mixing_ratio(dry_volume, dry_radius_um=0.05, sigma_g=1.6)
        self.assertEqual(number.shape, dry_volume.shape)
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
        refractive = {"SO4": (1.4, 0.0), "BC": (1.8, 0.4), "WAT": (1.33, 0.0)}
        temperature = np.full((1, 1, 1, 1), 280.0, dtype=np.float32)
        state = mix_mode_state(species_info, q, refractive, np.full((1, 1, 1, 1), 0.5), temperature, 0.05)
        self.assertAlmostEqual(float(state["n_re"][0, 0, 0, 0]), 1.6, places=5)
        self.assertGreaterEqual(float(state["r_w_um"][0, 0, 0, 0]), 0.05)

    def test_kohler_wet_radius_solves_equation(self):
        rh = np.array([0.80], dtype=np.float32)
        temperature = np.array([280.0], dtype=np.float32)
        dry_radius_um = 0.05
        b = np.array([1.0], dtype=np.float32)
        wet = kohler_wet_radius_um(dry_radius_um, b, rh, temperature)
        self.assertGreater(float(wet[0]), dry_radius_um)
        dry_m = dry_radius_um * 1.0e-6
        wet_m = wet[0] * 1.0e-6
        a = 2.0 * 18.016 * 0.076 / (8.3143e3 * 1000.0 * temperature[0])
        residual = np.log(rh[0]) - (a / wet_m - b[0] * dry_m ** 3 / (wet_m ** 3 - dry_m ** 3))
        self.assertAlmostEqual(float(residual), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m unittest tests.test_mode_physics -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mode_physics'`.

- [ ] **Step 3: Implement mode physics helpers**

Create `mode_physics.py`:

```python
import math
import numpy as np


def lognormal_volume_factor(dry_radius_um, sigma_g):
    radius_m = float(dry_radius_um) * 1.0e-6
    sigma = float(sigma_g)
    return (4.0 / 3.0) * math.pi * radius_m ** 3 * math.exp(4.5 * math.log(sigma) ** 2)


def derive_number_mixing_ratio(dry_volume_m3_per_kg, dry_radius_um, sigma_g):
    particle_volume = lognormal_volume_factor(dry_radius_um, sigma_g)
    return np.divide(
        dry_volume_m3_per_kg,
        particle_volume,
        out=np.zeros_like(dry_volume_m3_per_kg, dtype=np.float32),
        where=dry_volume_m3_per_kg > 0.0,
    ).astype(np.float32)


def _dry_volume(species_info, q):
    shape = next(iter(q.values())).shape
    total = np.zeros(shape, dtype=np.float32)
    for species, values in q.items():
        rho_g_cm3 = float(species_info[species]["density"])
        rho_kg_m3 = rho_g_cm3 * 1000.0
        total += values.astype(np.float32) / rho_kg_m3
    return total


def _mixed_hygroscopicity(species_info, q, rh):
    numerator = np.zeros_like(rh, dtype=np.float32)
    denominator = np.zeros_like(rh, dtype=np.float32)
    for species, values in q.items():
        rho = float(species_info[species]["density"])
        b0, b1, b2 = species_info[species]["hygroscopicity"]
        coeff = float(b0) + float(b1) * rh + float(b2) * rh ** 2
        volume = values / rho
        numerator += volume * coeff
        denominator += volume
    return np.divide(numerator, denominator, out=np.zeros_like(rh, dtype=np.float32), where=denominator > 0.0)


def kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh, temperature):
    dry_radius_m = float(dry_radius_um) * 1.0e-6
    rh_clip = np.clip(rh.astype(np.float64), 1.0e-6, 0.99)
    b = np.maximum(hygroscopicity.astype(np.float64), 0.0)
    temp = np.clip(temperature.astype(np.float64), 180.0, 330.0)
    wet = np.full(rh.shape, dry_radius_m, dtype=np.float64)
    active = (rh_clip > 1.0e-6) & (b > 0.0)
    if not np.any(active):
        return (wet * 1.0e6).astype(np.float32)
    m_w = 18.016
    sigma = 0.076
    rho_w = 1000.0
    gas_r = 8.3143e3
    a = 2.0 * m_w * sigma / (gas_r * rho_w * temp)
    low = np.full(rh.shape, dry_radius_m * (1.0 + 1.0e-8), dtype=np.float64)
    high = dry_radius_m * np.maximum(2.0, (1.0 - b / np.log(rh_clip)) ** (1.0 / 3.0) * 2.0)
    log_rh = np.log(rh_clip)
    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = a / mid - b * dry_radius_m ** 3 / (mid ** 3 - dry_radius_m ** 3) - log_rh
        high = np.where((f_mid > 0.0) & active, mid, high)
        low = np.where((f_mid <= 0.0) & active, mid, low)
    wet = np.where(active, 0.5 * (low + high), dry_radius_m)
    return (wet * 1.0e6).astype(np.float32)


def mix_mode_state(species_info, q, refractive, rh, temperature, dry_radius_um):
    shape = rh.shape
    dry_volume = _dry_volume(species_info, q)
    hygroscopicity = _mixed_hygroscopicity(species_info, q, rh.astype(np.float32))
    r_w_um = kohler_wet_radius_um(dry_radius_um, hygroscopicity, rh.astype(np.float32), temperature.astype(np.float32))
    n_re_num = np.zeros(shape, dtype=np.float32)
    n_im_num = np.zeros(shape, dtype=np.float32)
    dry_volume_cm = np.zeros(shape, dtype=np.float32)
    for species, values in q.items():
        rho = float(species_info[species]["density"])
        volume = values / rho
        dry_volume_cm += volume
        n_re_num += float(refractive[species][0]) * volume
        n_im_num += float(refractive[species][1]) * volume
    n_re = np.divide(n_re_num, dry_volume_cm, out=np.ones(shape, dtype=np.float32), where=dry_volume_cm > 0.0)
    n_im = np.divide(n_im_num, dry_volume_cm, out=np.zeros(shape, dtype=np.float32), where=dry_volume_cm > 0.0)
    return {
        "dry_volume": dry_volume,
        "hygroscopicity": hygroscopicity,
        "r_w_um": r_w_um,
        "n_re": n_re.astype(np.float32),
        "n_im": n_im.astype(np.float32),
    }
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_mode_physics -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mode_physics.py tests/test_mode_physics.py
git commit -m "Add mode physics helpers"
```

---

### Task 5: VIS Correction

**Files:**
- Create: `tests/test_vis_correction.py`
- Create: `vis_correction.py`

- [ ] **Step 1: Write failing correction tests**

Create `tests/test_vis_correction.py`:

```python
import unittest
import numpy as np
import xarray as xr

from vis_correction import apply_column_factor, compute_vis_factor


class TestVisCorrection(unittest.TestCase):
    def test_compute_factor_caps_and_skips_tiny_columns(self):
        external = xr.DataArray(np.array([[2.0, 1.0, 0.0]], dtype=np.float32), dims=["lat", "lon"])
        internal = xr.DataArray(np.array([[1.0, 0.1, 0.0]], dtype=np.float32), dims=["lat", "lon"])
        factor, stats = compute_vis_factor(external, internal, min_aod=1.0e-8, min_factor=0.25, max_factor=4.0)
        self.assertAlmostEqual(float(factor.values[0, 0]), 2.0)
        self.assertAlmostEqual(float(factor.values[0, 1]), 4.0)
        self.assertAlmostEqual(float(factor.values[0, 2]), 1.0)
        self.assertEqual(stats["capped"], 1)
        self.assertEqual(stats["skipped"], 1)

    def test_apply_factor_preserves_column_scaling(self):
        layer = xr.DataArray(np.ones((2, 1, 2), dtype=np.float32), dims=["lev", "lat", "lon"])
        factor = xr.DataArray(np.array([[2.0, 3.0]], dtype=np.float32), dims=["lat", "lon"])
        scaled = apply_column_factor(layer, factor)
        self.assertAlmostEqual(float(scaled.sum("lev").values[0, 0]), 4.0)
        self.assertAlmostEqual(float(scaled.sum("lev").values[0, 1]), 6.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m unittest tests.test_vis_correction -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vis_correction'`.

- [ ] **Step 3: Implement correction helpers**

Create `vis_correction.py`:

```python
import numpy as np
import xarray as xr


def compute_vis_factor(external_column, internal_column, min_aod=1.0e-8, min_factor=0.25, max_factor=4.0):
    raw = xr.where(internal_column > min_aod, external_column / internal_column, 1.0)
    tiny = (external_column <= min_aod) & (internal_column <= min_aod)
    clipped = raw.clip(min=min_factor, max=max_factor)
    factor = xr.where(tiny, 1.0, clipped).astype(np.float32)
    capped = ((raw < min_factor) | (raw > max_factor)) & (~tiny)
    stats = {
        "capped": int(capped.sum().item()),
        "skipped": int(tiny.sum().item()),
    }
    return factor, stats


def apply_column_factor(layer_field, factor):
    if "lev" not in layer_field.dims:
        raise ValueError("layer_field must include lev dimension")
    return (layer_field * factor).astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_vis_correction -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vis_correction.py tests/test_vis_correction.py
git commit -m "Add VIS column correction helpers"
```

---

### Task 6: Mode External Mix

**Files:**
- Create: `tests/test_mode_external_mix.py`
- Create: `mode_external_mix.py`

- [ ] **Step 1: Write failing external-mix tests**

Create `tests/test_mode_external_mix.py`:

```python
import unittest
import numpy as np
import xarray as xr

from mode_external_mix import mix_mode_datasets


def _dataset(ext, sca, asm):
    ext_da = xr.DataArray(np.array(ext, dtype=np.float32), dims=["lev", "lat", "lon"])
    sca_da = xr.DataArray(np.array(sca, dtype=np.float32), dims=["lev", "lat", "lon"])
    asm_da = xr.DataArray(np.array(asm, dtype=np.float32), dims=["lev", "lat", "lon"])
    return xr.Dataset(
        {
            "DELP": xr.DataArray(np.ones_like(ext_da.values), dims=["lev", "lat", "lon"]),
            "Extinction_Layer_Optical_Depth": ext_da,
            "Scattering_Layer_Optical_Depth": sca_da,
            "Layer_Asymmetry_Parameter": asm_da,
            "Extinction_Column_Optical_Depth": ext_da.sum("lev"),
        }
    )


class TestModeExternalMix(unittest.TestCase):
    def test_sums_extinction_and_scattering_and_weights_asm(self):
        ds1 = _dataset([[[1.0]]], [[[0.5]]], [[[0.2]]])
        ds2 = _dataset([[[2.0]]], [[[1.5]]], [[[0.6]]])
        mixed = mix_mode_datasets([ds1, ds2])
        self.assertAlmostEqual(float(mixed["Extinction_Layer_Optical_Depth"].values[0, 0, 0]), 3.0)
        self.assertAlmostEqual(float(mixed["Scattering_Layer_Optical_Depth"].values[0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(mixed["Layer_Asymmetry_Parameter"].values[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(mixed["Extinction_Column_Optical_Depth"].values[0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m unittest tests.test_mode_external_mix -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mode_external_mix'`.

- [ ] **Step 3: Implement mode external mix**

Create `mode_external_mix.py`:

```python
import argparse
import sys
import numpy as np
import xarray as xr

TAU_THRESH = 1.0e-5


def mix_mode_datasets(datasets):
    if not datasets:
        raise ValueError("at least one mode dataset is required")
    out = datasets[0].copy(deep=True)
    ext = xr.zeros_like(out["Extinction_Layer_Optical_Depth"])
    sca = xr.zeros_like(out["Scattering_Layer_Optical_Depth"])
    asm_num = xr.zeros_like(out["Layer_Asymmetry_Parameter"])
    for ds in datasets:
        ext += ds["Extinction_Layer_Optical_Depth"]
        sca += ds["Scattering_Layer_Optical_Depth"]
        asm_num += ds["Scattering_Layer_Optical_Depth"] * ds["Layer_Asymmetry_Parameter"]
    asm = xr.where(sca > TAU_THRESH, asm_num / sca, 0.0).astype(np.float32)
    out["Extinction_Layer_Optical_Depth"] = ext.astype(np.float32)
    out["Scattering_Layer_Optical_Depth"] = sca.astype(np.float32)
    out["Layer_Asymmetry_Parameter"] = asm
    out["Extinction_Column_Optical_Depth"] = ext.sum("lev").astype(np.float32)
    out.attrs["mixed_modes"] = str(len(datasets))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Externally mix corrected mode files into one AER file")
    parser.add_argument("--output", required=True)
    parser.add_argument("mode_files", nargs="+")
    args = parser.parse_args(argv)
    datasets = [xr.open_dataset(path) for path in args.mode_files]
    out = mix_mode_datasets(datasets)
    out.to_netcdf(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_mode_external_mix -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mode_external_mix.py tests/test_mode_external_mix.py
git commit -m "Add corrected mode external mixing"
```

---

### Task 7: Mode Optics Dataset Builder

**Files:**
- Create: `mode_optics.py`
- Test: `tests/test_mode_external_mix.py`

- [ ] **Step 1: Add dataset-builder test**

Append to `tests/test_mode_external_mix.py`:

```python
from mode_optics import build_mode_output_dataset


class TestModeOutputDataset(unittest.TestCase):
    def test_builds_sarb_style_output(self):
        delp = xr.DataArray(np.ones((2, 1, 1), dtype=np.float32), dims=["lev", "lat", "lon"])
        ext = xr.DataArray(np.full((2, 1, 1), 0.2, dtype=np.float32), dims=["lev", "lat", "lon"])
        sca = xr.DataArray(np.full((2, 1, 1), 0.1, dtype=np.float32), dims=["lev", "lat", "lon"])
        asm = xr.DataArray(np.full((2, 1, 1), 0.7, dtype=np.float32), dims=["lev", "lat", "lon"])
        ds = build_mode_output_dataset(delp, ext, sca, asm, {"scheme": "MAM4", "mode": "a1"})
        self.assertIn("Extinction_Layer_Optical_Depth", ds)
        self.assertIn("Scattering_Layer_Optical_Depth", ds)
        self.assertIn("Layer_Asymmetry_Parameter", ds)
        self.assertAlmostEqual(float(ds["Extinction_Column_Optical_Depth"].values[0, 0]), 0.4)
        self.assertEqual(ds.attrs["scheme"], "MAM4")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m unittest tests.test_mode_external_mix.TestModeOutputDataset -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mode_optics'`.

- [ ] **Step 3: Implement dataset builder**

Create `mode_optics.py`:

```python
import argparse
import sys
import numpy as np
import xarray as xr


def build_mode_output_dataset(delp, tau_ext, tau_sca, asm, attrs):
    tau_ext = tau_ext.astype(np.float32)
    tau_sca = tau_sca.clip(min=0.0, max=tau_ext).astype(np.float32)
    asm = asm.clip(min=-1.0, max=1.0).astype(np.float32)
    ds = xr.Dataset(
        {
            "DELP": delp.astype(np.float32),
            "Extinction_Layer_Optical_Depth": tau_ext,
            "Scattering_Layer_Optical_Depth": tau_sca,
            "Layer_Asymmetry_Parameter": asm,
            "Extinction_Column_Optical_Depth": tau_ext.sum("lev").astype(np.float32),
        }
    )
    ds.attrs.update(attrs)
    return ds


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute native-grid internally mixed mode optics")
    parser.add_argument("--source", choices=["geosit", "merra2"], required=True)
    parser.add_argument("--scheme", default="MAM4")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--band", default=None)
    parser.add_argument("--wvl", type=float, default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--aerosol", default="aerosol.yaml")
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--external-vis", default=None)
    args = parser.parse_args(argv)
    if args.band is None and args.wvl is None:
        args.band = "sw01"
    if args.band is not None and args.wvl is not None:
        parser.error("--band and --wvl are mutually exclusive")
    raise SystemExit("mode_optics computation is added in Task 8")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run:

```bash
python -m unittest tests.test_mode_external_mix.TestModeOutputDataset -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mode_optics.py tests/test_mode_external_mix.py
git commit -m "Add mode optics output dataset builder"
```

---

### Task 8: Mode Optics Computation

**Files:**
- Modify: `mode_optics.py`
- Modify: `mode_physics.py`
- Test: `tests/test_mode_physics.py`

- [ ] **Step 1: Add synthetic optical-depth test**

Append to `tests/test_mode_physics.py`:

```python
from mode_physics import layer_optical_depth


class TestLayerOptics(unittest.TestCase):
    def test_layer_optical_depth_uses_number_and_cross_section(self):
        number = np.full((1, 1, 1), 10.0, dtype=np.float32)
        delp = np.full((1, 1, 1), 98.0, dtype=np.float32)
        ext_um2 = np.full((1, 1, 1), 2.0, dtype=np.float32)
        tau = layer_optical_depth(delp, number, ext_um2)
        self.assertAlmostEqual(float(tau[0, 0, 0]), 10.0 * 98.0 * 2.0e-12 / 9.8)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m unittest tests.test_mode_physics.TestLayerOptics -v
```

Expected: FAIL with `ImportError: cannot import name 'layer_optical_depth'`.

- [ ] **Step 3: Add optical-depth helper**

Append to `mode_physics.py`:

```python
def layer_optical_depth(delp_pa, number_per_kg, cross_section_um2):
    return (cross_section_um2.astype(np.float32) * 1.0e-12 * number_per_kg.astype(np.float32) * delp_pa.astype(np.float32) / 9.8).astype(np.float32)
```

- [ ] **Step 4: Wire `mode_optics.py` computation**

Replace the `raise SystemExit("mode_optics computation is added in Task 8")` line in `mode_optics.py` with this call:

```python
    return run(args)
```

Add these imports near the top of `mode_optics.py`:

```python
import os
import numpy as np
import pandas as pd
import yaml
import xarray as xr

from mode_config import load_config, resolved_allocations
from source_fields import open_source_fields
from utils import fill_date_hour_template
```

Add these functions above `main` in `mode_optics.py`:

```python
def _band_label(args):
    if args.wvl is not None:
        return f"{int(args.wvl)}NM"
    return args.band.upper()


def _date_strings(start, end):
    for date in pd.date_range(start=start, end=end, freq="3h"):
        yield date, date.strftime("%Y-%m-%b-%d-%j-%H")


def _source_spec(config, source):
    return config["Sources"][source.upper()]


def _mode_species(config, scheme, mode):
    allocations = resolved_allocations(config, scheme)
    return [species for species, weights in allocations.items() if mode in weights and float(weights[mode]) > 0.0]


def _build_path(root, pattern, date_str, label, band_label):
    path = fill_date_hour_template(pattern, date_str)
    path = path.replace("{label}", label).replace("{band}", band_label)
    return os.path.join(root, path)


def run(args):
    config = load_config(args.aerosol)
    source_key = args.source.upper()
    source_spec = _source_spec(config, args.source)
    band_label = _band_label(args)
    datadir = args.datadir or os.path.expandvars("$HOME/Data")
    outdir = args.outdir or datadir
    for _, date_str in _date_strings(args.start, args.end):
        species = _mode_species(config, args.scheme, args.mode)
        input_path = _build_path(datadir, source_spec["input_pattern"], date_str, args.mode, band_label)
        output_label = f"{args.scheme}_{args.mode}"
        output_path = _build_path(outdir, source_spec["output_pattern"], date_str, output_label, band_label)
        fields = open_source_fields(input_path, source_spec, species)
        print(f"read {input_path}")
    return 0
```

- [ ] **Step 5: Run unit tests**

Run:

```bash
python -m unittest tests.test_mode_physics tests.test_mode_external_mix -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mode_physics.py mode_optics.py tests/test_mode_physics.py
git commit -m "Start native-grid mode optics computation"
```

---

### Task 9: Finish Optics Lookup Wiring

**Files:**
- Modify: `mode_optics.py`
- Modify: `mode_physics.py`

- [ ] **Step 1: Inspect one SARB lookup file**

Run:

```bash
python - <<'PY'
import os
import xarray as xr
path = os.path.expandvars('${HOME}/Data/Optics/SARB/mam4_mode1_larc_c000002.v2.nc')
ds = xr.open_dataset(path)
print(ds)
PY
```

Expected: dataset lists `n_real`, `n_imag`, `radius`, `ext`, `abs`, and `asm` variables. If the file is unavailable locally, run the same command on CERES with `/CERES/sarb/dfillmor/Optics/SARB/mam4_mode1_larc_c000002.v2.nc`.

- [ ] **Step 2: Add lookup wrapper**

Add to `mode_physics.py`:

```python
def lookup_mode_optics(n_re, n_im, r_w_um, ds_table):
    n_re_table = ds_table["n_real"].values
    n_im_table = ds_table["n_imag"].values
    radius_table = ds_table["radius"].values
    ext_table = ds_table["ext"].values
    abs_table = ds_table["abs"].values
    asm_table = ds_table["asm"].values
    i_re = np.clip(np.searchsorted(n_re_table, n_re) - 1, 0, len(n_re_table) - 1)
    i_im = np.clip(np.searchsorted(n_im_table, np.abs(n_im)) - 1, 0, len(n_im_table) - 1)
    i_r = np.clip(np.searchsorted(radius_table, r_w_um) - 1, 0, len(radius_table) - 1)
    return (
        ext_table[i_re, i_im, i_r].astype(np.float32),
        abs_table[i_re, i_im, i_r].astype(np.float32),
        asm_table[i_re, i_im, i_r].astype(np.float32),
    )
```

- [ ] **Step 3: Add mode table and source refractive readers**

Add these functions to `mode_optics.py`:

```python
def _mode_table_path(mode_spec, args):
    path = os.path.expandvars(mode_spec["filename_sarb"])
    if args.wvl is not None:
        return path.replace("larc", f"{int(args.wvl)}nm_larc")
    return path.replace("larc", f"{args.band}_larc")


def _band_wavelength_um(args, ds_table):
    if args.wvl is not None:
        return float(args.wvl) / 1000.0
    band_idx = int(args.band[2:])
    if args.band.startswith("sw"):
        edges = ds_table["LFL_SW_bands"].values
    else:
        edges = ds_table["LFL_LW_bands"].values
    return 0.5 * (float(edges[band_idx]) + float(edges[band_idx - 1]))


def _type_dataset(type_info):
    return xr.open_dataset(os.path.expandvars(type_info["filename"]))


def _nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def _type_refractive_index(type_name, type_info, wavelength_um):
    ds = _type_dataset(type_info)
    if type_name == "WAT":
        wavelengths = ds["wavelength1"].values
        idx = _nearest_index(wavelengths, wavelength_um)
        return float(ds["watern"].values[idx]), float(ds["wateri"].values[idx])
    wavelengths = ds["lambda"].values * 1.0e6
    idx = _nearest_index(wavelengths, wavelength_um)
    return float(ds["refreal"].values[0, 0, idx]), float(ds["refimag"].values[0, 0, idx])


def _type_key(species_name):
    if species_name.startswith("NO3"):
        return "NI"
    if species_name.startswith("OC"):
        return "POM"
    if species_name.startswith("BC"):
        return "BC"
    if species_name.startswith("DU"):
        return "DU"
    if species_name.startswith("SS"):
        return "SS"
    if species_name == "SO4":
        return "SU"
    return species_name


def _species_info(config, species_names):
    return {name: config["Types"][_type_key(name)] for name in species_names}


def _refractive_indices(config, species_names, wavelength_um):
    refractive = {}
    for name in species_names:
        type_key = _type_key(name)
        refractive[name] = _type_refractive_index(type_key, config["Types"][type_key], wavelength_um)
    refractive["WAT"] = _type_refractive_index("WAT", config["Types"]["WAT"], wavelength_um)
    return refractive
```

- [ ] **Step 4: Run compile check**

Run:

```bash
python -m compileall mode_optics.py mode_physics.py
```

Expected: both files compile.

- [ ] **Step 5: Commit**

```bash
git add mode_optics.py mode_physics.py
git commit -m "Wire mode lookup table access"
```

---

### Task 10: Complete Mode Output and VIS Scaling Flow

**Files:**
- Modify: `mode_optics.py`
- Modify: `mode_external_mix.py`
- Modify: `README.md`

- [ ] **Step 1: Complete mode field computation**

In `mode_optics.py`, replace the `print(f"read {input_path}")` line in `run` with the computation using functions already added:

```python
        from mode_physics import derive_number_mixing_ratio, layer_optical_depth, lookup_mode_optics, mix_mode_state

        mode_spec = config["Schemes"][args.scheme]["modes"][args.mode]
        table_path = _mode_table_path(mode_spec, args)
        ds_table = xr.open_dataset(table_path)
        wavelength_um = _band_wavelength_um(args, ds_table)
        allocations = resolved_allocations(config, args.scheme)
        q = {name: fields.species[name] * float(allocations[name][args.mode]) for name in species}
        species_info = _species_info(config, q.keys())
        refractive = _refractive_indices(config, q.keys(), wavelength_um)
        state = mix_mode_state(species_info, q, refractive, fields.rh, fields.temperature, mode_spec["dry_radius_um"])
        number = derive_number_mixing_ratio(state["dry_volume"], mode_spec["dry_radius_um"], mode_spec["sigma_g"])
        ext_um2, abs_um2, asm = lookup_mode_optics(state["n_re"], state["n_im"], state["r_w_um"], ds_table)
        tau_ext = layer_optical_depth(fields.delp, number, ext_um2)
        tau_abs = layer_optical_depth(fields.delp, number, abs_um2)
        tau_sca = np.clip(tau_ext - tau_abs, 0.0, tau_ext)
        delp_da = xr.DataArray(fields.delp[0], dims=["lev", "lat", "lon"], coords={k: v for k, v in fields.coords.items() if k in ["lev", "lat", "lon"]})
        ext_da = xr.DataArray(tau_ext[0], dims=["lev", "lat", "lon"], coords=delp_da.coords)
        sca_da = xr.DataArray(tau_sca[0], dims=["lev", "lat", "lon"], coords=delp_da.coords)
        asm_da = xr.DataArray(asm[0], dims=["lev", "lat", "lon"], coords=delp_da.coords)
        out = build_mode_output_dataset(
            delp_da,
            ext_da,
            sca_da,
            asm_da,
            {"source": source_key, "scheme": args.scheme, "mode": args.mode, "Langley_Fu_Liou_band": band_label},
        )
        out.to_netcdf(output_path)
```

- [ ] **Step 2: Add README commands**

Append to `README.md`:

```markdown

## Native-Grid Mode Optics

Compute one internally mixed mode on the source native grid:

```bash
python mode_optics.py --source geosit --scheme MAM4 --mode a1 --wvl 550 --start 2010-01-01T00 --end 2010-01-01T00
```

Externally mix corrected modes:

```bash
python mode_external_mix.py --output /path/to/AER_MAM4_550NM.nc4 /path/to/MAM4_a1_550NM.nc4 /path/to/MAM4_a2_550NM.nc4 /path/to/MAM4_a3_550NM.nc4 /path/to/MAM4_a4_550NM.nc4
```
```

- [ ] **Step 3: Run syntax check**

Run:

```bash
python -m compileall .
```

Expected: command exits 0.

- [ ] **Step 4: Run unit tests**

Run:

```bash
python -m unittest discover -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mode_optics.py mode_external_mix.py README.md
git commit -m "Complete native-grid mode optics workflow"
```

---

### Task 11: Integration Smoke Checks

**Files:**
- No code files unless the smoke run exposes a concrete defect.

- [ ] **Step 1: Run GEOSIT VIS smoke when data exists**

Run:

```bash
python mode_optics.py --source geosit --scheme MAM4 --mode a1 --wvl 550 --start 2010-01-01T00 --end 2010-01-01T00
```

Expected: one `MAM4_a1_550NM` SARB-style NetCDF file on the GEOSIT native grid.

- [ ] **Step 2: Run all GEOSIT MAM4 modes at VIS**

Run:

```bash
for mode in a1 a2 a3 a4; do
    python mode_optics.py --source geosit --scheme MAM4 --mode "$mode" --wvl 550 --start 2010-01-01T00 --end 2010-01-01T00
done
```

Expected: four mode files.

- [ ] **Step 3: Mix modes to total**

Run `mode_external_mix.py` with the four mode paths from Step 2 and an output name containing `AER_MAM4_550NM`.

Expected: total file has `Extinction_Column_Optical_Depth` and native `lat/lon`.

- [ ] **Step 4: Run MERRA2 smoke when `inst3_3d_aer_Nv` data exists**

Run:

```bash
python mode_optics.py --source merra2 --scheme MAM4 --mode a1 --wvl 550 --start 2010-01-01T00 --end 2010-01-01T00
```

Expected: one `MAM4_a1_550NM` MERRA2 native-grid file.

- [ ] **Step 5: Commit smoke-check corrections**

After a smoke-check correction, rerun the failing smoke command, then commit only the native-grid files touched by the correction:

```bash
git add mode_optics.py mode_physics.py source_fields.py mode_external_mix.py aerosol.yaml aerosol_ceres.yaml README.md
git commit -m "Fix native-grid smoke issue"
```

---

## Final Verification

- [ ] Run all unit tests:

```bash
python -m unittest discover -v
```

Expected: PASS.

- [ ] Run syntax check:

```bash
python -m compileall .
```

Expected: exit 0.

- [ ] Check workspace:

```bash
git status --short
```

Expected: only intentional untracked local files remain.
