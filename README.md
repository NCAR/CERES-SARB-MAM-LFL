# CERES-SARB-MAM-LFL

Python pre-processing that converts modal-aerosol (MAM4) mass fields into the
extinction / scattering / asymmetry optical properties consumed by the CERES SARB
Langley Fu-Liou (LFL) radiative-transfer code. Aerosols are internally mixed within
each mode and externally mixed between modes.

Physics documentation and the full stage-by-stage verification live in the
**[project wiki](https://github.com/NCAR/CERES-SARB-MAM-LFL/wiki)**. For production
GEOS-IT path conventions and the alpha_4 reference workflow, use
`../CERES-SARB-GEOSIT-LFL` as the companion reference repository.

## Scripts

### `verify_physics.py` — parameter-space physics verification

Verifies each stage of the internal-mixing / radiative chain against an independent
reference (analytic identity, conservation law, Köhler residual, or independent Mie).
Prints a pass/fail table and exits non-zero on any FAIL, so it doubles as a
regression gate. No 3-D model fields required.

```bash
python verify_physics.py              # all stages (currently 46 PASS / 0 FAIL / 5 FINDING)
python verify_physics.py --stage E,G  # selected stages
python verify_physics.py --aerosol aerosol_ceres.yaml
```

See [Physics Verification](https://github.com/NCAR/CERES-SARB-MAM-LFL/wiki/Physics-Verification)
and the in-repo report [`docs/physics-verification.md`](docs/physics-verification.md).

### `mode_optics.py` — native-grid mode optics

Computes one internally mixed, source-native mode:

```bash
python mode_optics.py --source geosit --scheme MAM4 --mode a1 --wvl 550 \
  --start 2010-01-01T00 --end 2010-01-01T00
```

For VIS-preserving correction, first compute uncorrected VIS mode files, externally
mix those into an uncorrected internal total, then rerun/apply each mode with the
same internal total:

```bash
python mode_external_mix.py --output /path/to/AER_MAM4_550NM_uncorrected.nc4 \
  /path/to/MAM4_a1_550NM.nc4 /path/to/MAM4_a2_550NM.nc4 \
  /path/to/MAM4_a3_550NM.nc4 /path/to/MAM4_a4_550NM.nc4
python mode_optics.py --source geosit --scheme MAM4 --mode a1 --wvl 550 \
  --start 2010-01-01T00 --end 2010-01-01T00 \
  --internal-vis /path/to/AER_MAM4_550NM_uncorrected.nc4
```

Key options: `--source {geosit,merra2}`, `--scheme`, `--mode`, one of `--band`/`--wvl`,
`--start`/`--end` (YYYY-MM-DDTHH, 3-hour steps), `--aerosol`, `--datadir`, `--outdir`,
`--external-vis`, `--internal-vis`.

### `mode_external_mix.py` — external mix of modes

Sums corrected per-mode files into the scheme total (extinction and scattering add;
asymmetry is scattering-weighted):

```bash
python mode_external_mix.py --output /path/to/AER_MAM4_550NM.nc4 \
  /path/to/MAM4_a1_550NM.nc4 /path/to/MAM4_a2_550NM.nc4 \
  /path/to/MAM4_a3_550NM.nc4 /path/to/MAM4_a4_550NM.nc4
```

### `generate_lut.py` — regenerate the SARB base optics tables

Builds `mam4_mode{1..4}_larc_c000003.v2.nc` as **number-averaged lognormal-mode**
Mie cross sections (the quantity `τ = σ·N` requires) at the canonical
σ_g = 1.8/1.6/1.8/1.6 and each SW/LW band midpoint.

```bash
python generate_lut.py                                    # local ~/Data/Optics/SARB
python generate_lut.py --optics-dir /CERES/sarb/dfillmor/Optics/SARB
python generate_lut.py --check                            # validate vs mie_lognormal, no write
```

Options: `--optics-dir`, `--modes 1 2 3 4`, `--template-suffix`, `--out-suffix`,
`--processes`. Background on why `c000003.v2` exists:
[Optics LUT Fix](https://github.com/NCAR/CERES-SARB-MAM-LFL/wiki/Optics-LUT-Fix).

### `refine_all.py` — rebuild all fine per-band tables

Runs `refine_lut.py` for every `(mode, band)` in parallel (14 SW + 12 LW × 4 modes =
104 tables), deriving band counts from the base table; exits non-zero on any failure.

```bash
python refine_all.py                                # all modes, all SW+LW bands
python refine_all.py --aerosol aerosol_ceres.yaml   # production paths
python refine_all.py --modes a3 --bands sw5 lw7
```

### `refine_lut.py` — refine one band's LUT

Interpolates one band's base-table Mie arrays onto the fine 100×100 refractive-index
grid the production lookup uses, and writes `…_<band>_larc_<ver>.nc`:

```bash
python refine_lut.py --aerosol aerosol.yaml --scheme MAM4 --mode a3 --band sw5
```

### `mode_internal_mix.py` — CAM6 path

Older CAM6 workflow that reads the model wet radius (`rwet_mode`) directly via
`microphysics.py`. See the wiki
[CAM6 guide](https://github.com/NCAR/CERES-SARB-MAM-LFL/wiki/README_CAM6).

### Diagnostics

```bash
python diagnose_mode_physics.py    --aerosol aerosol_ceres.yaml --source geosit --band sw05 --time 2008-07-01T00
python diagnose_external_species.py --aerosol aerosol_ceres.yaml --input <GEOSIT.nc4> --band sw05
python diagnose_mie_lut.py         --aerosol aerosol_ceres.yaml --mode a3 --band sw05 --time 2008-07-01T00
```

## Modules (not run directly)

- `mode_physics.py` — internal mixing, Köhler solver, number, layer optical depth, LUT lookup.
- `mie_sphere.py` — homogeneous-sphere Mie (ext/sca/abs efficiencies + asymmetry `g`).
- `mie_lognormal.py` — number-averaged lognormal-mode Mie (the optics reference).
- `mode_config.py` — config loading, allocation normalization, lognormal size-bin split.
- `source_fields.py`, `utils.py`, `microphysics.py`.

## Configuration

- `aerosol.yaml` — local paths (`${HOME}/Data/Optics`).
- `aerosol_ceres.yaml` — CERES production paths.
- `bands.yaml` — spectral band definitions.

## Environment

Python 3 with numpy, scipy, xarray, netCDF4, pandas, PyYAML. `microphysics.py`
additionally imports `numba`; the native-grid optics, the verifier, and the table
generators do **not** require it. Tests: `python -m unittest discover -s tests`.
