# Native-Grid Mode Optics Design

## Overview

Create a new native-grid aerosol optics path for GEOSIT and MERRA2. The path
internally mixes source aerosol species into YAML-defined modes such as MAM3 or
MAM4, computes SARB/LFL SW and LW band properties on each source grid, applies a
VIS column-preserving correction, and externally mixes corrected modes into a
scheme total.

## Goals

- Support `--source geosit` and `--source merra2`.
- Keep each source on its native horizontal and vertical grid.
- Write SARB-style NetCDF outputs matching the GEOSIT repository conventions.

## Non-Goals

- No SARB `288x180x24` subsampling in this path.
- No species-by-species output, except for existing external-reference inputs.
- No hard-coded MAM3 or MAM4 membership in Python.

## Inputs

GEOSIT input uses the existing `aer_inst_3hr_glo_L576x361_v72` files. MERRA2
input uses `inst3_3d_aer_Nv` files. Both sources provide aerosol mass mixing
ratios, RH, pressure thickness, and native coordinates. If a source lacks direct
pressure thickness, the source adapter must derive it before optics.

External-reference VIS AOD comes from the existing species-external-mix path at
550 nm. It is used only to build the column correction factor.

## Configuration

YAML owns source variables, schemes, modes, and aerosol type properties.

`Sources` defines source file patterns, output patterns, coordinate names, and
variable names or aliases. `Schemes` defines modes. Each mode includes dry
geometric radius, geometric sigma, SARB lookup table path, RRTMG/MAM metadata
path if needed, and a species membership list. `Types` keeps optical constants,
density, and hygroscopicity coefficients.

## Mixing State Defaults

YAML determines source-species allocation into modes. Within each mode, allocated
species are internally mixed. Between modes, mode outputs remain externally mixed
until `mode_external_mix.py` sums them.

Size-binned fields such as `DU001..DU005`, `SS001..SS005`, and
`NO3AN1..NO3AN3` are allocated to modes by overlap between each source-bin
radius and the configured mode lognormal dry size distribution. Fractions are
normalized so each source species sums to 1.0 across modes.

Bulk soluble or aged species default mostly to accumulation mode. Bulk fresh
hydrophobic carbon defaults to the primary carbon mode when the scheme has one.

MAM4 defaults:

- `SO4`: `a1=0.90`, `a2=0.10`, `a3=0.00`, `a4=0.00`
- `OCPHILIC`, `BCPHILIC`: `a1=1.00`
- `OCPHOBIC`, `BCPHOBIC`: `a4=1.00`
- bulk `NO3`: `a1=0.70`, `a3=0.30`
- bulk `POM`, `SOA`: `a1=0.80`, `a2=0.20`

For MAM3, any primary-carbon `a4` allocation maps to accumulation `a1` unless a
scheme-specific YAML override is provided.

## Processing

Add `mode_optics.py` as the main script.

For each source, date, scheme, mode, and band:

1. Read source-native fields.
2. Map configured source variables into mode species.
3. Compute dry volume, mixed hygroscopicity, wet radius, mixed refractive index,
   extinction, absorption, asymmetry, and layer optical depths.

Mode number concentration is derived from dry volume and the configured
lognormal dry radius/sigma. Wet radius is derived from mixed hygroscopicity, RH,
and temperature through the full Kohler equation, not from a source-provided wet
mode radius. Temperature is read from the source file when available and falls
back to `273.15 K` when absent.

Kohler water uptake solves:

`log(RH) = A / r_w - B * r_d^3 / (r_w^3 - r_d^3)`

where `A = 2 M_w sigma / (R T rho_w)`, `B` is the mixed hygroscopicity, `r_d` is
dry radius, and `r_w >= r_d` is the solved wet radius. The implementation must
use a bounded numeric solve and fall back to `r_w = r_d` for dry or
non-hygroscopic cells.

## VIS Correction

The correction preserves total VIS column AOD when moving from species-external
mixing to mode-internal mixing.

Order:

1. Compute uncorrected VIS mode files for all modes.
2. Externally mix those modes into an uncorrected internal total VIS column.
3. Compute `factor(lat, lon) = external_total_VIS / internal_total_VIS`.
4. Cap factors to `0.25..4.0`; skip correction where both columns are below a
   tiny AOD threshold.
5. Apply the same 2D factor to every layer in every mode for all SW and LW bands.

The factor is applied last to layer extinction and scattering optical depths.
Column AOD is recomputed after correction. The vertical shape of each column is
preserved.

## Outputs

Mode output files use SARB-style variables:

- `DELP`
- `Extinction_Layer_Optical_Depth`
- `Scattering_Layer_Optical_Depth`
- `Layer_Asymmetry_Parameter`
- `Extinction_Column_Optical_Depth`

Files are one per source, scheme, mode, band, and timestep. File names include
the scheme and mode, for example `MAM4_a1_SW01`.

Add `mode_external_mix.py` to sum corrected mode files into total files. Total
filenames include the scheme, for example `AER_MAM4_SW01`.

## Validation

Required checks:

- Native grid dimensions and coordinates are preserved.
- Extinction and scattering are nonnegative.
- Scattering optical depth is not greater than extinction optical depth.

VIS preservation check:

- Corrected total VIS column matches external species-mix VIS column within
  float tolerance, except skipped tiny-AOD columns or capped-factor columns.
- Report global mean difference, max absolute difference, capped-column count,
  and skipped-column count.

Missing required files or variables stop processing for that timestep.

## Testing

Start with one GEOSIT timestep at 550 nm, using existing external VIS output as
the reference. Then run one SW band and one LW band for all MAM4 modes. Repeat
with one MERRA2 `inst3_3d_aer_Nv` timestep when sample data is available.

Run `python -m compileall .` after code changes.
