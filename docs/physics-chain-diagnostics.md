# GEOSIT MAM Physics Chain Diagnostics

Reference repo: `../CERES-SARB-GEOSIT-LFL`

Smoke case:

- Time: `2008-07-01T0000`
- Band: `SW05`
- Input: `/CERES_prd/GMAO/GEOSIT/2008/07/GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294.2008-07-01T0000.V01.nc4`
- MAM output: `/CERES/sarb/dfillmor/GEOSIT-MAM/2008/07`
- alpha_4 reference: `/CERES/sarb/dfillmor/GEOSIT_alpha_4/2008/07/GEOS.it.asm.aer_inst_3hr_glo_L288x180_v24.GEOS5294.AER_SW05.2008-07-01T0000.V01.nc4`

## Commands

```bash
/homedir/dfillmor/miniconda3/envs/sarb/bin/python -u diagnose_mode_physics.py \
  --aerosol aerosol_ceres.yaml \
  --source geosit \
  --scheme MAM4 \
  --band sw05 \
  --time 2008-07-01T00 \
  --reference-total /CERES/sarb/dfillmor/GEOSIT_alpha_4/2008/07/GEOS.it.asm.aer_inst_3hr_glo_L288x180_v24.GEOS5294.AER_SW05.2008-07-01T0000.V01.nc4
```

```bash
/homedir/dfillmor/miniconda3/envs/sarb/bin/python -u diagnose_external_species.py \
  --aerosol ../CERES-SARB-GEOSIT-LFL/aerosol_ceres.yaml \
  --input /CERES_prd/GMAO/GEOSIT/2008/07/GEOS.it.asm.aer_inst_3hr_glo_L576x361_v72.GEOS5294.2008-07-01T0000.V01.nc4 \
  --band sw05 \
  --optics-dir /CERES/sarb/dfillmor/Optics \
  --reference-total /CERES/sarb/dfillmor/GEOSIT_alpha_4/2008/07/GEOS.it.asm.aer_inst_3hr_glo_L288x180_v24.GEOS5294.AER_SW05.2008-07-01T0000.V01.nc4
```

Use `--mode a3` with `diagnose_mode_physics.py` for a faster focused check of the dominant coarse mode.

Compare SARB LUT cross sections against independent homogeneous-sphere Mie
calculations:

```bash
/homedir/dfillmor/miniconda3/envs/sarb/bin/python -u diagnose_mie_lut.py \
  --aerosol aerosol_ceres.yaml \
  --mode a3 \
  --band sw05 \
  --time 2008-07-01T00 \
  --sample-count 3
```

## Findings

- The reported MAM AOD mean is an area-weighted global mean over the native GEOSIT grid for one instantaneous field.
- Native-grid external-species physics reproduces alpha_4: `0.143699` versus alpha_4 `0.143692`.
- Current MAM mode-sum is low: `0.039907`, a factor `3.60068` below alpha_4.

## Checked Stages

- Input mass is not the issue. External species dry mass column is `8.76496e-05 kg/m2`; MAM mode allocations conserve the same total mass.
- SW05 band selection is correct. Bounds are `0.4975-0.595 um`, midpoint `0.54625 um`, and `sw5` fine LUTs match base SARB `extpsw_mie` band 5.
- LUT clipping is not the issue. For a3, selected radius is `0.4-7.91306 um` inside table range `0.01-24.9375 um`.
- Production dust optics now align with the reference repo: `optics_DU.v15_3.nc`.

## Mode Geometry Mismatch

Current `aerosol.yaml` and `aerosol_ceres.yaml` point at SARB
`mam4_mode*_larc_c000002.v2.nc` tables, but `Schemes.MAM4.modes` does not
match the scalar mode geometry stored in those tables.

`ncdump -v sigmag,dgnum,dgnumlo,dgnumhi` on the CERES production files shows:

| mode | config `dry_radius_um` | config `sigma_g` | RRTMG `0.5 * dgnum` | RRTMG `sigmag` | SARB v2 `0.5 * dgnum` | SARB v2 `sigmag` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | 0.055 | 1.8 | 0.055 | 1.8 | 0.055 | 1.6 |
| a2 | 0.012 | 1.6 | 0.013 | 1.6 | 0.013 | 1.6 |
| a3 | 0.40 | 1.8 | 1.0 | 1.8 | 0.45 | 1.2 |
| a4 | 0.050 | 1.6 | 0.025 | 1.60000002384186 | 0.025 | 1.6 |

This appears to be SARB table version drift. The original
`mam4_mode*_larc_c000000.nc` SARB tables used a1/a3 `sigmag = 1.8` and a3
`dgnum = 2`, matching the RRTMG mode widths and coarse-mode diameter. The v2
SARB tables used by the configs changed a1 to `sigmag = 1.6` and a3 to
`sigmag = 1.2`, `dgnum = 0.9`. The current native-grid path uses the YAML
`sigma_g` in `derive_number_mixing_ratio`, so modes a1 and a3 are normalized
with a different lognormal width than the SARB lookup table metadata.

## First Failing Stage

The loss starts in the MAM optical representation after mass allocation:

- External-species total effective extinction: `1639.47 m2/kg`.
- Current MAM mode-sum effective extinction: about `455 m2/kg`.
- a3 carries most mass and gives `0.0388693` AOD with `495.607 m2/kg`.
- a1 fine-mode mass gives only `0.000973925` AOD with `126.676 m2/kg`; the same fine species in the external path have much larger mass extinction.

The base SARB `extpsw_scaled` variables are not a drop-in fix:

- a1 with `extpsw_scaled * dry_mass` gives `0.00413`, still too low.
- a3 with `extpsw_scaled * dry_mass` gives `0.508`, too high.

Independent Mie checks indicate the SW05 LUT values themselves may also be low
for the selected internal-mix points. The local environment does not have
`miepython`, `PyMieScatt`, or `pymiecoated`, so `mie_sphere.py` uses SciPy's
spherical Bessel functions for a direct homogeneous-sphere calculation.

Representative a3/SW05 samples:

- `r=0.40818 um`, `n=1.52405 + 0.00173i`: LUT `0.323511 um2`, independent Mie `1.95196 um2`.
- `r=0.439189 um`, `n=1.472 + 0.000001i`: LUT `0.271317 um2`, independent Mie `2.42699 um2`.
- `r=1.12499 um`, `n=1.34072 + 0.0000147i`: LUT `3.47069 um2`, independent Mie `8.31453 um2`.
- `r=7.91306 um`, `n=1.33302 + 0i`: LUT `113.663 um2`, independent Mie `424.045 um2`.

This does not yet prove the SARB LUT generation is wrong, because the SARB
MAM tables may encode a mode-averaged or band-averaged quantity rather than a
plain homogeneous-sphere cross section at band midpoint. It does show the LUT
cross sections are much smaller than a direct Mie calculation at comparable
selected points.

## Open Physics Checks

- Decide whether GEOSIT bins should remain bin-resolved inside each MAM mode instead of collapsing to one mode radius before optics lookup.
- Reconcile configured mode radius/sigma values with the SARB table version actually used by `filename_sarb`.
- Check whether water uptake should be mass-weighted by species/bin before mode-level refractive-index and radius lookup.
- Confirm whether SARB `extpsw_mie` is intended to equal homogeneous-sphere
  midpoint Mie cross sections or a different band/mode-averaged quantity.
