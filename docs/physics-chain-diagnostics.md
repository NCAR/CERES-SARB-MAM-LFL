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

## Findings

- The reported MAM AOD mean is an area-weighted global mean over the native GEOSIT grid for one instantaneous field.
- Native-grid external-species physics reproduces alpha_4: `0.143699` versus alpha_4 `0.143692`.
- Current MAM mode-sum is low: `0.039907`, a factor `3.60068` below alpha_4.

## Checked Stages

- Input mass is not the issue. External species dry mass column is `8.76496e-05 kg/m2`; MAM mode allocations conserve the same total mass.
- SW05 band selection is correct. Bounds are `0.4975-0.595 um`, midpoint `0.54625 um`, and `sw5` fine LUTs match base SARB `extpsw_mie` band 5.
- LUT clipping is not the issue. For a3, selected radius is `0.4-7.91306 um` inside table range `0.01-24.9375 um`.
- Production dust optics now align with the reference repo: `optics_DU.v15_3.nc`.

## First Failing Stage

The loss starts in the MAM optical representation after mass allocation:

- External-species total effective extinction: `1639.47 m2/kg`.
- Current MAM mode-sum effective extinction: about `455 m2/kg`.
- a3 carries most mass and gives `0.0388693` AOD with `495.607 m2/kg`.
- a1 fine-mode mass gives only `0.000973925` AOD with `126.676 m2/kg`; the same fine species in the external path have much larger mass extinction.

The base SARB `extpsw_scaled` variables are not a drop-in fix:

- a1 with `extpsw_scaled * dry_mass` gives `0.00413`, still too low.
- a3 with `extpsw_scaled * dry_mass` gives `0.508`, too high.

## Open Physics Checks

- Decide whether GEOSIT bins should remain bin-resolved inside each MAM mode instead of collapsing to one mode radius before optics lookup.
- Reconcile configured mode radii with MAM table metadata. RRTMG `0.5 * dgnum` gives a3 radius `1.0 um` and a4 radius `0.025 um`, while current config uses a3 `0.40 um` and a4 `0.050 um`.
- Check whether water uptake should be mass-weighted by species/bin before mode-level refractive-index and radius lookup.
