# CERES-SARB-MAM-LFL
Python scripts for CAM6 LFL pre-processing.

## Native-Grid Mode Optics

For production GEOS-IT path conventions and the alpha_4 reference workflow, use
`../CERES-SARB-GEOSIT-LFL` as the companion reference repository.

Compute one internally mixed source-native mode:

```bash
python mode_optics.py --source geosit --scheme MAM4 --mode a1 --wvl 550 --start 2010-01-01T00 --end 2010-01-01T00
```

For VIS-preserving correction, first compute uncorrected VIS mode files, externally mix those modes into an uncorrected internal total, then rerun/apply each mode with the same internal total:

```bash
python mode_external_mix.py --output /path/to/AER_MAM4_550NM_uncorrected.nc4 /path/to/MAM4_a1_550NM.nc4 /path/to/MAM4_a2_550NM.nc4 /path/to/MAM4_a3_550NM.nc4 /path/to/MAM4_a4_550NM.nc4
python mode_optics.py --source geosit --scheme MAM4 --mode a1 --wvl 550 --start 2010-01-01T00 --end 2010-01-01T00 --internal-vis /path/to/AER_MAM4_550NM_uncorrected.nc4
```

Externally mix corrected modes into the scheme total:

```bash
python mode_external_mix.py --output /path/to/AER_MAM4_550NM.nc4 /path/to/MAM4_a1_550NM.nc4 /path/to/MAM4_a2_550NM.nc4 /path/to/MAM4_a3_550NM.nc4 /path/to/MAM4_a4_550NM.nc4
```
