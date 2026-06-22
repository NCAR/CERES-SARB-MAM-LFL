# Plan: reduce MAM–alpha_4 AOD bias

Plan for bringing the MAM mode-optics column AOD closer to the alpha_4
external-species reference. Grounded in the per-species diagnosis of the
GEOS-IT 2008-07-01T00 SW05 slice (see the validated numbers below).

## Diagnosis (validated) — the +32% is two opposing biases

MAM total 0.189 vs alpha_4 0.144 (+32%) is **partly accidental cancellation**.
Per-species column AOD (area-weighted global mean), reference reconciled exactly
(`SO4002` is not in the external mix — `external_mix.py:SPECIES_NO_BIN` — so
alpha_4 fine = 0.076, and 0.076 + dust 0.035 + sea-salt 0.032 = 0.144):

| family | MAM | alpha_4 | Δ | ratio |
| --- | ---: | ---: | ---: | ---: |
| **sea salt** | 0.116 | 0.032 | **+0.084** | **3.6×** |
| dust | 0.043 | 0.035 | +0.008 | 1.23× |
| **internal fine (SO₄+C+NO₃)** | 0.030 | 0.076 | **−0.046** | **0.39×** |
| total | 0.189 | 0.144 | +0.045 | 1.32× |

Same input mass in both models, so each ratio is a mass-extinction-efficiency
(MEE) ratio. **Because sea-salt (+0.084) and internal (−0.046) offset, a
one-sided fix moves the total the wrong way** — both must be fixed, judged
per family, not on the net total.

Method check: applying GOCART `bext` to the real SS field reproduces alpha_4
(0.0330 vs 0.0325), confirming the diagnostic framework.

## Fix 1 — sea salt: cap hygroscopic growth near saturation

**Root cause (validated).** GOCART caps sea-salt growth at RH≈0.95 — its
`rEff`/`bext` plateau (GF 2.61, MEE 4.85 m²/g held flat over RH 0.95→0.99 in
`optics_SS.v3_3.nc`, whose `rh` grid ends at 0.99). MAM's Köhler growth runs
away: GF 2.9→6.1 and MEE 6.7→28.5 m²/g over RH 0.95→0.995 (production clips RH
only at 1−10⁻⁶). ~10% of SS003 mass sits above RH 0.985, so that runaway tail
dominates the column integral → ss3 6× too bright.

**Change.** Cap the RH used for **bin hygroscopic growth** at `rh_growth_cap`
(default **0.95**, matching the GOCART table plateau). Applies to the external
monodisperse bins (σ_g = 1.0); internal modes a1–a4 stay uncapped.

- Add `rh_growth_cap` to the sea-salt (and, pending Fix-1 validation, dust) bin
  specs in `aerosol.yaml` / `aerosol_ceres.yaml`, or to the `SS`/`DU` `Types`.
- In `mode_optics.py` `compute_mode_dataset` (the `mix_mode_state` call at
  ~line 518), pass `np.minimum(fields.rh, cap)` when the mode carries a cap;
  leave `fields.rh` untouched otherwise. The cap only *lowers* the wet radius,
  so `lookup_mode_optics` stays inside the LUT range — no table-range risk.
- Keep it config-driven (no hard-coded 0.95) so the cap is auditable and the
  uncapped behaviour is recoverable.

**Alternative considered.** Adopt the GOCART per-bin growth curve
(`rEff(RH)`) directly instead of Köhler for the external bins. More faithful to
the reference but a larger change; defer unless the cap leaves too much residual.

**Expected result (to confirm by re-run).** Sea-salt total 0.116 → ~0.04–0.05
(reconstruction with cap = 0.040; alpha_4 = 0.033). A residual ~1.2× baseline
will remain below the cap — monodisperse single-particle Mie vs GOCART's
bin-integrated optics — and is the subject of a possible follow-up.

**Validation.** Re-run the 2008-07-01T00 slice for ss1–ss5 on the local L576
input; check each bin and the sea-salt total against alpha_4 (SS001–005).
`verify_physics` is unaffected — the cap is an input clip ahead of the
(unchanged) Köhler solver, so stage E monotonicity/residual still hold.

## Fix 2 — internal modes: recover the 2.5× MEE deficit

**Root cause (validated, structural — not the old LUT bug).** These files are
c000003.v2 (total 0.189 matches the fixed bin-resolved value; stage G verified
to 0.5%). Same fine-aerosol mass ⇒ the 0.39 AOD ratio is the MEE ratio: the
internal accumulation mode (r_g = 0.055 µm, σ_g = 1.8, volume-mixed) gives
**2.5× less extinction per unit fine mass** than GOCART's external per-species
optics. OC is a large part of the gap (alpha_4 OCPHILIC alone = 0.028).

**Step 2a — decompose first (diagnostic).** Before changing anything, attribute
the deficit:
- Compare the accumulation-mode MEE at SW05 vs GOCART `SO4` and `OCPHILIC`
  `bext`, across RH, to see whether the gap is dry-size, growth, or
  mixing-state, and whether sulfate or OC dominates.
- Check the mode dry radius / σ_g and resulting mass-weighted effective radius
  against GOCART's per-species effective radii (which are larger / more
  optically efficient at 0.55 µm).

**Step 2b — candidate fixes (choose after 2a).**
- Adjust the accumulation/Aitken mode geometry (dry radius and/or σ_g) so the
  mode MEE matches the GOCART fine-species MEE; or
- Treat the most divergent species (likely OC) with its own size/growth; or
- If the gap is an intrinsic internal-vs-external mixing-state difference,
  document it as a known structural offset rather than "fix" it.

**Expected result (to confirm).** Internal fine total 0.030 → ~0.076.

**Validation.** Re-run the slice; compare MAM internal (a1–a4) against alpha_4
fine (SO4 + BCPHILIC + BCPHOBIC + OCPHILIC + OCPHOBIC + NO3AN1–3), and confirm
no regression in `verify_physics` / unit tests.

## Dust (minor, +0.008, 1.23×)

Dust κ is low (0.14) so growth is not the driver; the modest overshoot is the
same monodisperse-vs-bin-integrated baseline as the sea-salt residual. Revisit
only after Fixes 1–2; the RH cap may be extended to dust if it helps.

## Sequencing & success criterion

1. **Fix 1** (clean, validated lever) → re-run slice, confirm sea salt ≈ 0.04.
   The *total* will temporarily undershoot 0.144 — expected, because the
   internal deficit is still present.
2. **Fix 2** (decompose, then fix) → re-run slice, confirm internal ≈ 0.076.
3. **Success = per-family agreement** with alpha_4 on the slice (sea salt 0.033,
   dust 0.035, internal 0.076, total 0.144). The net total alone is not a valid
   metric while the biases offset.

Validation runs use the local L576 slice + optics; production deployment needs
the full mode-optics regen on the CERES host.
