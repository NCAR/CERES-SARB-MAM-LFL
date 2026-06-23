# Plan: diagnose the under-absorbing VIS single-scattering albedo

Plan for the follow-on session. The 2008-07-01T00 single-slice run produced a
visible column single-scattering albedo **SSA ≈ 0.99** (sw02–sw12), versus the
alpha_4 reference **0.95** on the *same* GEOS-IT mass — i.e. MAM produces ~7×
too little absorption. SSA is intrinsic (the SW05 scaling factor scales EXT and
SCA together and cancels), so this is a pure optics problem, independent of the
AOD-anchoring work. Goal: localize where the absorption is lost and decide a fix.

## Diagnosis (validated) — absorption is BC + dust, and MAM loses it

Reference = alpha_4 GOCART optics applied to the same GEOS-IT species mass
(external mixing, each species keeps its own optics). Area-weighted **global
column** values at 0.55 µm (ratio-of-sums; per-species files sum to the combined
AER file with EXT/SCA ratios = 1.0000):

| quantity | reference (alpha_4) | MAM run |
| --- | ---: | ---: |
| column AOD (EXT) | 0.1437 | 0.133 (sw05, anchored) |
| column SCA | 0.1365 | — |
| column ABS | 0.0072 | — |
| **column SSA = SCA/EXT** | **0.9498** | **0.993** |
| absorbing fraction (1−SSA) | 0.050 | 0.007 (**~7× low**) |

Reference absorption budget by species (where the 0.0072 ABS comes from):

| family | EXT | SSA | share of ABS |
| --- | ---: | ---: | ---: |
| **Black Carbon** | 0.0051 | **0.208** | **55.6 %** |
| **Dust** | 0.0352 | 0.925 (0.97 fine → 0.77 coarse) | **36.7 %** |
| Organic (OC/POM/SOA) | 0.0308 | 0.982 | 7.7 % |
| Sulfate / Sea salt / Nitrate | 0.072 / 0.032 / 0.006 | 1.000 | 0 % |

**So recovering MAM's SSA means recovering BC absorption (dominant) and dust
absorption (secondary).** Everything else is essentially pure-scattering.

## Mechanism (code-grounded) — volume-averaged k dilutes BC

The mode absorption cross section comes from the LUT indexed by the mode's
**internal-mix imaginary refractive index** `n_im`:

- `cross_ext, cross_abs = lookup_mode_optics(n_re, n_im, r_w_um, ds_table)`
  → `tau_abs`; `tau_sca = clip(tau_ext − tau_abs)` (`mode_optics.py:540–548`).
- `n_im` is a **volume-weighted average over the mode's species *and* Köhler
  water** (`mode_physics.py:298–313`): `n_im = Σ(V_i·k_i)/ΣV_i`, water included
  in the denominator.

Imaginary indices at 0.55 µm (`~/Data/Optics/MERRA2/optics_*.nc`):

| species | k(0.55 µm) | n_real |
| --- | ---: | ---: |
| BC | **0.44** | 1.75 |
| Dust | 0.0024 | 1.53 |
| OC (POM/SOA) | 0.006 | 1.53 |
| SU / SS / NI | ~0 | 1.43 / 1.5 / 1.56 |

BC mass routing (allocations): **BCPHILIC → a1** (accumulation), **BCPHOBIC → a4**
(primary carbon). In a1, BC(k=0.44) is volume-mixed with sulfate (~0), organics
(0.006), nitrate (~0) **plus the Köhler water**, so
`n_im(a1) ≈ f_BC · 0.44` where `f_BC` is the BC wet-volume fraction — small, and
shrunk further by water uptake. Low `n_im` → `cross_abs ≈ 0` → SSA → 1.

This volume-average-k + homogeneous-sphere Mie is doubly biased for mixed BC: it
**omits the absorption (lensing) enhancement** that real internal mixing gives
*and* **dilutes k** across the particle. **Leading cause of the high SSA.**

Two secondary notes:
- **Sign-convention inconsistency:** `refimag` is stored signed — BC/SU/SS/OC/NI
  negative, **dust positive (+0.0024)**. `lookup_mode_optics` applies
  `np.abs(n_im)` (`mode_physics.py:72`), so the magnitude is recovered. But
  `mix_mode_state` volume-sums the *signed* values first
  (`mode_physics.py:305–306`), so a mode mixing a negative-convention and a
  positive-convention absorber would partially cancel before `abs()`. Current
  allocations never co-locate BC and dust, so it is latent — fix anyway.
- **BC density = 1.0** in config (`aerosol.yaml` Types) vs physical ~1.8 g cm⁻³.
  A density too low *inflates* BC volume and thus `f_BC`, partly masking the
  dilution; correcting it would *raise* SSA further. Track it.

## Investigation steps (run these first)

1. **Regenerate the 14 per-mode SW05 files** (uncorrected; ~3 min, serial) — the
   run deleted intermediates. Keep them this time. See the run procedure in
   memory `sw05-scaling-band-run`.
2. **Per-mode SSA decomposition.** `SSA_mode = ΣSCA/ΣEXT` and each mode's ABS
   contribution. Expect a1 and a4 to show SSA ≈ 1 *despite carrying all the BC* —
   that confirms the dilution localization.
3. **Quantify `n_im` for BC modes.** Print `state['n_im']` for a1/a4 and the BC
   wet-volume fraction; check `n_im ≈ |k_BC|·f_BC` and how much Köhler water
   suppresses it (compare dry vs wet `n_im`).
4. **Verify `cross_abs` LUT normalization** in `c000003.v2` against an
   independent homogeneous-sphere Mie absorption (`mie_sphere.py`) for a known
   (n_re, n_im, r). The v2 ext channel had a documented normalization bug
   ([[mam-optics-deficit-root-cause]]); confirm the abs channel was regenerated
   correctly and is not separately under-normalized.
5. **Confirm BC is fully allocated** (`resolved_allocations`: BCPHILIC→a1,
   BCPHOBIC→a4; no BC mass dropped) and re-examine BC density (1.0 → 1.8).
6. **Dust check.** Compare du-bin SSA to the reference dust budget (0.97 fine →
   0.77 coarse); confirm the mono-LUT absorption survives the lookup.

## Candidate fixes (decide after diagnosis)

- **If `cross_abs` is under-normalized** → regenerate the LUT abs channel
  (`generate_lut.py`/`refine_lut.py`); cleanest, mirrors the ext-channel fix.
- **If it is the volume-average dilution (method limitation)** → give BC-bearing
  modes an absorption-enhancing effective medium (core-shell / Maxwell-Garnett)
  instead of volume-averaged k, or an empirical lensing factor, or treat BC as
  its own external mode carrying undiluted optics (GOCART-like).
- **Fix the sign-convention inconsistency** and take per-species `abs(k)` inside
  `mix_mode_state` so mixing is robust regardless of file sign.
- **Revisit BC density** (1.0 → 1.8) once the dilution path is settled.

## Validation

- **Target:** MAM area-weighted column SSA(0.55 µm) → ≈ 0.95 (ratio-of-sums,
  like-for-like with the reference), with BC and dust supplying ~56 % / 37 % of
  absorption — matching the validated reference budget above.
- Per-mode SSA and the species-attributed absorption budget should track the
  reference table.
- **Add a `verify_physics.py` stage** comparing `cross_abs` to independent Mie
  absorption, so the abs channel is a permanent regression gate (the VIS-
  correction stage J is the model for this).

## Caveats

- Compare SSA **like-for-like**: use the ratio-of-area-weighted-sums (0.9498),
  not the per-cell mean (0.9585) — clean high-SSA cells inflate the per-cell mean.
- The reference is external-mixing optics on identical mass, so this is a clean
  optics/MEE comparison; the gap is a treatment difference, not an input-mass
  difference.

_Grounded in the 2008-07-01T00 slice: reference computed from the per-species
SW05/550NM files; MAM mechanism traced through `mode_optics.py` /
`mode_physics.py`. See [[sw05-scaling-band-run]] and
[[mam-optics-deficit-root-cause]]._
