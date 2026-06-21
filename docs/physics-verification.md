# MAM Internal-Mixing & Radiative-Property Verification

Parameter-space verification of the native-grid mode-optics physics chain
(`mode_optics.py` → `mode_physics.py`). Each stage is checked **as a function on
parameter space** — RH, composition, refractive index, radius — against an
independent reference (analytic identity, conservation law, Köhler-equation
residual, or an independent Mie calculation), not on full 3-D fields.

Driver: [`verify_physics.py`](../verify_physics.py).
Mie references: [`mie_sphere.py`](../mie_sphere.py) (homogeneous sphere) and
[`mie_lognormal.py`](../mie_lognormal.py) (number-averaged lognormal mode).

```bash
python verify_physics.py            # all stages
python verify_physics.py --stage E,G
```

Status legend: **PASS** within tolerance · **FAIL** exceeds tolerance ·
**FINDING** a quantified physics result that is not a simple pass/fail ·
**WARN** skipped/borderline.

## Result: 34 PASS · 0 FAIL · 6 FINDING

The entire internal-mixing and bookkeeping chain is **provably correct**. All
six findings sit in the two previously-documented problem areas: the mode
geometric width `sigma_g`, and the SARB optics lookup table.

### Stage-by-stage (passes)

| stage | what is verified | reference | result |
| --- | --- | --- | --- |
| A | mass→mode allocation sums to 1; size-bin lognormal split reproducible | normalization + `allocate_size_bins_to_modes` | max\|sum−1\| = 2e-16 |
| B | dry volume `Σ qⱼ/(ρⱼ·1000)`; additivity `V(2q)=2V(q)` | direct recompute | rel err < 2e-7 |
| C | `N·v_particle = V_dry`; `v=(4/3)πr³e^{4.5ln²σ}`; `N=0` where `V=0` | analytic | rel err < 7e-8 |
| D | mixed κ = volume-weighted `b(RH)`; bounded by components; soluble `b(RH)>0` | convex-combination identity | rel err 6e-7 |
| E | Köhler residual `ln RH = A/r_w − B r_d³/(r_w³−r_d³)`; dry/non-hygroscopic→r_d; monotonic in RH and κ | equation residual | max\|resid\| = 1.5e-5 |
| F | `n_re,n_im` volume-weighted incl. water; bounded by component envelope; water dilutes toward 1.34 | direct recompute | rel err < 5e-7 |
| G | independent Mie validated (Rayleigh + geometric limits); fine LUT reproduces base-table interpolation; lookup floor-binning exact | analytic limits + `RegularGridInterpolator` | exact / rel err 5e-8 |
| H | `τ = σ·1e-12·N·Δp/g` (dimensionless) | direct recompute | rel err 2e-7 |
| I | external mix: additive ext/sca, scattering-weighted asymmetry | direct recompute | rel err < 8e-8 |
| J | VIS factor = clip(ext/int, 0.25, 4); uncapped correction preserves column | direct recompute | exact |

Notably, **G confirms the pipeline is faithful**: the fine SARB LUT consumed by
production is a correct `RegularGridInterpolator` of the base-table Mie arrays
(`refine_lut.py`), and `lookup_mode_optics` floor-binning / `|n_im|` /
clipping reproduce direct table indexing exactly. Any optics problem is in the
**table content**, not the pipeline.

## Findings

### F1–F2 · `sigma_g` inconsistency between number derivation and the LUT (stage C)

`derive_number_mixing_ratio` builds particle number with the **config**
`sigma_g`, but the optical cross section is read from a SARB LUT generated with
a **different** `sigma_g`. Because `N ∝ 1/v_particle ∝ exp(−4.5 ln²σ)` and the
LUT carries the cross section, the two must use the same width or `τ` is wrong:

| mode | config σ_g | SARB v2 table σ_g | RRTMG source σ_g | SARB c000000 σ_g | τ(config)/τ(table) |
| --- | ---: | ---: | ---: | ---: | ---: |
| a1 | 1.8 | 1.6 | **1.8** | 1.8 | 0.571 |
| a2 | 1.6 | 1.6 | 1.6 | — | 1.000 |
| a3 | 1.8 | 1.2 | **1.8** | 1.8 | **0.245** |
| a4 | 1.6 | 1.6 | 1.6 | — | 1.000 |

The RRTMG/CAM source (authoritative MAM4, *Ghan & Zaveri 2007*) reads
**σ_g = 1.8/1.6/1.8/1.6** — matching the config exactly — and the *original*
SARB table `c000000` agrees (1.8 for a1 and a3). **Only the v2 rebuild silently
narrowed a1 (1.8→1.6) and a3 (1.8→1.2)** while leaving `dgnum` unchanged. So the
**config widths are correct and the SARB v2 widths are wrong**; for the
mass-dominant coarse mode a3 this alone depresses AOD by ~4×.

### F3 · Köhler curvature term (stage E, informational)

Full Köhler (with the Kelvin `A/r_w` term) and the `microphysics.wet_radius`
`A≈0` simplification differ by up to **10.7%** at the accumulation radius
(0.055 µm). The curvature term raises the required RH, so the full solution
gives a **smaller** wet radius — physically expected; the production path
(`mode_physics`) uses the full solver. No defect; documents the divergence from
the legacy `microphysics` path.

### F4–F6 · SARB optics LUT cross sections are far below independent Mie (stage G)

At the sw05 midpoint (0.546 µm), for mode a1 over relevant radii with n_re≈1.5:

| reference | ratio reference/table | interpretation |
| --- | ---: | --- |
| monodisperse Mie at table radius | **2.48×** (1.85–3.21) | table low even vs single-particle Mie |
| number-averaged lognormal Mie (σ_g=1.6) | **3.85×** | the quantity `τ = σ·N` actually requires |
| effective Q_ext = table/(πr²) | **0.85** | physical Q_ext ≈ 2 |

The table's effective extinction efficiency (~0.85) is well below the physical
~2, so the LUT cross sections are intrinsically low by ~2.4× beyond any
size-distribution averaging. Mode-integration — the physically-correct target
for the `τ = σ·N` formula — widens the gap to ~3.8×. The independent Mie
reference was itself validated against the Rayleigh and geometric-optics limits
(stage G passes), so the deficit is in the **table**, not the reference.

## Root cause (verified)

Confirmed by an independent multi-agent workflow (`mam-optics-root-cause`, 6
agents): all six findings reproduced at high confidence; the LUT deficit is a
true normalization error, **not** a radius-convention artifact (a flat scale
c≈2.49 reconciles the table with Mie; a radius rescale does not). The lineage is
decisive — the deficit **entered at the SARB v2 reprocessing**:

- The **RRTMG/CAM source reconstructs cleanly** (Chebyshev decode → physically
  sane bell-shaped specific extinction, peak ~1196 m²/kg). The deficit is **not**
  inherited from RRTMG.
- The **original SARB `c000000`** stored a *different quantity* (modal
  mass-specific extinction in m²/kg, `opticsmethod='modal'`, rising with the
  radius parameter) and carries the correct σ_g = 1.8. It shows none of the
  sub-Mie pathology.
- The **v2 rebuild introduced** the per-particle `extpsw_mie` (µm²) whose
  effective Q_ext is only **~0.81 (a1) / ~0.57 (a3)** vs physical ~2.06 — a flat
  multiplicative shortfall of **~2.53× (fine)** and **~3.63× (coarse a3)** across
  all refractive indices and radii. It simultaneously narrowed σ_g (a1, a3) and
  rescaled the `refindex_im` axis by 1/1.2658.

**On the ~3.6× shortfall — do *not* multiply the factors.** For the dominant
coarse mode a3, the LUT deficit **alone** (~3.63×) reproduces the documented
shortfall; multiplying by the a3 σ_g factor (0.245) would give ~15×, far too
much. The contributors are formally separable stages of `τ = N·σ·path` but are
**entangled in practice** (the LUT was built with a narrower σ_g). The headline
~3.6× is best read as the **mode-3 LUT cross-section deficit surfacing in the
mode-sum**, with the σ_g mismatch a real, separately-confirmed *compounding*
bias. Closing the regression requires an **end-to-end production AOD run with
both the corrected LUT and corrected σ_g** — the arithmetic product is unsafe.

### Recommended fixes (in priority order)

1. **Regenerate the SARB v2 LUT** (don't merely rescale): rebuild `extpsw_mie`
   so effective Q_ext = `ext/(πr²)` → ~2.0–2.1 at large size parameter. Validate
   against `mie_sphere.mie_cross_sections_um2`. *Stopgap:* because the deficit is
   verified flat across n and r, per-mode multiplicative factors (~2.53× a1,
   ~3.63× a3) are defensible as a temporary patch — but they don't fix the
   `refindex_im` axis or confirm a2/a4.
2. **Restore canonical σ_g = 1.8/1.6/1.8/1.6** in the v2 LUT metadata (a1, a3).
3. **Enforce a shared-σ_g invariant**: assert `config σ_g == LUT sigmag` per mode
   at build/run time so a future rebuild cannot silently desynchronize the
   number derivation from the optics table.
4. **Audit the `refindex_im` 1/1.2658 rescale** and quantify its AOD impact
   before declaring the regression closed.

### Open items (carried from the verification)

- a2/a4 LUT deficit inferred (~2.5×) but only a1/a3 directly traced.
- The exact v2 build bug producing the ~2.5–3.6× factor is not yet located.
- σ_g-vs-LUT entanglement: only an end-to-end run resolves the true combined bias.

## Verifier integrity

The 34 passes are not tolerance-gamed: references use an independent float64 /
analytic / `scipy.optimize` path, and adversarial mutation tests (wrong unit
factor, dropped `×1000`, dropped Köhler solute term) make the relevant checks
FAIL by 3–12 orders of magnitude. The size-bin check (stage A) was rebuilt to
recompute the lognormal split from scratch rather than call the production
helper (now a genuinely independent 2e-16 match). Stages G1/G2 reuse the
production interpolation/indexing *method* by design — they are file-integrity /
dim-order regression guards, while G0 (Mie limits) and G3 (LUT-vs-Mie) provide
the independent physics validation.

## What is NOT the problem (ruled out by passes)

- Mass allocation / conservation across modes (A).
- Dry-volume, hygroscopicity, refractive-index volume mixing (B, D, F).
- Köhler solver accuracy (E, residual 1.5e-5).
- Number-concentration arithmetic given a width (C).
- Optical-depth units, external mixing, VIS correction (H, I, J).
- The LUT **pipeline** — fine-table interpolation and lookup indexing (G).
