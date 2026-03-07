---
description: Audit all equations and numeric values in LaTeX files for consistency with the physics engine
---

# Mathematical Consistency Audit Workflow

Verify every equation, numeric value, and derivation chain in `.tex` files against the canonical physics engine (`src/ave/core/constants.py` and `src/ave/axioms/scale_invariant.py`).

> **Before starting:** Read `LIVING_REFERENCE.md` and `src/ave/core/constants.py` in full.
> These are the two canonical sources of truth for all physics in this repo.

## Scope

All `.tex` files in `manuscript/`, `future_work/`, `spice_manual/`, `periodic_table/`, and `standard_model/`.

## Checks

### 1. Known Stale Values (High Priority — Historical Bugs)

These specific errors have occurred before. Search for them first:

- [ ] **V_YIELD misquotes**: Search for `60 kV`, `60.3 kV`, `73 kV`, or any voltage limit other than the canonical `V_YIELD ≈ 43.65 kV` and `V_SNAP = 511 kV`. Note: 60.3 kV has appeared in `14_applied_fusion.tex` as a *derived* tokamak collision voltage — verify whether each occurrence is a stale constant or a legitimate derived value with a clear derivation shown.
- [ ] **sin²θ_W errors**: Search for `7/24`, `0.231`, `0.2315`, or any value other than `2/9 ≈ 0.2222`. The canonical value is `sin²θ_W = 2/9` (on-shell).
- [ ] **ν_vac errors**: Verify all occurrences use `2/7` — not `1/3`, `1/4`, or other fractions.
- [ ] **Stale Planck-scale pitch**: `variables.tex` line 13 lists lattice pitch as `1.62 × 10⁻³⁵ m` (Planck length). The canonical value is `L_NODE = ℏ/(m_e·c) ≈ 3.86 × 10⁻¹³ m` (reduced Compton wavelength). Flag any Planck-scale lattice pitch.

### 2. Constant Cross-Reference

For every numeric constant appearing in a `.tex` file, verify it matches `constants.py`:

| Constant | Canonical Value | `constants.py` Variable |
|----------|----------------|------------------------|
| V_SNAP | 511 kV | `V_SNAP` |
| V_YIELD | ≈ 43.65 kV | `V_YIELD` |
| B_SNAP | ≈ 1.89 × 10⁹ T | `B_SNAP` |
| L_NODE | ≈ 3.86 × 10⁻¹³ m | `L_NODE` |
| Z₀ | ≈ 376.73 Ω | `Z_0` |
| α | ≈ 7.297 × 10⁻³ | `ALPHA` |
| P_C | 8πα ≈ 0.1834 | `P_C` |
| η_eq | P_C × 5/7 ≈ 0.1310 | `ETA_EQ` |
| T_EM | ≈ 0.212 N | `T_EM` |
| sin²θ_W | 2/9 ≈ 0.2222 | `SIN2_THETA_W` |
| cos(θ_W) | √(7/9) | `A_CKM` |
| α_s | α^(3/7) ≈ 0.1214 | `ALPHA_S` |
| κ_FS (cold) | 8π ≈ 25.133 | `KAPPA_FS_COLD` |
| δ_th | 1/(14π²) ≈ 0.00721 | `DELTA_THERMAL` |
| M_W | ≈ 80,940 MeV | `M_W_MEV` |
| M_Z | ≈ 91,760 MeV | `M_Z_MEV` |
| M_H | ≈ 124,417 MeV | `M_HIGGS_MEV` |
| H∞ | ≈ 69.32 km/s/Mpc | `H_INFINITY` |
| K_MUTUAL | ≈ 11.337 MeV·fm | `K_MUTUAL` |
| ν_vac | 2/7 | `NU_VAC` |
| N_K4 | 4 | `N_K4` |
| λ_H | 1/8 | `LAMBDA_HIGGS` |
| All CKM elements | λ=2/9, A=√(7/9), √(ρ²+η²)=1/√7 | `V_US`, `V_CB`, `V_UB` |
| All PMNS angles | sin²θ₁₃=1/45, etc. | `SIN2_THETA_13`, etc. |
| Proton charge radius | ≈ 0.8412 fm | `D_PROTON` |

### 3. Equation Structure Verification

For each `\begin{equation}` / `\begin{align}` environment:

- [ ] The formula matches the corresponding expression in `constants.py` (check algebra, not just numbers)
- [ ] Derivation steps are pedantically shown — no jumps from axiom to result without intermediate steps
- [ ] Axiom traceability: each equation should cite which axiom(s) it derives from
- [ ] No undeclared symbols — every variable in an equation is defined somewhere in the same chapter or in `variables.tex`

### 4. Prediction Table Consistency

- [ ] All prediction values in `.tex` files match the output of `scripts/future_work/master_predictions.py`
- [ ] Percentage errors (Δ%) match between LaTeX and the script
- [ ] The total count of predictions matches `LIVING_REFERENCE.md` (currently 39)

### 5. Magic Number Detection

- [ ] Flag any numeric literal in a `.tex` equation that:
  - Does not appear in `constants.py`
  - Is not a trivially obvious number (0, 1, 2, π, etc.)
  - Does not have a derivation shown
- [ ] Any "magic number" found should be documented per user rules: either derived from axioms or flagged for the numerology appendix

### 6. Cross-Chapter Consistency

- [ ] The same physical quantity uses the same symbol across all books (e.g., `ℓ_node` vs `ℓ_p` vs `L_NODE`)
- [ ] Axiom statements are identical across all books (verify via `common_equations/` usage)
- [ ] Chapter numbering is consistent across books (no duplicate chapter numbers)

### 7. Derivation Chain Completeness

- [ ] Each book's derivation flow goes Axiom → Intermediate → Result with no gaps
- [ ] The full derivation chain in `backmatter/02_full_derivation_chain.tex` covers all 39 predictions
- [ ] No circular dependencies in derivation chains

## Output

Produce a structured report:
1. **VIOLATION** — Value contradicts `constants.py` (must fix)
2. **STALE** — Likely outdated value from earlier version of theory
3. **MISSING** — Derivation step or axiom citation not shown
4. **MAGIC** — Unexplained numeric literal (needs derivation or numerology documentation)
5. **OK** — Spot-checked and consistent

For each finding, cite file, line number, the incorrect value, and the canonical value from `constants.py`.
