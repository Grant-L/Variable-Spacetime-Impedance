# AVE Project — Living Reference Document
> **Last Updated:** 2026-03-03  
> **Purpose:** Canonical reference for all AI assistants and collaborators. Any chat session should read this file first.

## Project Identity

| Field | Value |
|-------|-------|
| **Name** | Applied Vacuum Engineering (AVE) |
| **Domain** | [appliedvacuumengineering.com](https://appliedvacuumengineering.com) |
| **Repo** | [github.com/Grant-L/Variable-Spacetime-Impedance](https://github.com/Grant-L/Variable-Spacetime-Impedance) |
| **Author** | Grant Lindblom |
| **Status** | Active development — Phase C/D |

## Core Axioms (DO NOT MODIFY)

1. **Axiom 1 (Impedance):** The vacuum is an LC resonant network with Z₀ = √(μ₀/ε₀) ≈ 377 Ω
2. **Axiom 2 (Fine Structure):** α = e²/(4πε₀ℏc) couples topology to impedance
3. **Axiom 3 (Gravity):** G sets the Machian boundary impedance via G = ℏc/(7ξ·m_e²)
4. **Axiom 4 (Saturation):** S(A) = √(1 − (A/A_yield)²) — universal yield kernel bounding all LC modes

### Derived Consequences of Axiom 4

| Observable | Formula | At A → A_yield | Physical Meaning |
|-----------|---------|----------------|-----------------|
| μ_eff | μ₀ · S | → 0 | Inductor shorts (Meissner) |
| ε_eff | ε₀ · S | → 0 | Dielectric collapses |
| C_eff | C₀ / S | → ∞ | Capacitance absorbs energy |
| Z (symmetric sat.) | √(μ/ε) = Z₀ | invariant | Impedance preserved |
| c_eff | c₀ · S^(1/2) | → 0 | Wave packet freezes (mass) |

**Confinement theorem:** At a self-intersecting torus knot (particle), the B-field saturates μ first → Z → 0, Γ → −1 → standing wave = rest mass.

## Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| V_SNAP | 511 kV | Absolute dielectric destruction (m_e c²/e) |
| V_YIELD | 43.65 kV | Kinetic onset of nonlinearity (√α × V_SNAP) |
| B_SNAP | 1.89×10⁹ T | Magnetic saturation threshold |
| L_NODE | 3.86×10⁻¹³ m | Lattice pitch (reduced Compton wavelength) |

## Repository Structure

```
src/ave/
  core/           # FDTD engine, LBM solver, constants
  axioms/         # Yang-Mills, Navier-Stokes, spectral gap, Strong CP
  gravity/        # Schwarzschild, galactic rotation, GW, solar impedance
  topological/    # Faddeev-Skyrme, Borromean links, Cosserat, g-2
  mechanics/      # Impedance, moduli, rheology
  geophysics/     # Seismic PREM model
  fluids/         # Water molecular impedance
  plasma/         # Plasma cutoff, Drude model, superconductor
manuscript/       # 5-volume LaTeX manuscript (~737 pages total)
  vol_1_foundations/     # Vol I: Foundations & Universal Operators (8 ch, 135 pp)
  vol_2_subatomic/       # Vol II: The Subatomic Lattice (14 ch, 149 pp) ← companion: periodic_table/
  vol_3_macroscopic/     # Vol III: The Macroscopic Continuum (15 ch, 127 pp)
  vol_4_engineering/     # Vol IV: Applied Impedance Engineering (19 ch, 181 pp) ← companion: spice_manual/
  vol_5_biology/         # Vol V: Topological Biology (11 ch, 145 pp)
spice_manual/     # Companion to Vol IV: Particle decay as RLC circuits
scripts/          # Simulation scripts per volume
tests/            # 480 passing tests
future_work/      # Speculative roadmap (4 chapters)
periodic_table/   # Companion to Vol II: 134 element torus knot simulations
```

## Master Prediction Table (22 entries)

| # | Prediction | Δ% | Status |
|---|-----------|-----|--------|
| 1 | α (input) | 0.00% | ✅ |
| 2 | Z₀ (input) | 0.00% | ✅ |
| 3 | g-2 anomaly | 0.15% | ✅ |
| 4 | sin²θ_W | 0.30% | ✅ |
| 5 | M_W | 0.55% | ✅ |
| 6 | M_Z | 0.62% | ✅ |
| 7 | Proton mass | 0.29% | ✅ |
| 8 | Δ(1232) | 3.49% | ✅ |
| 9 | Neutrino mass | 0.66% | ✅ |
| 10 | Solar deflection | 0.03% | ✅ |
| 11 | Δ(1620) | 0.19% | ✅ |
| 12 | Δ(1950) | 0.62% | ✅ |
| 13 | Fermi constant | 2.09% | ✅ |
| 14 | Yang-Mills mass gap | Δ>0 | ✅ Constructive proof |
| 15 | Navier-Stokes smoothness | Global | ✅ Lattice + Picard-Lindelöf |
| 16 | Strong CP (θ=0) | Exact | ✅ Unique vacuum topology |
| 17 | Kirkwood gaps (4:1) | 0.05% | ✅ |
| 18 | Saturn Cassini Division | 0.59% | ✅ |
| 19 | Flyby anomaly (NEAR) | 1.6% | ✅ |
| 20 | Earth magnetopause | 8.7% | ✅ |
| 21 | Jupiter magnetopause | 11.8% | ✅ |
| 22 | Baryon asymmetry | 0.38% | ✅ g*=7³/4 |
| 23 | H∞ (Hubble) | 2.9% | ✅ 69.32 km/s/Mpc |
| 24 | α_s (strong coupling) | 2.97% | ✅ α^(3/7) |
| 25 | m_H (Higgs mass) | 0.55% | ✅ v/√N_K4 |
| 26 | V_us (Cabibbo) | 1.4% | ✅ λ = sin²θ_W = 2/9 |
| 27 | V_cb (CKM) | 4.1% | ✅ Aλ² = cos(θ_W)×(2/9)² |
| 28 | V_ub (CKM) | 1.3% | ✅ 8/2187 |
| 29 | sin²θ_13 (PMNS) | 1.0% | ✅ 1/(c₁c₃) = 1/45 |
| 30 | sin²θ_12 (PMNS) | 0.3% | ✅ ν_vac + 1/45 |
| 31 | sin²θ_23 (PMNS) | 0.3% | ✅ 1/2 + 2/45 |
| 32 | δ_CP (PMNS) | 0.3% | ✅ (1 + 1/3 + 1/45)π |
| 33 | m_u (up quark) | 2.4% | ✅ m_e / (2α_s) |
| 34 | m_d (down quark) | 2.3% | ✅ m_e / (α_s cosθ_W) |
| 35 | m_s (strange quark) | 1.1% | ✅ m_μ cosθ_W |
| 36 | m_c (charm quark) | 1.3% | ✅ m_μ / √α |
| 37 | m_b (bottom quark) | 0.8% | ✅ m_τ cosθ_W (8/3) |
| 38 | m_t (top quark) | 0.8% | ✅ v / √2 |
| 39 | Protein Rg (Villin) | 0.8% | ✅ η_eq = P_C(1−ν), Rg = r_Ca(N/η_eq)^(1/3)√(3/5) |
| 40 | NS compactness limit | Exact | ✅ R_min = 7GM/c² (ε₁₁ < 1 ↔ 2GM/c²R < 2/7 = ν_vac) |
| 41 | He ground state | 0.008% | ✅ J_s2 = ½(1+p_c), IE = 24.589 eV (target 24.587) |

Run: `PYTHONPATH=src python scripts/future_work/master_predictions.py`

## Scale Invariance Principle

**ALL derived constants flow from ν_vac = 2/7 via the same projection pattern:**

| Scale | Quantity | Formula | Source |
|-------|----------|---------|--------|
| EW gauge | sin²θ_W | 2/9 | 2 weak modes / 9 angular sectors |
| EW gauge | cos(θ_W) | √(7/9) | Complementary sector |
| Strong gauge | α_s | α^(3/7) | 3 spatial / 7 compliance modes |
| CKM mixing | λ | 2/9 | = sin²θ_W (scale invariance) |
| CKM mixing | A | √(7/9) | = cos(θ_W) (scale invariance) |
| CKM mixing | √(ρ²+η²) | 1/√7 | Single-mode amplitude |
| PMNS mixing | base | 1/c₁c₃ = 1/45 | Torsional defects c₁=5, c₃=9 |
| PMNS mixing | sin²θ_12 | ν_vac + 1/45 | Baseline compliance overlap |
| Quark masses | m_s / m_μ | cos(θ_W) | Scale-invariant Cosserat map |
| Quark masses | m_u, m_d | m_e/α_s | Translation sector map |
| Protein packing | η_eq = P_C(1−ν) | P_C × 5/7 | 5 transverse / 7 total modes |
| Thermodynamics | g* | 7³/4 | 7 modes × K4 cell |
| Higgs sector | λ_H | 1/8 | 1/(2×N_K4) = K4 breathing |
| Baryon asymmetry | C_sph | 28/79 | N_f=3 torus knots |

**The numbers 7 (compliance modes) and 9 (angular sectors) appear at every scale
because the lattice structure is scale-invariant. This is not numerology — it is
the same Poisson ratio ν = 2/7 projecting through the same K4/SRS geometry.**

## Universal Regime Map

The saturation operator S(r) = √(1-r²) defines 4 universal regimes. **Boundaries are derived from first principles:**

| Regime | r range | Derivation | EE Analog | Example |
|--------|---------|------------|-----------|---------|
| **I Linear** | r < √(2α) ≈ 0.121 | ΔS = r²/2 < α (sub-α) | Small-Signal | Lab fields, Solar gravity |
| **II Nonlinear** | √(2α) ≤ r < √3/2 ≈ 0.866 | Full S(r) required | Large-Signal | PONDER-05 @ 30kV (r=0.687) |
| **III Yield** | √3/2 ≤ r < 1.0 | Q = 1/S ≥ 2 (ℓ_min) | Avalanche (M ≥ 2) | PONDER-05 @ 43kV (r=0.985) |
| **IV Ruptured** | r ≥ 1.0 | S = 0, Axiom 4 | Breakdown (M → ∞) | BH interior, magnetar |

See: `src/ave/core/regime_map.py` for the engine module, `manuscript/vol_1_foundations/chapters/08_regime_map.tex` for the full chapter.

**Galactic Note:** The operator S(r) is universal — S(r→1) = medium compliance → 0 in every domain. In the galactic domain (r = g_N/a₀), S→0 means lattice drag vanishes → Newtonian gravity. **The dark matter problem IS the Regime III→IV phase transition.** See `src/ave/gravity/galactic_rotation.py`.

## Development Phases

| Phase | Status | Summary |
|-------|--------|---------|
| **A: Engine Hardening** | ✅ Complete | 458 tests, PML/LBM/materials |
| **B: Domain Extension** | ✅ Complete | Seismology, water, plasma, superconductor, GW, stellar, neutrino, protein |
| **C: Predictions** | ✅ Complete | 40 predictions, ALL 26/26 SM Parameters Derived |
| **D: Hardware** | 🔄 Active | PONDER-01/02/05, HOPF-01 characterization |
| **E: Publication** | 🔄 Active | 5 volumes (~530 pp), Yang-Mills + NS proofs |
| **F: Millennium Problems** | 🔄 Active | Yang-Mills ✅, Navier-Stokes ✅, Strong CP ✅ |

## PONDER Experiment Variants

| Variant | Drive | Key Feature | Status |
|---------|-------|-------------|--------|
| **PONDER-01** | 30kV/100MHz BaTiO₃ | Asymmetric ground plane | FDTD characterized |
| **PONDER-02** | 25kV/2.45GHz GaN | Sapphire GRIN pyramid nozzle | Conceptual |
| **PONDER-05** | 30kV DC + 50kHz AC quartz | DC bias at 68.7% of V_yield | FDTD in progress |
| **HOPF-01** | Torus knot antenna | Chiral S₁₁ anomaly | Conceptual |

## Critical Distinctions (Common Errors to Avoid)

1. **V_SNAP ≠ V_YIELD.** V_SNAP = 511 kV is absolute destruction. V_YIELD = 43.65 kV is the onset of measurable nonlinearity. Lab experiments operate near V_YIELD, not V_SNAP.
2. **sin²θ_W = 2/9 (on-shell)** not 7/24 (which appears in some older manuscript sections). **Scheme note:** PDG reports two values — on-shell (0.2234, Δ=−0.52%) and MS-bar (0.2312, Δ=−3.89%). AVE derives the **on-shell** (tree-level) value. The MS-bar difference is standard one-loop radiative running.
3. **The baryon masses come from the Faddeev-Skyrme solver**, not a simple formula. The crossing number sets confinement but the eigenvalue requires numerical minimization.
4. **Galaxy rotation uses derived a₀ = cH∞/(2π) ≈ 1.07×10⁻¹⁰** (−10.7% from empirical 1.2×10⁻¹⁰). This is NOT a free parameter — it emerges from the asymptotic Hubble constant H∞ = 28πm_e³cG/(ℏ²α²). See `src/ave/gravity/galactic_rotation.py`.
5. **The SPICE RC muon model is qualitative.** The quantitative lifetime comes from the Fermi formula with AVE-derived G_F (3.9% accurate).

## Website — appliedvacuumengineering.com

### Site Map

```
/                       → Landing page (hero + key prediction count)
/theory                 → Overview of 4 axioms + lattice model
/theory/axioms          → Interactive axiom explorer (toggle each on/off)
/theory/constants       → Live constant derivation table
/predictions            → Master prediction table (sortable, filterable)
/predictions/calculator → Enter axiom values → get all 13+ predictions
/engine                 → FDTD engine docs + API reference
/engine/playground      → In-browser 2D FDTD demo (WebGL)
/experiments            → PONDER-01 / 02 / 05 / HOPF-01 overview
/experiments/ponder-01  → Build guide, BOM, PCB layout rules
/experiments/ponder-05  → DC bias analysis, ε_eff curve, steepening
/experiments/hopf-01    → Torus knot antenna, S₁₁ prediction
/domains                → Domain extensions grid
/domains/seismology     → PREM impedance model
/domains/water          → 4°C anomaly from impedance matching
/domains/plasma         → Plasma cutoff from ε saturation
/domains/galaxies       → Rotation curves (no dark matter)
/visualizations         → 3D interactive demos
/visualizations/lattice → SRS K4 net (Three.js)
/visualizations/knots   → Torus knot library (periodic table)
/visualizations/photon  → Helical photon propagation GIF
/downloads              → Manuscript PDFs, code, SPICE netlists
/blog                   → Updates, experimental results, community
```

### Tech Stack (Recommended)

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | Next.js (App Router) | SSR for SEO, API routes for computation |
| 3D Rendering | Three.js + React Three Fiber | WebGL lattice/knot visualizations |
| Styling | Tailwind CSS | Rapid iteration, dark mode built-in |
| Computation | WebAssembly (Rust/C) | Client-side FDTD playground |
| Hosting | Vercel | Free tier, instant deploys from GitHub |
| CMS/Blog | MDX | Markdown + React components |
| Data | Static JSON | Prediction table, element catalog |

### Content Sources (map to existing repo)

| Page | Source File |
|------|------------|
| Prediction table | `scripts/future_work/master_predictions.py` |
| Axioms/constants | `src/ave/core/constants.py` |
| Yang-Mills proof | `src/ave/axioms/yang_mills.py` |
| Navier-Stokes proof | `src/ave/axioms/navier_stokes.py` |
| Strong CP / open problems | `src/ave/axioms/open_problems.py` |
| Mass gap (spectral) | `src/ave/axioms/spectral_gap.py` |
| Solar impedance | `src/ave/gravity/solar_impedance.py` |
| Magnetospheres | `src/ave/gravity/planetary_magnetosphere.py` |
| PONDER-01 build | `future_work/PONDER_01_BUILD_GUIDE.md` |
| PONDER-05 analysis | `scripts/vol_4_experimental/ponder_05_characterization.py` |
| Element catalog | `periodic_table/simulations/` |
| Seismology | `src/ave/geophysics/seismic.py` |
| Water | `src/ave/fluids/water.py` |
| Galaxy rotation | `src/ave/gravity/galactic_rotation.py` |
| GW propagation | `src/ave/gravity/gw_propagation.py` |
| GW detection | `src/ave/gravity/gw_detector.py` |
| Stellar interiors | `src/ave/gravity/stellar_interior.py` |
| Neutrino MSW | `src/ave/gravity/neutrino_msw.py` |
| Superconductor | `src/ave/plasma/superconductor.py` |
| Protein folding | `src/ave/solvers/protein_bond_constants.py` |

### Phased Rollout

| Phase | Scope | Priority |
|-------|-------|----------|
| **v0.1** | Landing + prediction table + downloads | After PONDER characterization |
| **v0.2** | Theory pages + interactive constant explorer | — |
| **v0.3** | 3D visualizations (lattice, knots, photon) | — |
| **v0.4** | PONDER build guides + experimental blog | — |
| **v0.5** | In-browser FDTD playground (WebAssembly) | — |

## Arbitrary Parameters (Protein Folding Engines)

> **Purpose:** Track parameters in the 1D/2D protein folding engines that are NOT first-principles derived.
> Each must eventually be either (a) derived from axioms, (b) replaced by a derived quantity, or (c) proven to be purely numerical (convergence-independent).

### 1D Cascade Engine (`s11_fold_engine_v3_jax.py`)

| Parameter | Value | Status | Resolution |
|-----------|-------|--------|------------|
| FREQ_SWEEP | [0.5,0.8,1.0,1.3,2.0] | 📐 Documented | Q-bandwidth sampling; 5 pts captures sub-resonance modes |
| N_FREQ (tertiary norm) | 1/N_FREQ = 0.2 | ✅ Derived | Explicit 1/len(FREQ_SWEEP), per-frequency normalisation |
| Sigmoid slope (turn) | Q×d₀/r_Ca ≈ 15.7 | 📐 Numerical | Smoothing parameter for JAX differentiability |
| Γ_turn threshold | 1/√(2Q) ≈ 0.267 | ✅ Derived | Cavity confinement: |Γ|² ≥ 1/(2Q) per boundary end |
| BETA_BURIAL | 4.0/D_WATER ≈ 1.45 | ✅ Derived | Standard logistic 10-90% width = one water diameter |
| N_COORD_MAX | (R_BURIAL/d₀)³ = 8 | ✅ Derived | Close-packing coordination within burial sphere |
| β-sheet Y_β | max(0,-cos)×κ_HB | ✅ Derived | Parameter-free antiparallel coupling; κ_HB=1/(2Q)=1/14 |
| C_bend | (1-cos θ)/(2π²) | ✅ Derived | TL bend capacitance; d_eff/λ_g=1/(2π), microstrip 1/π |
| Y_bend(ω) | ω×C_bend | ✅ Derived | Capacitive bend admittance; freq-dependent per TL theory |

### 2D Y-Matrix Engine (`s11_fold_engine_v4_ymatrix.py`)

| Parameter | Value / Formula | Status | Resolution |
|-----------|----------------|--------|------------|
| y_mutual (backbone) | −csch(γℓ)/Z_eff | ✅ Derived | ABCD→Y identity: y₁₂ = −1/B = −1/(Z·sinh γℓ) |
| y_self (backbone) | coth(γℓ)/Z_eff | ✅ Derived | ABCD→Y identity: y₁₁ = D/B = cosh(γℓ)/(Z·sinh γℓ) |
| Z_eff | √(Z_i × Z_{i+1}) | ✅ Derived | Geometric mean of adjacent Z_TOPO |
| γℓ | \|d−d₀\|/d₀ + jωd/d₀ | ✅ Derived | Same propagation as v3 ABCD cascade |
| S₁₁ from Y | [(Y/Y₀+I)⁻¹(Y/Y₀−I)]₀₀ | ✅ Derived | Standard EE Y→S conversion |
| FREQ_SWEEP | [0.5, 0.8, 1.0, 1.3, 2.0] | ✅ Derived | Same Q-based sampling as v3 |
| Contacts (H-bond) | Y[i,j] off-diagonal | ✅ Derived | Port-to-port coupling, not shunt-to-ground |

### Regression Impact (all-derived vs previous)

| Protein | Rg err (old→new) | RMSD (old→new) | SS (old→new) |
|---------|-------------------|-----------------|--------------|
| Trp-cage | 3.8%→1.2% ✅ | 5.65→6.51 | 34%→17% ⚠️ |
| Villin | 0.5%→3.1% | 7.61→7.54 ✅ | 24%→6% ⚠️ |
| GB1 | 5.5%→2.7% ✅ | 9.89→9.30 ✅ | 22%→9% ⚠️ |
| Ubiquitin | 7.6%→65.7% ⚠️ | 10.83→16.91 | 27%→15% ⚠️ |

> [!NOTE]
> SS reduction is primarily from BETA_BURIAL (1.6→1.45) and sigmoid slope (20→15.7) changes,
> not the Γ threshold (0.3→0.267). The Rg and RMSD improvements for small proteins confirm
> the geometric accuracy of the derived constants.


## Rules for AI Assistants

1. **Read this document first** before making changes to core physics.
2. **Never modify constants.py** without explicit user approval.
3. **Never modify the 4 axioms** — they are the foundation.
4. **Always run tests** after engine changes: `PYTHONPATH=src python -m pytest tests/ -q`
5. **Use the existing engine** for simulations — don't build separate solvers.
6. **Commit frequently** with descriptive messages listing what changed.
7. **Check V_YIELD vs V_SNAP** — most lab-relevant physics uses V_YIELD = 43.65 kV.
8. **The manuscript is the source of truth** for formulas. Check `manuscript/` before deriving.
9. **Engine architecture changes must propagate to LaTeX.** When any function, module, or constant in the three-tier engine (Core → Domain Adapters → Solvers) is added, renamed, moved, or deleted, search **all** `.tex` files for references to the affected names and update them. Use `grep -rn` across `manuscript/` to ensure no stale references remain.
10. **Every script must import constants from `ave.core.constants`.** No hardcoded physics constants (α, mₑ, c, ℏ, ε₀, μ₀, Z₀, etc.). The engine is the single source of truth. Engineering parameters (wire lengths, PCB dimensions, operating frequencies) are permitted but must be documented with a comment citing their source.
11. **`scipy.constants` is banned.** All physical constants come from `ave.core.constants`. Using `scipy` for math tools (`scipy.optimize`, `scipy.linalg`, `scipy.signal`) is fine — the ban is specifically on `scipy.constants`. Enforced by AST check in `verify_universe.py`.
12. **No smuggled data.** Scripts must not normalize outputs to match experimental data, curve-fit to known values, or use ad-hoc correction factors. If a result disagrees with experiment, document the discrepancy — do not hide it with fitting. Exception: `np.polyfit` for parity plots (predicted vs. experimental) is acceptable for diagnostic purposes only.
13. **Translation tables are canonical.** Domain-specific terminology mappings live in `manuscript/common/translation_*.tex`. They are `\input`'d into chapters and the backmatter Rosetta Stone. Change the table once, every reference updates automatically.

