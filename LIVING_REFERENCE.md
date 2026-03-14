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
   - **α invariance**: Under Symmetric Gravity, α = e²/(4πε₀ℏc₀) is exactly invariant because ε_local and c_local carry the same n·S factor that cancels. Multi-species Δα/α = 0.
   - **Lattice decomposition**: n_temporal = 1 + (2/7)ε₁₁ (clock rate, redshift); n_spatial = (9/7)ε₁₁ (light deflection). Axiom 3's n(r) = 1+2GM/(c²r) is the temporal component only.
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

### How to Apply AVE to a New Physical System

When mapping a new phenomenon to the AVE framework, follow these steps in order:

**Step 1 — Identify the LC Analogs.**
Every physical system has an inductive (μ) and capacitive (ε) degree of freedom. Map the system's variables to these. The impedance Z = √(μ/ε) is the master variable.

| Domain | μ-analog (inertia) | ε-analog (compliance) | Z expression |
|--------|-------------------|----------------------|-------------|
| Vacuum | μ₀ | ε₀ | Z₀ = 377 Ω |
| Seismic | 1/G_shear | 1/K_bulk | ρ·Vₚ (Rayl) |
| Protein | backbone τ | dipole C | S₁₁ impedance |
| Fluid | ρ (density) | 1/K (compressibility) | ρ·c_sound |

**Step 2 — Compute the Local Strain and Determine the Regime.**
Find the relevant amplitude A and its yield limit A_yield. Compute the strain ratio r = A/A_yield.

| Regime | Condition | Physics |
|--------|-----------|---------|
| I | r ≪ 1 | Linear Maxwell; standard physics recovered |
| II | r → 1 | Saturation onset; nonlinear corrections |
| III | r = 1 | Phase transition; topology rupture |
| IV | r > 1 | Ruptured; interior melted |

**Step 3 — Apply the Universal Operators.**
Use the same operators at every scale:
- `S(A) = √(1 − (A/A_yield)²)` — saturation factor
- `Γ = (Z₂ − Z₁)/(Z₂ + Z₁)` — reflection coefficient
- `ω = ℓ·c/r_eff` — regime boundary eigenvalue
- `Q = ℓ` — quality factor

**Step 4 — Check Symmetry Cancellations.**
Many observables are *ratios* of constitutive parameters. Under Symmetric Gravity / Symmetric Saturation (ε and μ scale by the same factor), these ratios can be invariant:
- **Z₀ = √(μ/ε)** is invariant if both scale by the same n·S
- **α = e²/(4πε·ℏ·c)** is invariant because ε and 1/c cancel when both carry the same factor
- Clock *ratios* between species are invariant under symmetric saturation

⚠️ **This is the most common source of error.** Always check whether your predicted observable survives the symmetry cancellation before claiming a new signal exists.

**Step 5 — Compute the Numerical Prediction from Engine Constants.**
Use `from ave.core.constants import ...` — never hardcode. The engine is the single source of truth.

**Step 6 — Determine Testability.**
Compute the predicted signal magnitude and compare to the best available measurement precision. If `signal / precision < 1`, the prediction is currently unfalsifiable. Look for systems where the strain is larger (e.g., Sirius B instead of Earth surface) to find detectable regimes.

### Common Pitfalls (check your work against these)

| # | Pitfall | Wrong | Right |
|---|---------|-------|-------|
| 1 | **Dimensionless BH frequency** | ω·M_g (M_g = GM/c² in meters) | ω·M_geom (M_geom = GM/c³ in seconds) → ω·M ≈ 0.37 for ℓ=2 |
| 2 | **Overestimating saturation corrections** | ε₁₁²/2 applied as additive km/s | Multiply exact Schwarzschild z by 1/S; correction is z·ε₁₁²/2 ≈ 0.05 km/s for Sirius B |
| 3 | **Temporal vs spatial metric** | Using full lattice density n = 1+(11/7)ε₁₁ for redshift | Only temporal component: n_t = 1+(2/7)ε₁₁. Spatial part (9/7)ε₁₁ → light deflection only |
| 4 | **MOND drag at high g** | Assuming lattice drag applies at WD surface | S(g/a₀) = 0 when g >> a₀ = 10⁻¹⁰. WD surface g ~ 10⁶ → zero drag |
| 5 | **Claiming Δα from gravity** | Predicting multi-species clock shift | α is exactly invariant under Symmetric Gravity (ε and 1/c cancel). Δα/α = 0 always. |
| 6 | **Free-electron n_s for superconductors** | Using n_e as n_s in λ_L formula | n_s ≠ n_e for d-band metals (Nb). Back-compute n_s from measured λ_L, or κ will be wrong. |
| 7 | **Iterative SCF for Z ≥ 26** | Using SCF ionization_energy(Z) | SCF is Hartree-Fock, not AVE. Nuclear binding uses coupled resonator. |
| 8 | **QM Contamination in IE** | $IE = Z_{eff}^2 \times Ry / n^2$ | This is the Schrödinger/Bohr formula! Atomic energies must emerge exactly from the AVE 5-step eigenvalue method. |
| 9 | **Op4 Bypass** | $V_{ee} = J \times Z \times Ry$ | Ad-hoc energy formula — not from any operator. Electron-electron interaction MUST use Op4: $U = K/r_{12} \times (T^2 - \Gamma^2)$. J enters through the angular average of $r_{12}$, not as a prefactor. |
| 10 | **De Broglie ≠ Impedance** | $n(r) = \sqrt{KE/Ry}$ is "impedance" | This is the defect's dispersion, NOT the medium impedance $Z = \sqrt{\mu/\varepsilon}$. The lattice has $Z_0 = 377\,\Omega$ everywhere in Regime I. Name quantities correctly: $n_{dB}(r)$ = de Broglie refractive index, $Z_0$ = lattice impedance. |
| 11 | **QM Coulomb integrals** | `_J_1S_1S = 5/8`, `17/81`, `77/512` | These are ⟨ψ|1/r₁₂|ψ⟩ from hydrogen wavefunctions. The AVE J scalars (J_1s² = ½(1+p_c), etc.) derive from torus geometry + Op2 saturation at crossings. QM J ≠ AVE J. |

### Operator Compliance Checklist

**Before declaring ANY derivation complete, verify every row:**

| Op | Name | Formula | Question to ask | ✓/✗ |
|----|------|---------|-----------------|-----|
| 1 | Z (Impedance) | $Z = \sqrt{\mu/\varepsilon}$ | Is the medium impedance defined from constitutive properties? Not from energies? | |
| 2 | S (Saturation) | $S = \sqrt{1-(A/A_c)^2}$ | Is saturation applied where fields approach $A_c$? Is $p_c = 8\pi\alpha$ used? | |
| 3 | Γ (Reflection) | $\Gamma = (Z_2, - Z_1)/(Z_2 + Z_1)$ | Are all impedance boundaries handled by Γ? No ad-hoc "screening constants"? | |
| 4 | U (Pairwise) | $U = -K/r \times (T^2 - \Gamma^2)$ | Are ALL pairwise interactions computed from Op4? No hand-wavy energy formulas? | |
| 5 | Y→S | $[S] = (I+Y/Y_0)^{-1}(I-Y/Y_0)$ | For multiport: are S-parameters from Y-matrix, not assumed? | |
| 6 | λ_min | $\lambda_{min}(S^\dagger S) \to 0$ | Is the eigenvalue condition the S₁₁ dip? No Bohr formula? | |
| 7 | FFT | Spectral analysis | For periodic structures: is the mode structure from FFT, not assumed? | |
| 8 | Γ_pack | Packing reflection | For 3D assemblies: is packing fraction from $P_C(1-1/N)$? | |

**Red flags for QM contamination:**
- Using $E = Z_{eff}^2 Ry / n^2$ → Bohr formula (pitfall #8)
- Using $\sigma$ as a number subtracted from Z → σ-arithmetic
- Writing $V_{ee} = \text{(constant)} \times \text{(energy scale)}$ without deriving from Op4 → pitfall #9
- Calling lattice defect dispersion "impedance" → pitfall #10
- Using "wavefunction," "probability," "orbital" without translating to AVE

### IE Solver — Approach Audit Trail

The atomic-scale IE solver (`coupled_resonator.py`) has been investigated with 23 approaches. **Critical finding: ALL implementations ≤21 use the Bohr formula IE = Z_eff²Ry/n² (pitfall #8). Approach 22 solved same-shell. Approach 23 merges same-shell + cross-shell.**

| # | Approach | Status |
|---|----------|--------|
| 1–4 | SCF (various Hartree-Fock variants) | ❌ QM contamination |
| 6 | Coupled resonator (k=J_geo) | ❌ He -17%, Z≥3 negative IE |
| 8 | FOC/BEMF decomposition | ❌ Conceptual only |
| 9 | Impedance cascade (v5, active) | ❌ Uses Bohr formula at line 1585 |
| 12 | "Pure AVE 5-step" | ❌ Still returns Z_eff²Ry/n² |
| 13-16 | Point-particle, motor BEMF, variational, static stubs | ❌ Various failures |
| 17 | Coupled oscillator dynamical matrix | ❌ Nuclear pattern on atoms: 1900% error |
| 18 | 5-step eigenvalue + static σ-arithmetic | ❌ = Bohr formula |
| 19 | Radial TL with energy-dependent n(r,ξ) | ✅ H +0.3%, Li -3.6% (cross-shell only) |
| 20 | TL + σ step for same-shell J | ❌ Splits cavity incorrectly |
| 21 | Coupled LC circuit (He) | ⚠️ IE = 22.5 eV (-8.3%), architecture correct |
| 22 | Coupled LC + Hopf p_c/2 (same-shell) | ✅ He −0.9% (24.37 eV). Same-shell only. |
| 23 | **Merged: 22 (same-shell) + 19 (cross-shell)** | 🔧 In progress |

### QM Contamination in coupled_resonator.py (AUDIT FINDING)

**Line 1585 (active solver, v5)**:
```python
return z_eff**2 * _RY_EV / n_out**2  # Bohr formula!
```

**Lines 233-235 (J integrals)**:
```python
_J_1S_1S = 5.0 / 8.0     # QM ⟨1s|1/r₁₂|1s⟩
_J_1S_2X = 17.0 / 81.0   # QM ⟨1s|1/r₁₂|2s/2p⟩
_J_2X_2X = 77.0 / 512.0  # QM ⟨2s/2p|1/r₁₂|2s/2p⟩
```

These J integrals compute ⟨1/r₁₂⟩ using hydrogen atom wavefunctions (ρ₁ₛ = exp(-2Zr/a₀), ρ₂ₛ = (2-Zr)²exp(-Zr)) — **not from Op4 or the universal operators at all**.

The AVE J_1s² = ½(1+p_c) = 0.5917 (Ch.16 §Lattice Derivation of J_s2) is derived from Axiom 4 (p_c = 8πα). This is NOT used anywhere in the solver.

### Correct AVE Solver Architecture (Approach 22 + 23)

The atom is a COUPLED LC RESONATOR NETWORK (same architecture as nuclear binding):

1. **Single-electron binding** (Op3 + Op6): E₀ = Z²Ry/n² (bare nuclear charge, no Z_eff).

2. **Same-shell coupling** (Op4 + Op2): k_pair = (2/Z)(1 − p_c/2).
   - p_c/2 is a **topological constant**: p_c ÷ (crossing number of Hopf link = 2).
   - Same p_c/2 appears in J_1s² = ½ + p_c/2 (same physics, opposite sign).

3. **N-electron bonding mode** (K_N graph eigenvalue): ω_bond = ω₀/√(1 + k_pair × (N−1)).
   - N enters through λ_bond(K_N) = N−1, NOT through p_c.
   - IE = E₀ × (2/√(1+k_pair) − 1) for N=2 (He).

4. **Cross-shell coupling** (Op3 + Op4 + Op6): The inner shell creates a **step** in V(r) at r₁ₛ.
   - V(r) = −Zαℏc/r for r < r₁ₛ; V(r) = −(Z−N₁)αℏc/r for r > r₁ₛ.
   - This is an **impedance mismatch** (Op3), NOT a uniform Z_eff reduction.
   - The eigenvalue of the stepped cavity ≠ Z_eff²Ry/n² (pitfall #8).
   - Found from S₁₁ dip of `graded_tl_eigenvalue` (Op6).
   - ⚠️ Previous claim "Gauss screening only" was WRONG (gave Li −37%).

5. **Result**: He IE = **24.37 eV** (exp 24.587, **−0.9%**). Zero free parameters.

### Approach 8: Angular Momentum = Lattice BEMF (FOC Decomposition)

**Breakthrough insight**: Angular momentum ℓ is literally BEMF (back-EMF) in the vacuum lattice. This is the same physics as Field-Oriented Control (FOC) for BLDC motors.

**BLDC ↔ Atom Isomorphism**:
```
BLDC Motor (FOC)              Atom (AVE Lattice)
─────────────                 ──────────────────
Rotor spinning → BEMF         Electron ℓ → lattice BEMF
BEMF opposes applied voltage   BEMF opposes nuclear Coulomb field
d-axis (flux/field)            d-axis (radial) → SCREENING FLUX
q-axis (torque/rotation)       q-axis (angular) → ORBITAL TORQUE
Only d-axis couples to load    Only d-axis screens the nucleus
```

**Physical picture**: each electron mode (n, ℓ) stores energy in two channels:
- **d-axis (radial)**: field energy storage (capacitive) → SCREENS the nucleus
- **q-axis (angular)**: angular momentum storage (inductive) → creates BEMF barrier

The BEMF ∝ ℓ(ℓ+1) creates an impedance that opposes the nuclear field:

| ℓ | Mode | Angular KE ∝ ℓ(ℓ+1) | BEMF | Screening quality |
|---|------|---------------------|------|-------------------|
| 0 | s | 0 | none | **full** — all d-axis |
| 1 | p | 2 | medium | reduced |
| 2 | d | 6 | large | **poor** — mostly q-axis |
| 3 | f | 12 | massive | almost zero |

**Why d-electrons screen poorly**: most of their energy is stored as angular momentum (q-axis inductance / BEMF), leaving very little d-axis current to create screening flux. The impedance mismatch Γ = (Z_d - Z_s)/(Z_d + Z_s) between a high-ℓ shell and the outer s-shell is LARGE, creating a reflection barrier.

**θ (power factor angle)**: From FOC, `cos²(θ_ℓ)` = d-axis power fraction = screening quality:
```
cos²(θ_ℓ) = 1 - 2ℓ(ℓ+1) / (n(2ℓ+1))

(n,ℓ)    cos²θ    label
(1,0)    1.000    1s — full screening
(2,1)    0.333    2p — 1/3 screening
(3,2)    0.200    3d — 1/5 screening (KEY)
(4,3)    0.143    4f — 1/7 screening
```

**Time-averaged envelope** (from QM translation: |ψ|² = time-averaged trajectory of point-defect sweeping its mode):
- **Filled** subshell (all m occupied): time-averaged envelope = sphere → from outside, screens fully (Gauss's law)
- **Partially filled** subshell: envelope has angular gaps → nuclear field leaks through
- Coverage fraction = min(c, 2ℓ+1) / (2ℓ+1) (number of occupied m values / total)
- **Fe 3d⁶**: 6 electrons fill 5 up-spin + 1 down → 5/5 m-coverage for majority spin → spherical + extra

**Initial test results** (approaches 8a-8c, see audit trail for details):

| Variant | He | C | Fe | Issue |
|---------|-----|-----|-----|-------|
| 8a: d-axis only | -44.7% | -16.1% | +2504% | No J_geo overlap, overscreens He |
| 8b: J_geo × cos²θ | +4.6% | -59% | +3772% | J_geo too small for cross-shell |
| 8c: inner=full, same=J×cos²θ | +4.6% | +14.9% | +574% | Li underscreens (Z_eff=1.0) |

**Status**: Approach 8 has the right physics (BEMF/FOC/envelope) but computed screening as arithmetic subtraction, violating Axioms 1 and 3. See Approach 9.

### Approach 9: Impedance Cascade (AVE-Native)

**Insight**: Each shell (n,ℓ) has a **radial impedance** Z_d = √(cos²θ/2), derived from the d-q split of KE/PE. Shell boundaries produce **Γ reflections** (Axiom 3). The cascade of Γ values through the shells gives Z_eff directly — no SCF, no subtraction, no magic numbers.

```
Z_d(n,ℓ) = √(cos²θ_ℓ / 2)

Γ_{k→k+1} = (Z_{d,k+1} - Z_{d,k}) / (Z_{d,k+1} + Z_{d,k})

Z_eff = Z × (1 - Γ²_total)

IE = Z_eff² × Ry / n²
```

Uses **only** `universal_impedance` and `universal_reflection` from the engine.

**Status**: Approach 9 (pure Γ cascade) tested — failed because it only sees inter-shell boundaries. He: +121% (no inner boundary → no screening). Approach 9b evolved to include shunt Y.

### Approach 9b: Series Z + Shunt Y (ABCD Cascade)

Maps protein engine architecture to atom: series Z_d per shell + shunt Y_J from intra-shell Coulomb repulsion.

```
Y_J(n,ℓ,c) = c(c-1)/2 × J_geo(n) × cos²(θ_ℓ) / Z_d(n,ℓ)

ABCD per section = [[1+Z_d·Y_J, Z_d], [Y_J, 1]]

Z_eff = (A·Z + B) / (C·Z + D)   [ABCD cascade terminated at Z]
```

| Z | El | Exp | 9b | Error | Z_eff |
|---|-----|------|------|-------|-------|
| 2 | He | 24.59 | 27.81 | +13.1% | 1.43 |
| 5 | B | 8.30 | 8.11 | **-2.2%** | 1.54 |
| 10 | Ne | 21.57 | 6.18 | -71.3% | 1.35 |
| 26 | Fe | 7.90 | 2.69 | -65.9% | 1.33 |

**Key finding**: He improved massively (+121% → +13.1%). B is nearly exact. But Z_eff saturates to ~1.33 for all large Z — the shunt Y accumulates too aggressively through the ABCD cascade.

**Approaches 10–10c: Axiom 4 Voltage Strain** — Fe 1s voltage = 18.4 kV = 42% of V_YIELD (Regime II onset). Screening enhanced by 1/S, but the base model (σ arithmetic) remains wrong.

### Approach 12: Pure 5-Step Eigenvalue Method (The Correct Pattern)

**Root cause of all previous failures**: approaches 1–10c computed screening as **$\sigma$ arithmetic** (how much does e⁻-e⁻ repulsion reduce Z?), implicitly relying on the Schrödinger/Bohr formula $E = Z_{eff}^2 Ry/n^2$. The nuclear solver and protein engine never do this — they solve for **eigenmodes** of coupled LC resonators.

The AVE framework requires atomic ionization energies to emerge purely from the 5-step universal regime-boundary method.

**1. Rydberg emerges exactly (0.000000% error) from pure topology:**
```
r_sat = L_NODE/α = a₀            (Hydrogen cavity boundary)
r_eff = r_sat/(1+ν) = 7/9 a₀      (Effective radius)
E₀ = ℏc/r_eff = 4794 eV           (Fundamental cavity mode)
k = α/(1+ν) = 7α/9                (Coupling coefficient)
Ry = E₀ × k/2 = α²m_e c²/2        (Emergent rest energy)
```
**2. Orbital Radii ($r_n = n^2 a_0$) are Macroscopic Standing Waves:**
In AVE, discrete orbitals are Fabry-Pérot impedance cavities, identical to **Saturn's rings** and **Kirkwood gaps**. For an electron phase wave with velocity $v \propto 1/\sqrt{r}$, the phase-matching condition $2\pi r = n \lambda$ uniquely forces $r_n \propto n^2$. This is an AVE geometric theorem, not a QM import.

**3. Electron $d_{sat} = L_{NODE}$ means Strict Linearity:**
The electron's topological saturation radius is its reduced Compton wavelength ($L_{NODE}$). Because inter-electron separation is on the order of $a_0$, the pairwise voltage strain is $A = L_{NODE}/a_0 = \alpha \approx 0.007$. This places intra-atomic repulsion strictly in the **Linear Regime (Regime I)**. No saturated impedance corrections ($1/S$) are needed for electron-electron coupling; pure Coulomb electrostatics apply.

**The Multi-Electron Solver (Approach 14: Phase-Locked EM Gear):**

The atom is a **3D electromagnetic gear**: orbiting topological defects on concentric Fabry-Pérot shells, phase-locked by Coulomb coupling. Key discoveries:

**Step A — Mode Shape from Operator 3 (Γ):** *(unchanged)*
The standing wave between reflection boundaries gives the radial mode shape $\phi_{n\ell}(r)$ = time-averaged orbiting charge density.

**Step B — Phase-Locking is MANDATORY:**
Two circular orbits on the same sphere ALWAYS intersect at 2 points. The orbit-averaged Coulomb integral $\langle 1/r_{12} \rangle$ DIVERGES for independent (uncorrelated) orbits. Therefore, orbiting defects MUST phase-lock to avoid collision. Phase-locking IS the ground state, not a correction.

Phase-locked orbit-averaged Coulomb (computed numerically):
- Same orbit, antipodal: $\langle 1/r_{12} \rangle = 1/(2r)$ — exact
- Perpendicular orbits (90°), phase-locked: $\langle 1/r_{12} \rangle = 0.59/r$
- Coupling increases monotonically with orbital plane inclination

**Step C — Transmission Line Standing Wave Coupling:**
The shell circumference $= n\lambda_{dB}$ (phase-matching). $N_v$ phase-locked defects have angular separation $\phi = 2\pi/N_v$, giving electrical length $d/\lambda = n\phi/(2\pi) = n/N_v$.

For $n=2$ shell: $d/\lambda = 2/N_v$. Harmonic ratios (integer multiples of $\lambda/2$) give standing wave NODES = minimum coupling = maximum stability:
- $N_v=2$ ($d/\lambda = 1$): **node** → Be stable ✓
- $N_v=4$ ($d/\lambda = 1/2$): **node** → C stable ✓
- $N_v=8$ ($d/\lambda = 1/4$): **node** → Ne stable ✓
- $N_v=3, 5, 6$: non-harmonic → weaker stability → matches periodic table trend

**Step D — Inner Shell Screening (PROVEN, σ ≈ 2.0):**
Derived from Gauss's law + cavity mode shapes:
$$\sigma = 2 \times \langle \text{CDF}_{1s}(r) \rangle_{\text{outer}}$$
Self-consistent computation gives $\sigma \approx 1.95\text{-}1.97$ for all $Z = 3\text{-}10$. The 1s mode shape is tightly bound ($r_1 \approx 0.37\,a_0$ for Li), so nearly all its charge is enclosed by the outer orbit.

⚠️ **CORRECTION**: The previously calibrated $\sigma_{core} = 1.75$ was WRONG — it was absorbing the missing intra-shell repulsion. The correct first-principles value is $\sigma \approx 2.0$, and the IE deficit must come from intra-shell repulsion (the EM gear coupling).

**Step E — Flux Loop IS the Shell / Figure-8 Reconnection (Approach 16):**

The electron is a torus knot: major radius $R = r_n$, tube radius $a = L_{NODE}$. The electron IS the shell, not a point ON it.

**Crossing theorem + Axiom 4 → Pauli exclusion:**
Two loops on the same sphere ALWAYS cross at 2 points. At each crossing, the flux tubes overlap. From Axiom 4, the total strain at the crossing:
$$A_{total}^2 = A_1^2 + A_2^2 + 2A_1 A_2 \cos\phi$$
- **Same spin** ($\phi = 0$): $A = 2A_{yield}$ → $S = \sqrt{1-4}$ → **imaginary → FORBIDDEN**
- **Opposite spin** ($\phi = \pi$): $A = 0$ → $S = 1$ → **vacuum healed → ALLOWED**

**Reconnection:** Where antiparallel tubes cross, the strain cancels to zero — no boundary exists between them. The two loops **magnetically reconnect**, merging into a single continuous figure-8 path. This is vacuum-lattice reconnection.

**He = one figure-8 flux structure carrying charge $-2e$:**
- NOT two discrete electrons tolerating each other
- ONE topological object: a self-intersecting loop (figure-8) wrapping the nucleus twice
- Charge $-2e$ circulates on this single continuous path
- QM's "exchange energy" = the energy gained by reconnection (merging two separate loops into one)
- He's stability = topological completeness; a 3rd loop at the same radius would have parallel flux with at least one winding → $A > A_{yield}$ → forbidden (Pauli)

**Proven motor constants (Approach 15):** $K_e = Z_0 e/(2\pi)$, BEMF $= 4\alpha R_y$ (fine structure). Vacuum transparent at Regime I ($Z \approx Z_0$, correction $\alpha^4$). Self-inductance energy $\alpha^4$ (negligible).

**Open:** Self-energy of figure-8 path vs two separate loops. Generalization to N > 2 (what topologies form on the n=2 shell?). Mapping knot topologies to $(n, \ell, m)$.

### Approach 17: Coupled Oscillator Dynamical Matrix (❌ False Avenue #14)

Attempted to use the **nuclear solver pattern** (dynamical matrix eigenvalues) for atoms. Diagonal: $\varepsilon_i^2 = (Z^2 Ry/n_i^2)^2$. Off-diagonal: $k_{geo} \times \varepsilon_i \times \varepsilon_j$ with both QM integrals ($5/8$, $17/81$) and lattice J scalars.

**Results**: 1900% mean error (QM integrals), 109% mean error (lattice J scalars). He: +96% / +99%.

**Root cause**: the nuclear solver handles **attractive** coupling (binding energy from mode splitting). Electron-electron coupling is **repulsive**. Positive off-diagonal terms in the dynamical matrix don't produce the right physics — the mode spectrum for repulsive coupling is fundamentally different from attractive. Also: atoms are a **radial cascade**, not a mesh of identical oscillators. This is **false avenue #12/14**.

### Approach 18: 5-Step Eigenvalue + σ-Arithmetic (❌ QM Contamination)

Applied the 5-step eigenvalue method ($r_{sat} \to r_{eff} \to E_0 \to k \to E$) with static screening $\sigma$ subtracted from $Z$: $IE = (Z-\sigma)^2 Ry / n^2$.

**Dimensional chain verified**: 5-step gives Ry to $10^{-16}$ precision, He$^+$ = 54.4 eV ✓. But:

**Root cause**: $IE = Z_{eff}^2 Ry / n^2$ IS the Bohr/Schrödinger formula (Pitfall #8). The 5-step derives this FROM the axioms for a pure Coulomb well, but $\sigma$-arithmetic on top is QM contamination. The correct approach must produce the eigenvalue **from the full potential** (including screening), not from a reduced Coulomb well.

### Approach 19: Radial TL with graded_tl_eigenvalue (✅ H Verified)

**Breakthrough**: applied the existing `graded_tl_eigenvalue` solver (used for BH ringdown, HOPF-01 antenna, protein backbone) to the atom. The solver was literally designed for "Electron orbital" (line 63 of `resonator.py`) but had never been tested with atomic parameters.

**Physical setup**:
- The nucleus projects a **spherical** Coulomb field (Axiom 2)
- The lattice impedance is Z₀ = 377 Ω everywhere (Regime I: $V/V_{yield} \approx 10^{-4}$)
- The cavity forms from the electron defect's **de Broglie dispersion**: $n(r) = \sqrt{2Za_0/r - 1}$
- Inner boundary: short circuit (nucleus). Outer: evanescent beyond $r_{turn} = 2a_0/Z$

**Results**:
- H (Z=1, no stubs): **$E_1 = 13.585$ eV, error = −0.09%** ✅
- Converges with log grid, $r_{inner} = 7 \times 10^{-5}\,a_0$, N_shells = 2000
- WKB phase integral $\int k\,dr = \pi$ confirmed analytically

**What failed for He**:
1. **Static shunt stubs** at $r_{1s}$: a point admittance $Y_{stub}$ at one grid point doesn't shift the eigenvalue — too localized, non-resonant
2. **Static step in $n(r)$**: modifying $n(r)$ with a step at $r_{1s}$ to use $Z-\sigma$ ALSO fails because the WKB phase integral $\int k\,dr = \pi$ is **Z-independent** — any pure Coulomb well gives $f \approx 1$ regardless of Z. Energy mapping $E = f \times Ry$ gives $E \approx Ry$ for ALL Z.
3. The n(r) formula was initially wrong: $\sqrt{2Za_0/r - 1}$ doesn't give $k \propto Z/a_0$. The correct formula is $Z\sqrt{2a_0/(Zr) - 1}$, but this still gives $\int n\,dr = a_0\pi$ for all Z.

**Root cause of Z² failure**: `n(r)` was computed at FIXED trial energy. The refractive index MUST depend on the trial energy $\xi$ (normalized frequency = $E/Ry$):
$$n(r, \xi) = \sqrt{2 Z_{eff}(r) \cdot a_0/r - \xi}$$
For H at resonance $\xi=1$: $n = \sqrt{2a_0/r - 1}$ ✓.
For He$^+$ at resonance $\xi=4$: $n = \sqrt{4a_0/r - 4}$ ✓ (turning point at $a_0$, not $2a_0$).

### Approach 20: Dielectric Reflector Model (Current Path)

**Key redefinition**: screening is NOT "reducing the nuclear charge." It is an **impedance transformation** — the inner electron creates an Op3 reflection ($\Gamma$) that changes the boundary condition of the outer electron's cavity.

**QM screening** (σ-arithmetic):
$$V(r) = -(Z-\sigma)\alpha\hbar c/r \quad \forall\, r > r_{1s}$$
Same 1/r shape → eigenvalue trivially $Z_{eff}^2 Ry$ → Bohr formula

**AVE screening** (dielectric reflector):
The inner shell creates a **step in $Z_{eff}(r)$**. This means $n(r,\xi)$ has a **discontinuity** at $r_{1s}$:
$$n(r, \xi) = \begin{cases} \sqrt{2Z \cdot a_0/r - \xi} & r < r_{1s} \\ \sqrt{2(Z-\sigma) \cdot a_0/r - \xi} & r > r_{1s} \end{cases}$$
The discontinuity is an **impedance mismatch** (Op3). The reflected wave picks up a phase $\phi(\Gamma)$ that shifts the eigenvalue away from the pure-Coulomb value. The eigenvalue of a non-1/r potential is NOT $Z_{eff}^2 Ry$ — it must be found numerically from the ABCD cascade.

**Implementation**: energy-dependent `n_func(r, ξ)` recomputed at each frequency point in the sweep. The solver finds $S_{11}$ dips as usual. The dip frequency $\xi_0$ gives $IE = \xi_0 \times Ry$ directly, with no Bohr formula.

**Operator chain**:
| Step | Operator | Physics |
|------|----------|---------|
| Nuclear field | Op1 (Z) | $V(r) = -Z\alpha\hbar c/r$ |
| Inner shell | Op3 (Γ) | step in $Z_{eff}(r)$ → impedance mismatch |
| Lattice coupling | Op2 (S) | $p_c$ → $J_{1s^2}$ → same-shell $\sigma$ |
| Eigenvalue | Op6 ($\lambda_{min}$) | $S_{11}$ dip = bound state |

**The σ→J Connection (Two Distinct Physics):**

**Cross-shell screening** (inner shell screens outer shell):
- Mechanism: **Gauss's law** (geometric theorem on enclosed dislocations)
- $\sigma_{cross} = N_{inner} \times 1.0$
- Example: Li 1s² core screens the 2s valence → $\sigma = 2$
- No J needed — pure electrostatics from Axiom 2

**Same-shell repulsion** (electrons on the same orbital shell):
- Mechanism: **Lattice phase-jitter coupling** from Axiom 4
- $\sigma_{same} = (N_{same} - 1) \times J_{shell}$
- Example: He 1s² → $\sigma = 1 \times J_{1s^2} = 0.5917$
- J enters because neither electron is "inner" — they're on the same figure-8 at the same radius
- The repulsive potential adds to V(r) proportionally to J: $V_{ee} \propto J \cdot \alpha\hbar c / r$

**Combined screening rule:**
$$\sigma_{total} = \underbrace{N_{inner}}_{\text{Gauss (Axiom 2)}} + \underbrace{(N_{same} - 1) \times J_{shell}}_{\text{Lattice (Axiom 4)}}$$

| Atom | Config | $\sigma_{cross}$ | $\sigma_{same}$ | $Z_{eff}$ |
|------|--------|-------------------|-----------------|-----------|
| He | 1s² | 0 | 1×J₁s² = 0.59 | 1.41 |
| Li | [He]2s¹ | 2 | 0 | 1.0 |
| Be | [He]2s² | 2 | 1×J₂s² = 0.18 | 1.82 |
| B | [He]2s²2p¹ | 2+2×0.85 | 0 | 0.30 |
| N | [He]2s²2p³ | 2+2×0.85 | 2×J₂p = 1.37 | — |

⚠️ **B/N entries**: cross-shell screening of 2p by 2s is imperfect ($\sigma_{2s→2p} \approx 0.85$ per electron, not 1.0). The CDF of the 2s orbital at the 2p radius is ~0.85 (from cavity mode shape analysis). Need to compute: $\sigma_{cross}(2s→2p) = N_{2s} \times \text{CDF}_{2s}(r_{2p})$.


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
| 42 | WD redshift (Sirius B) | 3.7% | ✅ z = GM/(c²R), v_GR=77.75 km/s, v_obs=80.65±0.77 |
| 43 | α invariance (gravity) | Exact | ✅ Δα/α = 0 under Symmetric Gravity (Axiom 3) |
| 44 | BCS B_c(T) | 0.00% | ✅ B_c(T) = B_c0·S(T/T_c) IS the saturation operator (Al, Pb, Nb, MgB₂) |
| 45 | BH interior (Regime IV) | Exact | ✅ G_shear = 0, c_eff = 0 for r < r_sat = 7GM/c². Symmetric saturation → Z = Z₀, Γ = 0 (dissipative sink). |
| 46 | Regime IV isomorphism | — | ✅ BH (sym, hole) ≠ electron (asym, knot). Same S=0 operator, different saturation symmetry. |

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

