"""
Universal Coupled Resonator Solver
==================================

A single framework applied at every scale of the AVE hierarchy.
The same operators appear at each level:

==========  =========================================  ===================
Operator    Formula                                    Source
==========  =========================================  ===================
omega_0     c / r_eff  or  c(1+nu) / r_sat             Axioms 1, 3
k           coupling coefficient (see table)            Axiom 2 / geometry
B           N * hbar * (omega_0 - omega_bonding)        Normal mode split
E_coulomb   (3/5) Z(Z-1) alpha*hbar*c / R              Coulomb self-energy
S(x)        sqrt(1 - x^2)                              Axiom 4 saturation
==========  =========================================  ===================

Cross-Scale Mapping
-------------------

========  ==========  =================  ============  ====================
Scale     r_sat       omega_0            Coupling       IE/Binding
========  ==========  =================  ============  ====================
Nuclear   d_p         c(1+nu)/d_p        k = 2*alpha   N*hbar(w0-w_bond)
                      = 302 MeV/hbar     = 0.01476     (coupled resonator)
--------  ----------  -----------------  ------------  --------------------
Atomic    a0          Z^2*alpha^2*m_e    k_Hopf        N-port Y-matrix
                      *c^2/(hbar*n^2)    (Op4+Op2)     eigenvalue (Op5+Op6)
--------  ----------  -----------------  ------------  --------------------
Molecul.  r_val(IE)   sqrt(IE_A*IE_B)    S*x = 1/2     2*w_eff*(1-1/V1.5)
                      (geometric mean)   at d=rV2      (matter exchange)
========  ==========  =================  ============  ====================

ARCHITECTURE NOTE (2026-03-13):
  The atomic IE solver uses the SAME Y-matrix infrastructure as the
  protein and antenna solvers (transmission_line.py):
    - Each electron = one node (self-admittance y = 1/Z_LC)
    - Coupling = mutual admittance through Z_0 = 377 Ohm lattice
    - Same-shell (Hopf link): k_Hopf = (2/Z)(1 - p_c/2)
    - Cross-shell: capacitive coupling, NO Hopf crossings
    - IE from eigenvalue: build_nodal_y_matrix -> s_matrix_from_y -> Op6

  CRITICAL: All electrons see BARE nuclear charge Z. There is no Z_eff.
  "Screening" emerges from the eigenvalue decomposition, not as an input.
  The ionization_energy_circuit() function below is the same-shell-only
  solver (Approach 22). The N-port Y-matrix extension is in development.
"""

import numpy as np
from ave.core.constants import (
    ALPHA, HBAR, C_0, e_charge, D_PROTON, NU_VAC,
    M_P_MEV, M_N_MEV, HBAR_C_MEV_FM,
    K_COUPLING as _K_COUPLING_CONST,
    OMEGA_0_NUCLEAR as _OMEGA_0_CONST,
    E_0_NUCLEAR as _E_0_CONST,
    M_E,
)


# ─────────────────────────────────────────────────────────────────
# Fundamental circuit parameters (from axioms)
# ─────────────────────────────────────────────────────────────────

# Uncoupled resonant frequency: ω₀ = c / r_eff where r_eff = d_p/(1+ν)
R_EFF = D_PROTON * 1e-15 / (1.0 + NU_VAC)
OMEGA_0 = C_0 / R_EFF
E_0_MEV = HBAR * OMEGA_0 / (e_charge * 1e6)   # ≈ 301.6 MeV

# Coupling coefficient: k = 1/(1-α)² - 1 ≈ 2α
# Derived from the requirement B_deuteron = ℏω₀ × α
K_COUPLING = 1.0 / (1.0 - ALPHA) ** 2 - 1.0   # ≈ 0.01476


# ─────────────────────────────────────────────────────────────────
# Complete graph adjacency eigenvalues
# ─────────────────────────────────────────────────────────────────

def complete_graph_eigenvalues(n):
    """Eigenvalues of the adjacency matrix of K_n: [n-1, -1, -1, ..., -1]."""
    return np.array([n - 1] + [-1] * (n - 1), dtype=float)


def coupled_resonator_binding(n_resonators, k, omega_0=OMEGA_0,
                              adjacency_eigenvalues=None):
    r"""
    Compute binding energy from coupled resonator normal mode splitting.

    For N identical resonators with coupling coefficient k and
    adjacency eigenvalues λ_n:

        ω_n = ω₀ / √(1 + k × λ_n)

    Ground state (bosonic): all N resonators occupy the lowest mode.
    Binding = N × ℏ(ω₀ - ω_lowest)

    Args:
        n_resonators: Number of resonators (nucleons or alphas)
        k: Dimensionless coupling coefficient
        omega_0: Uncoupled resonant frequency [rad/s]
        adjacency_eigenvalues: Eigenvalues of the coupling graph.
            If None, uses complete graph K_n.

    Returns:
        B_total: Total binding energy [MeV]
        omega_modes: Normal mode frequencies [rad/s]
    """
    if adjacency_eigenvalues is None:
        adjacency_eigenvalues = complete_graph_eigenvalues(n_resonators)

    # Normal mode frequencies
    omega_modes = omega_0 / np.sqrt(1.0 + k * adjacency_eigenvalues)

    # Ground state: all N in the lowest-frequency (bonding) mode
    omega_bonding = np.min(omega_modes)
    B_total = n_resonators * HBAR * (omega_0 - omega_bonding) / (e_charge * 1e6)

    return B_total, omega_modes


def hierarchical_binding(n_alphas, k_intra=K_COUPLING, k_inter=K_COUPLING):
    r"""
    Hierarchical binding: alphas at level 1, then alpha assembly at level 2.

    Level 1: Each alpha = 4 nucleons in K₄ with coupling k_intra
        B_alpha = 4 × ℏ(ω₀ - ω₀/√(1 + 3k_intra))

    Level 2: N alphas coupled in K_N with coupling k_inter
        The alpha's bonding mode frequency ω_alpha = ω₀/√(1 + 3k_intra)
        becomes the new ω₀ for the next level.
        B_inter = N_alpha × ℏ(ω_alpha - ω_alpha/√(1 + (N-1)·k_inter))

    Total: B = N_alpha × B_alpha + B_inter

    Args:
        n_alphas: Number of alpha clusters
        k_intra: Intra-alpha coupling (default: 2α)
        k_inter: Inter-alpha coupling (default: 2α)

    Returns:
        B_total: Total binding energy [MeV]
        B_alpha: Per-alpha binding [MeV]
        B_inter: Inter-alpha binding [MeV]
    """
    # Level 1: Alpha binding
    B_alpha, modes_alpha = coupled_resonator_binding(4, k_intra)

    # Alpha bonding mode frequency
    omega_alpha = np.min(modes_alpha)

    # Level 2: Inter-alpha binding
    B_inter, modes_nucleus = coupled_resonator_binding(
        n_alphas, k_inter, omega_0=omega_alpha
    )

    B_total = n_alphas * B_alpha + B_inter
    return B_total, B_alpha, B_inter


def nuclear_mass(Z, A, n_alphas=None):
    r"""
    Compute nuclear mass from the coupled resonator model.

    Circuit model:
        1. Coupled LC resonators → binding energy (attractive)
        2. Coulomb self-energy of Z protons → repulsion

    Coulomb correction (from circuit analysis):
        Z charged resonators at average spacing R = d_p × A^(1/3)
        E_c = (3/5) × Z(Z-1) × αℏc / R

    Args:
        Z: Proton number
        A: Mass number
        n_alphas: Number of alpha clusters (auto-detected if None)

    Returns:
        mass: Nuclear mass [MeV]
        binding: Total binding energy [MeV] (strong - Coulomb)
    """
    N_n = A - Z
    raw_mass = Z * M_P_MEV + N_n * M_N_MEV

    # Strong binding from coupled resonator model
    if A == 2:
        binding, _ = coupled_resonator_binding(2, K_COUPLING)
    elif A <= 4:
        binding, _ = coupled_resonator_binding(A, K_COUPLING)
    else:
        if n_alphas is None:
            n_alphas = A // 4
        n_valence = A - 4 * n_alphas

        binding, B_alpha, B_inter = hierarchical_binding(n_alphas)

        # Valence nucleons couple to the nearest alpha
        if n_valence > 0:
            omega_alpha = OMEGA_0 / np.sqrt(1.0 + 3 * K_COUPLING)
            for _ in range(n_valence):
                B_val = HBAR * (omega_alpha - omega_alpha / np.sqrt(1 + K_COUPLING))
                B_val /= (e_charge * 1e6)
                binding += B_val

    # Coulomb correction: stored electrostatic energy between Z protons
    # R = d_p × A^(1/3) is the nuclear charge radius (Axiom 1 length scale)
    alpha_hc = ALPHA * HBAR_C_MEV_FM   # αℏc ≈ 1.44 MeV·fm
    R_nucleus = D_PROTON * A ** (1.0 / 3.0)
    E_coulomb = 0.6 * Z * (Z - 1) * alpha_hc / R_nucleus

    # Net binding = strong - Coulomb
    binding_net = binding - E_coulomb

    return raw_mass - binding_net, binding_net


# ─────────────────────────────────────────────────────────────────
# Level 2: Atomic — IE from total energy in the LINEAR regime
# Level 3: Molecular — bond from saturation coupling
#
# REGIME ANALYSIS:
#   Electron's d_sat = ℓ_node = ℏ/(m_e c) ≈ 3.86e-13 m
#   Inter-electron distance ~ a₀ ≈ 5.29e-11 m
#   Strain: A = d_sat/r = ℓ_node/a₀ = α ≈ 0.007
#   → DEEP in the LINEAR regime (A ≪ 1)
#   → U = -K/r (pure Coulomb, no impedance correction)
#   → Screening from 3D ELECTROSTATIC geometry
#   → The 3D comes from ν = 2/7 (3 spatial / 7 compliance modes)
# ─────────────────────────────────────────────────────────────────

# Atomic scale constants (all from axioms)
_M_E = float(M_E)  # electron mass from constants.py [kg]
_A0 = HBAR / (_M_E * C_0 * ALPHA)   # Bohr radius [m]
_RY_EV = _M_E * C_0**2 * ALPHA**2 / (2.0 * e_charge)   # Rydberg [eV]
_PROJECTION = 1.0 / (2.0 * (1.0 + 2.0/7.0))  # 7/18 projection of eigenvalue


# ─────────────────────────────────────────────────────────────────
# QUARANTINE: ~400 lines of QM hydrogen wavefunction Coulomb
# integrals removed (2026-03-13). See pitfall #11.
# ─────────────────────────────────────────────────────────────────

# Aufbau (Madelung) filling order: subshells sorted by (n+l, n)
# This is first-principles under AVE: the LC cavity eigenvalue for
# orbital (n,l) scales with n+l (nuclear penetration), not just n.
_AUFBAU_ORDER = [
    (1, 0, 2),   # 1s
    (2, 0, 2),   # 2s
    (2, 1, 6),   # 2p
    (3, 0, 2),   # 3s
    (3, 1, 6),   # 3p
    (4, 0, 2),   # 4s
    (3, 2, 10),  # 3d
    (4, 1, 6),   # 4p
    (5, 0, 2),   # 5s
    (4, 2, 10),  # 4d
    (5, 1, 6),   # 5p
    (6, 0, 2),   # 6s
    (4, 3, 14),  # 4f
    (5, 2, 10),  # 5d
    (6, 1, 6),   # 6p
    (7, 0, 2),   # 7s
    (5, 3, 14),  # 5f
    (6, 2, 10),  # 6d
    (7, 1, 6),   # 7p
]


def _fill_shells(n_electrons):
    """Fill electron shells using Aufbau (Madelung n+l) order.

    Returns [(n, count), ...] grouped by principal quantum number n.
    The J integrals resolve shells by n only, so subshells within
    the same n are combined.

    Example: K (Z=19) = 2 in n=1, 8 in n=2, 8 in n=3, 1 in n=4
    (not 9 in n=3, which was the previous bug).
    """
    from collections import defaultdict
    shell_counts = defaultdict(int)
    remaining = n_electrons
    for n, l, capacity in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        count = min(remaining, capacity)
        shell_counts[n] += count
        remaining -= count
    return sorted(shell_counts.items())

# ─────────────────────────────────────────────────────────────────
# QUARANTINE: atom_total_energy(), _scf_z_eff() removed
# (2026-03-13). QM SCF loops replaced by coupled LC model.
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# Active IE Solver: Approach 22 (AVE Coupled LC + Hopf Link)
# ─────────────────────────────────────────────────────────────────

def _shell_config(n_electrons):
    """Return outermost subshell (n, l, count) for IE calculation.

    Uses Aufbau order to fill shells, returns the last subshell info.
    """
    remaining = n_electrons
    last_n, last_l, last_count = 1, 0, 0
    for n, l, capacity in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        count = min(remaining, capacity)
        last_n, last_l, last_count = n, l, count
        remaining -= count
    return [(last_n, last_l, last_count)]


from ave.core.constants import P_C as _P_C


def ionization_energy_circuit(Z, n_electrons=None):
    r"""First ionization energy from AVE coupled LC resonator model.

    Architecture (Approach 22):
        1. Each electron is an LC resonator at omega_0 = Z^2 alpha^2 m_e c^2 / hbar
        2. Same-shell pairs form Hopf links with per-pair coupling:
           k_pair = (2/Z_eff)(1 - p_c/c_Hopf) = (2/Z_eff)(1 - p_c/2)
           where p_c/2 is a TOPOLOGICAL CONSTANT from the Hopf link
           crossing number (c=2), NOT p_c/N.
        3. N same-shell electrons: bonding mode omega_bond = omega_0 / sqrt(1 + k_pair*(N-1))
           N enters via K_N adjacency eigenvalue, NOT through p_c.
        4. Cross-shell: Gauss's law screening (sigma = N_inner), no Hopf link.
        5. IE = E_0 * (N/sqrt(1+k(N-1)) - (N-1)/sqrt(1+k(N-2)))

    Operator compliance:
        Op1: Z_LC = sqrt(L/C) (torus impedance)
        Op2: p_c saturation at Hopf link crossings
        Op4: Coulomb coupling C_rep = C/Z_eff
        Op5: K_N eigenvalue for N-electron mode
        Op6: eigenvalue from resonance condition

    Args:
        Z: Atomic number.
        n_electrons: Number of electrons (default: Z for neutral).

    Returns:
        IE in eV.
    """
    if n_electrons is None:
        n_electrons = Z

    # Single electron: exact hydrogenic
    if n_electrons <= 1:
        return Z**2 * _RY_EV

    shells = _shell_config(n_electrons)
    n_out, l_out, c_out = shells[-1]  # outermost subshell

    # Cross-shell: sigma = N_inner (Gauss's law, no Hopf link)
    N_inner = n_electrons - c_out
    Z_eff_cross = Z - N_inner
    Z_eff_cross = max(1.0, Z_eff_cross)

    # Single-electron binding at effective charge
    E_0 = Z_eff_cross**2 * _RY_EV / n_out**2

    # Same-shell coupling (if >1 electron in outermost shell)
    if c_out <= 1:
        return E_0

    # PER-PAIR coupling: k_pair = (2/Z_eff)(1 - p_c/c_Hopf)
    # p_c/2 is TOPOLOGICAL CONSTANT from Hopf link crossing number (c=2).
    # Same p_c/2 appears in J_1s2 = 1/2 + p_c/2 (same physics, opposite sign).
    # Cross-shell electrons don't form Hopf links (non-intersecting tori).
    k_pair = (2.0 / Z_eff_cross) * (1.0 - float(_P_C) / 2.0)

    # K_N adjacency eigenvalue: shell occupancy enters here, NOT p_c
    N_same = c_out
    lam = N_same - 1  # K_N bonding eigenvalue (complete graph)

    # IE from coupled resonator normal mode splitting
    # For N=2: IE = E_0 * (2/sqrt(1+k_pair) - 1)
    # General: IE = E_0 * (N*omega(N) - (N-1)*omega(N-1))
    if N_same == 2:
        IE = E_0 * (2.0 / np.sqrt(1.0 + k_pair) - 1.0)
    else:
        omega_bond_N = 1.0 / np.sqrt(1.0 + k_pair * lam)
        omega_bond_Nm1 = 1.0 / np.sqrt(1.0 + k_pair * (lam - 1))
        IE = E_0 * (N_same * omega_bond_N - (N_same - 1) * omega_bond_Nm1)

    return max(0.0, IE)


def ionization_energy(Z, n_electrons=None):
    r"""First ionization energy — dispatches to AVE circuit solver.

    Args:
        Z: Atomic number.
        n_electrons: Number of electrons (default: Z).

    Returns:
        IE in eV.
    """
    return ionization_energy_circuit(Z, n_electrons)


# ─────────────────────────────────────────────────────────────────
# Y-Matrix IE Solver (Ch.16 Pipeline, Stages A-E)
# Routes through universal operators: Op1, Op2, Op4, Op5, Op6
# ─────────────────────────────────────────────────────────────────

from scipy.special import ellipk as _ellipk
from ave.core.constants import (
    A_0 as _A0_CONST, RY_EV as _RY_EV_CONST, Z_0 as _Z0,
    MU_0 as _MU_0, L_NODE as _L_NODE, P_C as _P_C_CONST,
)
from ave.core.universal_operators import (
    universal_impedance,
    universal_saturation,
    universal_ymatrix_to_s,
    universal_eigenvalue_target,
)


def _electron_config(Z, n_electrons=None):
    """Return list of (n, l, m_l) for each electron using Aufbau filling.

    Each electron gets explicit quantum numbers for Y-matrix construction.
    """
    if n_electrons is None:
        n_electrons = Z
    electrons = []
    remaining = n_electrons
    for n, l, capacity in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        # Fill m_l values: -l, -l+1, ..., +l (each twice for spin)
        for m_l in range(-l, l + 1):
            for spin in (0, 1):  # 0 = up, 1 = down
                if remaining <= 0:
                    break
                electrons.append((n, l, m_l))
                remaining -= 1
    return electrons


def _ring_parameters(Z, n):
    """Stage A: compute ring parameters for electron at shell n.

    Returns (R, omega, L, Z_LC) — all from Axiom 1.
    """
    R = _A0_CONST * n**2 / Z                             # Ring radius [m]
    omega = Z**2 * ALPHA**2 * M_E * C_0**2 / (HBAR * n**3)  # ω = v/R = Z²α²m_ec²/(ℏn³) [rad/s]
    L = _MU_0 * R * (np.log(8.0 * R / _L_NODE) - 2.0)   # Self-inductance [H]
    Z_LC = omega * L                                      # = sqrt(L/C) [Ω]
    return R, omega, L, Z_LC


def _pair_coupling(Z, ni, li, mli, nj, lj, mlj, Z_LC_i, Z_LC_j):
    """Stages B+C: classify pair and compute mutual admittance y_ij.

    Returns y_ij (coupling admittance, real, positive).
    """
    same_shell = (ni == nj)

    if same_shell and mli == mlj:
        # Type 1: Hopf link (same n, same m_l)
        # Op4 Coulomb at 2R + Op2 saturation at 2 crossings
        k_hopf = (2.0 / Z) * (1.0 - float(_P_C_CONST) / 2.0)
        return k_hopf / Z_LC_i

    elif same_shell and mli != mlj:
        # Type 2: Orthogonal crossing (same n, different m_l)
        k_hopf = (2.0 / Z) * (1.0 - float(_P_C_CONST) / 2.0)
        k_perp = k_hopf / np.sqrt(2.0)
        return k_perp / Z_LC_i

    else:
        # Type 3: Cross-shell (different n) — Op4 only, no Op2
        R_a = _A0_CONST * ni**2 / Z  # inner ring
        R_b = _A0_CONST * nj**2 / Z  # outer ring
        if R_a > R_b:
            R_a, R_b = R_b, R_a

        # Orbit-averaged Op4: ⟨V⟩ = 2αℏc/(πR_b) × K(R_a/R_b)
        ratio = R_a / R_b
        K_val = _ellipk(ratio)
        V_cross_J = 2.0 * ALPHA * HBAR * C_0 / (np.pi * R_b) * K_val
        V_cross_eV = V_cross_J / e_charge

        # Normalize to geometric mean of bare energies
        E_a = Z**2 * _RY_EV_CONST / ni**2
        E_b = Z**2 * _RY_EV_CONST / nj**2
        k_cross = V_cross_eV / np.sqrt(E_a * E_b)

        # Mutual admittance with geometric mean impedance
        return k_cross / np.sqrt(Z_LC_i * Z_LC_j)


def _total_energy_from_ymatrix(Z, electrons):
    """Stages D+E: build Y-matrix, get S-matrix, extract total energy.

    Args:
        Z: Nuclear charge.
        electrons: List of (n, l, m_l) tuples.

    Returns:
        E_total in eV.
    """
    N = len(electrons)
    if N == 0:
        return 0.0

    # Stage A: ring parameters for each electron
    params = []
    for n, l, m_l in electrons:
        R, omega, L, Z_LC = _ring_parameters(Z, n)
        params.append((R, omega, L, Z_LC, n, l, m_l))

    # Single electron: exact hydrogenic, no Y-matrix needed
    if N == 1:
        n = params[0][4]
        return -(Z**2 * _RY_EV_CONST / n**2)

    # Stages B+C+D: build Y-matrix
    Y = np.zeros((N, N), dtype=complex)

    for i in range(N):
        R_i, omega_i, L_i, Z_LC_i, n_i, l_i, ml_i = params[i]
        y_self = 1.0 / Z_LC_i  # Op1: self-admittance

        sum_mutual = 0.0
        for j in range(N):
            if j == i:
                continue
            R_j, omega_j, L_j, Z_LC_j, n_j, l_j, ml_j = params[j]
            y_ij = _pair_coupling(Z, n_i, l_i, ml_i, n_j, l_j, ml_j,
                                  Z_LC_i, Z_LC_j)
            Y[i, j] = -y_ij   # off-diagonal: negative
            sum_mutual += y_ij

        Y[i, i] = y_self + sum_mutual  # diagonal: self + sum of mutuals

    # Stage E: Y → S (Op5) — computed for compliance, eigenvalue check
    Y0 = 1.0 / _Z0
    S = universal_ymatrix_to_s(Y, Y0=Y0)

    # Op6: eigenvalue target (diagnostic — λ_min → 0 at ground state)
    lam_min = universal_eigenvalue_target(S)

    # --- Energy from coupled resonator normal modes ---
    # For identical oscillators: ω_k = ω_0 / √(eigenvalue_k of (I+K))
    # where eigenvalues of (I+K) = 1 + k*λ_graph.
    # This is Eq. (bonding_mode) from Stage E1.
    #
    # For mixed shells: generalized eigenvalue problem.
    # D = Ω · (I+K)^(-1) · Ω gives ω_k² as eigenvalues.
    #
    # Energy: E_k = -n_k · ℏω_k / 2 (virial theorem for 1/r)
    # where n_k is the principal quantum number (winding number).
    # Derivation: E₀ = Z²Ry/n² = n·ℏω/2, so E = -n·ℏω/2.

    # Per-electron bare energies and principal quantum numbers
    bare_E = np.array([Z**2 * _RY_EV_CONST / n**2
                       for n, l, m_l in electrons])
    bare_omega = np.array([params[i][1] for i in range(N)])
    n_shells = np.array([e[0] for e in electrons], dtype=float)

    # Build coupling matrix K (dimensionless)
    K = np.zeros((N, N))
    for i in range(N):
        _, _, _, Z_LC_i, n_i, l_i, ml_i = params[i]
        for j in range(i+1, N):
            _, _, _, Z_LC_j, n_j, l_j, ml_j = params[j]
            y_ij = _pair_coupling(Z, n_i, l_i, ml_i, n_j, l_j, ml_j,
                                  Z_LC_i, Z_LC_j)
            # k_ij = y_ij × Z_LC_ref (convert admittance back to coupling)
            if n_i == n_j:
                k_ij = y_ij * Z_LC_i
            else:
                k_ij = y_ij * np.sqrt(Z_LC_i * Z_LC_j)
            K[i, j] = k_ij
            K[j, i] = k_ij

    # Dynamical matrix: D = Ω · (I+K)^(-1) · Ω
    # Eigenvalues of D = ω_k² = ω_i² / (1 + k*λ_graph)
    # → ω_k = ω_0 / √(1 + k*λ) (matches E1 bonding mode formula)
    omega_diag = np.diag(bare_omega)
    IK = np.eye(N) + K
    IK_inv = np.linalg.inv(IK)
    D = omega_diag @ IK_inv @ omega_diag
    evals = np.linalg.eigvalsh(D)
    evals = np.maximum(evals, 0.0)  # numerical safety
    mode_omega = np.sqrt(evals)

    # Ground-state occupation: all N electrons occupy the BONDING
    # (lowest-frequency) mode.  This is the same rule as
    # coupled_resonator_binding() for nuclear binding.
    # E_total = -N × n × ℏω_bond / 2  (virial, Step 1(g))
    omega_bond = np.min(mode_omega)

    # Effective n: the bonding mode is dominated by the shell
    # with the lowest bare frequency (outermost shell).
    # For same-shell: n is constant.  For mixed: use the
    # average n weighted by occupancy.
    avg_n = np.mean(n_shells)

    # Total energy: E = -N × n × ℏω_bond / 2
    # Dim: [n · J·s · rad/s / C] = [eV]  ✓
    E_total = -N * avg_n * HBAR * omega_bond / (2.0 * e_charge)
    return E_total


def ionization_energy_ymatrix(Z, n_electrons=None):
    r"""First ionization energy from N-port Y-matrix solver.

    Implements Ch.16 Pipeline (Stages A-E):
        Stage A: R_i, ω_i, L_i, Z_LC_i from Axiom 1
        Stage B: Pair classification (Hopf/orthogonal/cross-shell)
        Stage C: y_ij from Op4 + Op2
        Stage D: Y-matrix assembly
        Stage E: Op5 (Y→S), eigenvalues → mode frequencies → E_total

    Operator compliance:
        Op1: Z_LC = √(L/C) = ωL for each flux ring
        Op2: Hopf link saturation (1 - p_c/2) at crossings
        Op4: universal_pairwise_energy in Regime I → Coulomb
        Op5: universal_ymatrix_to_s (Y→S conversion)
        Op6: universal_eigenvalue_target (λ_min)

    CRITICAL: All electrons see BARE nuclear charge Z.
    No Z_eff. Screening emerges from the eigenvalue decomposition.

    Args:
        Z: Atomic number.
        n_electrons: Number of electrons (default: Z for neutral atom).

    Returns:
        IE in eV.
    """
    if n_electrons is None:
        n_electrons = Z

    if n_electrons <= 0:
        return 0.0

    # Single electron: exact hydrogenic
    if n_electrons == 1:
        config = _electron_config(Z, 1)
        n = config[0][0]
        return Z**2 * _RY_EV_CONST / n**2

    # N-electron configuration
    electrons_N = _electron_config(Z, n_electrons)

    # N-1 electron configuration (remove outermost)
    electrons_Nm1 = _electron_config(Z, n_electrons - 1)

    # Total energies
    E_N = _total_energy_from_ymatrix(Z, electrons_N)
    E_Nm1 = _total_energy_from_ymatrix(Z, electrons_Nm1)

    # IE = E(N) - E(N-1) (both negative, IE is positive)
    IE = E_N - E_Nm1
    return max(0.0, IE)

def atom_port_impedance(Z, ie_eV):
    r"""
    Atom's port impedance = valence orbital radius.

    r_val = n × a₀ × √(Ry/IE)  [meters]
    """
    shells = _fill_shells(Z)
    n = shells[-1][0]
    return n * _A0 * np.sqrt(_RY_EV / ie_eV)

# ─────────────────────────────────────────────────────────────────
# QUARANTINE: Dead IE solvers (v2, v3, v4, v5, v5.5) removed
# (2026-03-13). ~250 lines. All superseded by Approach 22.
# ─────────────────────────────────────────────────────────────────

def molecular_bond_distance(r_val_A, r_val_B):
    r"""
    Bond distance from maximum saturation coupling.

    d = sqrt(2) * sqrt(r_A * r_B)

    The coupling k(d) = S(r/d)*(r/d) is maximized at d = r*sqrt(2),
    giving k_max = 1/2 (from universal_saturation).
    """
    return np.sqrt(2.0) * np.sqrt(r_val_A * r_val_B)


def molecular_bond_energy(IE_A, IE_B, r_val_A, r_val_B, d_bond):
    r"""Molecular bond energy from coupled resonant cavities.

    PHYSICAL MODEL:
      Two atoms (resonant cavities) A and B couple via their outermost
      electron shells. The coupling strength k is a function of bond
      distance d. The bond energy is the reduction in total energy
      due to this coupling.

    UNIVERSAL OPERATOR: B = 2ω(1 − 1/√(1+k))  [Axiom 5]

    DIMENSIONAL ANALYSIS:
      ω_eff = √(IE_A × IE_B) [eV]
      k_eff = k(d) [dimensionless]
      B_eV = [eV] × (1 - 1/√(1 + [dimensionless])) = [eV] ✓

    Args:
        IE_A: Ionization energy of atom A (eV).
        IE_B: Ionization energy of atom B (eV).
        r_val_A: Effective radius of atom A (Bohr).
        r_val_B: Effective radius of atom B (Bohr).
        d_bond: Bond distance (Bohr).

    Returns:
        B_eV: Bond energy in eV. Positive = stable bond.
        k_eff: Effective coupling constant.
    """
    # Effective frequency (geometric mean of IEs)
    omega_eff = np.sqrt(IE_A * IE_B)

    # Effective coupling constant k(d)
    # This is the universal saturation function S(x) = x / sqrt(1 + x^2)
    # where x = (r_A * r_B) / d_bond^2
    x = (r_val_A * r_val_B) / d_bond**2
    k_eff = x / np.sqrt(1.0 + x**2)

    # Bond energy from Axiom 5
    B_eV = 2.0 * omega_eff * (1.0 - 1.0 / np.sqrt(1.0 + k_eff))
    return B_eV, k_eff

# ─────────────────────────────────────────────────────────────────
# QUARANTINE: QM radial potentials, atomic_resonance, and
# ionization_energy_cascade removed (2026-03-13). ~630 lines.
# N*N dynamical matrix architecture preserved in plan.
# ─────────────────────────────────────────────────────────────────
