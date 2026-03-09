"""
Universal Coupled Resonator Solver
==================================

A single framework applied at every scale of the AVE hierarchy.
The same five operators appear at each level:

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

========  ==========  =================  ==========  ====================
Scale     r_sat       omega_0            k           B formula
========  ==========  =================  ==========  ====================
Nuclear   d_p         c(1+nu)/d_p        2*alpha     N*hbar(w0-w_bond)
                      = 302 MeV/hbar     = 0.01476   (EM / photon exch.)
--------  ----------  -----------------  ----------  --------------------
Atomic    a0          Ry/hbar            screening   IE = Ry * Z_eff^2/n^2
                      = 3.29e15 rad/s    from E_c    (same Coulomb op.)
--------  ----------  -----------------  ----------  --------------------
Molecul.  r_val(IE)   sqrt(IE_A*IE_B)    S*x = 1/2   2*w_eff*(1-1/V1.5)
                      (geometric mean)   at d=rV2    (matter exchange)
========  ==========  =================  ==========  ====================

Key insight: the Coulomb operator E_c = (3/5)*Z(Z-1)*alpha*hbar*c/R
appears TWICE:
  - Nuclear scale: Coulomb repulsion between Z protons
  - Atomic scale:  Coulomb screening between Z electrons
Same operator, same fields, different particles.
"""

import numpy as np
from ave.core.constants import (
    ALPHA, HBAR, C_0, e_charge, D_PROTON, NU_VAC,
    M_P_MEV, M_N_MEV, HBAR_C_MEV_FM,
    K_COUPLING as _K_COUPLING_CONST,
    OMEGA_0_NUCLEAR as _OMEGA_0_CONST,
    E_0_NUCLEAR as _E_0_CONST,
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
_M_E = 9.1093837015e-31   # electron mass [kg]
_A0 = HBAR / (_M_E * C_0 * ALPHA)   # Bohr radius [m]
_RY_EV = _M_E * C_0**2 * ALPHA**2 / (2.0 * e_charge)   # Rydberg [eV]
_TWO_RY = 2.0 * _RY_EV   # e²/(4πε₀a₀) = Hartree ≈ 27.21 eV
_PROJECTION = 1.0 / (2.0 * (1.0 + 2.0/7.0))  # 7/18 projection of eigenvalue

# 3D electrostatic constants for exponential charge distributions.
# These are geometric constants of 3-dimensional space, computable
# from the integral <1/r₁₂> for charge distributions confined by
# saturation (evanescent fields beyond the orbital boundary).
# The 3D-ness is a consequence of ν = 2/7 (Axiom 3).
_J_1S_1S = 5.0 / 8.0    # <1/r₁₂> for two 1s distributions
_J_1S_2X = 17.0 / 81.0  # <1/r₁₂> for 1s × 2s/2p
_J_2X_2X = 77.0 / 512.0 # <1/r₁₂> for two 2s/2p distributions


def _coulomb_J(n1, n2, z_eff):
    r"""
    Electron-electron Coulomb integral in the LINEAR regime.

    In the linear regime (A = α ≪ 1), the pairwise potential is
    U = -K/r (pure Coulomb). The screening between two electrons
    in orbitals n₁, n₂ is:

        J = <1/r₁₂> × e²/(4πε₀) = geometric_factor × Z_eff × 2Ry

    The geometric factors (5/8, 17/81, 77/512) are constants of
    3D integration for exponentially confined charge distributions.

    Returns J in eV.
    """
    if n1 == 1 and n2 == 1:
        return _J_1S_1S * z_eff * _TWO_RY
    elif (n1 == 1 and n2 == 2) or (n1 == 2 and n2 == 1):
        return _J_1S_2X * z_eff * _TWO_RY
    elif n1 == 2 and n2 == 2:
        return _J_2X_2X * z_eff * _TWO_RY
    else:
        return _J_1S_1S * z_eff * _TWO_RY / (n1 * n2)


def atom_total_energy(Z, shells, z_eff_list):
    r"""
    Total energy of N electrons in a nuclear field Z.

    E = Σᵢ (Z_eff² - 2Z × Z_eff) × Ry/nᵢ² + Σᵢ<ⱼ Jᵢⱼ

    First term: kinetic + nuclear (one-electron, from eigenvalue)
    Second term: electron-electron repulsion (linear regime Coulomb)

    Args:
        Z: Nuclear charge
        shells: [(n, count), ...]
        z_eff_list: Z_eff per shell

    Returns:
        E_total in eV (negative = bound)
    """
    E = 0.0
    electrons = []
    for idx, (n, count) in enumerate(shells):
        ze = z_eff_list[idx]
        for _ in range(count):
            E += (ze**2 - 2.0 * Z * ze) * _RY_EV / n**2
            electrons.append((n, ze))

    for i in range(len(electrons)):
        for j in range(i + 1, len(electrons)):
            n_i, ze_i = electrons[i]
            n_j, ze_j = electrons[j]
            E += _coulomb_J(n_i, n_j, (ze_i + ze_j) / 2.0)

    return E


def _analytical_z_eff(Z, shells):
    r"""
    Z_eff per shell from dE/dZ_eff = 0 (linear equation).

    The derivative of E_total w.r.t. Z_eff for shell s:
        dE/dZ_s = count_s × 2(Z_s - Z) × Ry/n_s² + Σ_partners dJ/dZ_s

    Setting to zero gives Z_eff_s as a LINEAR function of Z.
    No optimizer needed — this is a direct algebraic solution.

    For He: Z_eff = Z - 5/16 = 27/16 = 1.6875 [exact known result]
    """
    z_eff = []
    for s, (n_s, count_s) in enumerate(shells):
        dJ_sum = 0.0
        for s2, (n2, count2) in enumerate(shells):
            n_partners = (count2 - 1) if s2 == s else count2
            if n_partners <= 0:
                continue
            if n_s == 1 and n2 == 1:
                j_deriv = _J_1S_1S * 2.0
            elif (n_s == 1 and n2 == 2) or (n_s == 2 and n2 == 1):
                j_deriv = _J_1S_2X * 2.0
            elif n_s == 2 and n2 == 2:
                j_deriv = _J_2X_2X * 2.0
            else:
                j_deriv = _J_1S_1S * 2.0 / (n_s * n2)
            dJ_sum += n_partners * j_deriv
        screening = (n_s**2 / (2.0 * count_s)) * dJ_sum
        z_eff.append(max(0.1, Z - screening))
    return z_eff


def _fill_shells(n_electrons):
    """Fill electron shells: [(n, count), ...]."""
    shells = []
    remaining = n_electrons
    for n in range(1, 8):
        capacity = 2 * n**2
        count = min(remaining, capacity)
        if count > 0:
            shells.append((n, count))
            remaining -= count
        if remaining <= 0:
            break
    return shells


def ionization_energy(Z, n_electrons=None):
    r"""
    First ionization energy from total energy difference.

    IE = E_total(Z, N-1 electrons) - E_total(Z, N electrons)

    Both energies computed with all Z protons present.
    The ion has Z protons and N-1 electrons.

    Total energy uses 3D electrostatic Coulomb integrals, justified
    by the regime analysis: electrons interact with strain
    A = ℓ_node/a₀ = α ≈ 0.007, deep in the LINEAR regime where
    U = -K/r (pure Coulomb).

    Args:
        Z: Atomic number
        n_electrons: Number of electrons (default = Z)

    Returns:
        IE_eV: First ionization energy [eV]
    """
    if n_electrons is None:
        n_electrons = Z

    if n_electrons <= 1:
        return Z**2 * _RY_EV

    neutral = _fill_shells(n_electrons)
    ze_neutral = _analytical_z_eff(Z, neutral)
    E_neutral = atom_total_energy(Z, neutral, ze_neutral)

    ion = list(neutral)
    n_last, count_last = ion[-1]
    if count_last > 1:
        ion[-1] = (n_last, count_last - 1)
    else:
        ion = ion[:-1]

    if len(ion) == 0:
        return -E_neutral

    ze_ion = _analytical_z_eff(Z, ion)
    E_ion = atom_total_energy(Z, ion, ze_ion)

    return E_ion - E_neutral


def atom_port_impedance(Z, ie_eV):
    r"""
    Atom's port impedance = valence orbital radius.

    r_val = n × a₀ × √(Ry/IE)  [meters]
    """
    shells = _fill_shells(Z)
    n = shells[-1][0]
    return n * _A0 * np.sqrt(_RY_EV / ie_eV)


def molecular_bond_distance(r_val_A, r_val_B):
    r"""
    Bond distance from maximum saturation coupling.

    d = √2 × √(r_A × r_B)

    The coupling k(d) = S(r/d)×(r/d) is maximized at d = r×√2,
    giving k_max = 1/2 (from universal_saturation).
    """
    return np.sqrt(2.0) * np.sqrt(r_val_A * r_val_B)


def molecular_bond_energy(IE_A_eV, IE_B_eV, n_bonds=1):
    r"""
    Bond energy from coupled eigenvalue splitting.

    B = 2×√(IE_A×IE_B)×(1-1/√(1+k)), k=1/2 at max coupling.
    """
    k_single = 0.5
    k_eff = n_bonds * k_single / (1.0 + (n_bonds - 1) * k_single)
    omega_eff = np.sqrt(IE_A_eV * IE_B_eV)
    B_eV = 2.0 * omega_eff * (1.0 - 1.0 / np.sqrt(1.0 + k_eff))
    return B_eV, k_eff






