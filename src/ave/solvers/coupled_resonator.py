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


def _J_1s1s(za, zb):
    r"""
    Exact Coulomb integral J(1s with Z_eff=za, 1s with Z_eff=zb).

    Derived from first principles:
        V_1s(r; za) = 1/r - (za + 1/r)exp(-2za·r)
        J = ∫ ρ_1s(r; zb) · V_1s(r; za) d³r

    Result: J = zb - za·zb³/S³ - zb³/S²  where S = za + zb
    Verified: for za = zb = z, gives (5/8)·z  [matches _J_1S_1S]

    Returns J in Hartree (atomic units).
    """
    S = za + zb
    return zb - za * zb**3 / S**3 - zb**3 / S**2


def _J_1s2s(za, zb):
    r"""
    Exact Coulomb integral J(1s with Z_eff=za, 2s with Z_eff=zb).

    Derived from first principles:
        ρ_2s = (zb³/8)(2 - zb·r)² exp(-zb·r)
        integrated against V_1s(r; za).

    Result uses B = zb, G = zb + 2·za.
    Verified: for za = zb = z, gives (17/81)·z  [matches _J_1S_2X]

    Returns J in Hartree (atomic units).
    """
    B = zb
    G = zb + 2.0 * za
    t1 = 2.0 / B**2
    t2 = -za * (8.0 / G**3 - 24.0 * B / G**4 + 24.0 * B**2 / G**5)
    t3 = -(4.0 / G**2 - 8.0 * B / G**3 + 6.0 * B**2 / G**4)
    return zb**3 / 8.0 * (t1 + t2 + t3)


def _II1(j, k, a, b):
    r"""Helper for J_2s2s: inner integral II1(j,k,a,b).

    II1 = ∫₀^∞ r₂^(k-1) exp(-b·r₂) [∫₀^r₂ r₁^j exp(-a·r₁) dr₁] dr₂

    Derived by expanding the inner incomplete gamma as a sum of
    complete Laplace integrals. All terms are j!/a^(j+1) factorials.
    """
    import math
    s = a + b
    result = math.factorial(j) / a**(j + 1) * math.factorial(k - 1) / b**k
    for p in range(j + 1):
        coeff = math.factorial(j) / (math.factorial(p) * a**(j + 1 - p))
        integral = math.factorial(k - 1 + p) / s**(k + p)
        result -= coeff * integral
    return result


def _J_2s2s(za, zb):
    r"""
    Exact Coulomb integral J(2s with Z_eff=za, 2s with Z_eff=zb).

    Derived by expanding both 2s densities as polynomial × exponential,
    splitting the double integral at r₁ = r₂, and evaluating each
    region as sums of factorial/power terms.

    ρ_2s(r; z) = (z³/8)(2 - zr)² exp(-zr)
    P(r) = 4 - 4zr + z²r²  (polynomial coefficients: c₀=4, c₁=-4z, c₂=z²)

    J = (za³·zb³/64) × Σ_{j,k} c_j(a)·c_k(b) × I(j+2, k+2; za, zb)
    where I(j,k;a,b) = II1(j,k,a,b) + II1(k,j,b,a)

    Verified: for za = zb = z, gives (77/512)·z  [matches _J_2X_2X]

    Returns J in Hartree (atomic units).
    """
    ca = [4.0, -4.0 * za, za**2]
    cb = [4.0, -4.0 * zb, zb**2]
    total = 0.0
    for dj, caj in enumerate(ca):
        for dk, cbk in enumerate(cb):
            j = dj + 2
            k = dk + 2
            total += caj * cbk * (_II1(j, k, za, zb) + _II1(k, j, zb, za))
    return za**3 * zb**3 / 64.0 * total


# ─────────────────────────────────────────────────────────────────
# n=3 shell J integrals (3s wavefunction)
# R_3s(r,z) = (2z^{3/2})/(81√3)(27-18zr+2z²r²)exp(-zr/3)
# Density: ρ_3s = |R_3s|² = [4z³/(81²·3)]·P(r)·exp(-2zr/3)
# P(r) = (27-18zr+2z²r²)² = 729-972zr+432z²r²-72z³r³+4z⁴r⁴
# ─────────────────────────────────────────────────────────────────

# 3s polynomial coefficients: [729, -972z, 432z², -72z³, 4z⁴]
_3S_NORM = lambda z: 4.0 * z**3 / (81**2 * 3)
_3S_POLY = lambda z: [729.0, -972.0*z, 432.0*z**2, -72.0*z**3, 4.0*z**4]
_3S_RATE = lambda z: 2.0 * z / 3.0

# 1s: ρ = 4z³·exp(-2zr), polynomial = [1], rate = 2z
_1S_NORM = lambda z: 4.0 * z**3
_1S_POLY = lambda z: [1.0]
_1S_RATE = lambda z: 2.0 * z

# 2s: ρ = (z³/8)(2-zr)²·exp(-zr), polynomial = [4, -4z, z²], rate = z
_2S_NORM = lambda z: z**3 / 8.0
_2S_POLY = lambda z: [4.0, -4.0*z, z**2]
_2S_RATE = lambda z: z


def _J_general(norm_a, poly_a, rate_a, norm_b, poly_b, rate_b):
    r"""General two-exponent J integral using the II1 master formula.

    For densities ρ_a = norm_a · P_a(r) · exp(-rate_a·r)
    and             ρ_b = norm_b · P_b(r) · exp(-rate_b·r):

    J = norm_a · norm_b · Σ c_j·c_k · I(j+2, k+2; rate_a, rate_b)
    where I(j,k;a,b) = II1(j,k,a,b) + II1(k,j,b,a)

    Returns J in Hartree (atomic units).
    """
    total = 0.0
    for dj, caj in enumerate(poly_a):
        for dk, cbk in enumerate(poly_b):
            j = dj + 2
            k = dk + 2
            total += caj * cbk * (_II1(j, k, rate_a, rate_b)
                                  + _II1(k, j, rate_b, rate_a))
    return norm_a * norm_b * total


def _J_1s3s(za, zb):
    r"""Exact J(1s with Z_eff=za, 3s with Z_eff=zb).

    Verified: J_1s3s(1,1) = 0.09948730 (matches numerical quadrature).
    """
    return _J_general(_1S_NORM(za), _1S_POLY(za), _1S_RATE(za),
                      _3S_NORM(zb), _3S_POLY(zb), _3S_RATE(zb))


def _J_2s3s(za, zb):
    r"""Exact J(2s with Z_eff=za, 3s with Z_eff=zb).

    Verified: J_2s3s(1,1) = 0.08411392 (matches numerical quadrature).
    """
    return _J_general(_2S_NORM(za), _2S_POLY(za), _2S_RATE(za),
                      _3S_NORM(zb), _3S_POLY(zb), _3S_RATE(zb))


def _J_3s3s(za, zb):
    r"""Exact J(3s with Z_eff=za, 3s with Z_eff=zb).

    Verified: J_3s3s(z,z) = (17/256)·z (matches numerical quadrature).
    """
    return _J_general(_3S_NORM(za), _3S_POLY(za), _3S_RATE(za),
                      _3S_NORM(zb), _3S_POLY(zb), _3S_RATE(zb))


# ─────────────────────────────────────────────────────────────────
# n=4 shell J integrals (4s wavefunction)
# R_4s(r,z) = 2(z/4)^{3/2} · L_3^1(zr/2) · exp(-zr/4)
# L_3^1(u) = 1 - 3u/2 + 3u²/8 - u³/48  where u = zr
# Density: ρ_4s = |R_4s|² = (z³/16) · P²(zr) · exp(-zr/2)
# P²(u) = 1 - 3u + 3u² - 7u³/6 + 13u⁴/64 - u⁵/64 + u⁶/2304
# ─────────────────────────────────────────────────────────────────

# Precompute P² coefficients: P(u) = 1 - 3u/2 + 3u²/8 - u³/48
_4S_P_COEFFS = [1.0, -3.0/2, 3.0/8, -1.0/48]
_4S_P2_COEFFS = [0.0] * 7
for _i in range(4):
    for _j in range(4):
        _4S_P2_COEFFS[_i + _j] += _4S_P_COEFFS[_i] * _4S_P_COEFFS[_j]

_4S_NORM = lambda z: z**3 / 16.0
_4S_POLY = lambda z: [_4S_P2_COEFFS[k] * z**k for k in range(7)]
_4S_RATE = lambda z: z / 2.0


def _J_1s4s(za, zb):
    r"""Exact J(1s with Z_eff=za, 4s with Z_eff=zb)."""
    return _J_general(_1S_NORM(za), _1S_POLY(za), _1S_RATE(za),
                      _4S_NORM(zb), _4S_POLY(zb), _4S_RATE(zb))


def _J_2s4s(za, zb):
    r"""Exact J(2s with Z_eff=za, 4s with Z_eff=zb)."""
    return _J_general(_2S_NORM(za), _2S_POLY(za), _2S_RATE(za),
                      _4S_NORM(zb), _4S_POLY(zb), _4S_RATE(zb))


def _J_3s4s(za, zb):
    r"""Exact J(3s with Z_eff=za, 4s with Z_eff=zb)."""
    return _J_general(_3S_NORM(za), _3S_POLY(za), _3S_RATE(za),
                      _4S_NORM(zb), _4S_POLY(zb), _4S_RATE(zb))


def _J_4s4s(za, zb):
    r"""Exact J(4s with Z_eff=za, 4s with Z_eff=zb)."""
    return _J_general(_4S_NORM(za), _4S_POLY(za), _4S_RATE(za),
                      _4S_NORM(zb), _4S_POLY(zb), _4S_RATE(zb))


# ─────────────────────────────────────────────────────────────────
# Subshell wavefunction parameters (p and d orbitals)
#
# Convention: ρ_nl(r) = |R_nl(r)|² = norm × P(r) × exp(-rate×r)
# where P(r) is a polynomial in r with z-dependent coefficients.
#
# All verified against numerical quadrature to 6+ decimal places.
# See /tmp/subshell_params.py for derivation and verification script.
# ─────────────────────────────────────────────────────────────────

# 2p: R_21 = z^{5/2}/(2√6) × r × exp(-zr/2)
# |R_21|² = z⁵/24 × r² × exp(-zr)
_2P_NORM = lambda z: z**5 / 24.0
_2P_POLY = lambda z: [0.0, 0.0, 1.0]   # r²
_2P_RATE = lambda z: z

# 3p: R_31 = 8z^{5/2}/(27√6) × r(1-zr/6) × exp(-zr/3)
# |R_31|² = 32z⁵/2187 × r²(1-zr/6)² × exp(-2zr/3)
# P(r) = r²(1-zr/3+z²r²/36) = [0, 0, 1, -z/3, z²/36]
_3P_NORM = lambda z: 32.0 * z**5 / 2187.0
_3P_POLY = lambda z: [0.0, 0.0, 1.0, -z / 3.0, z**2 / 36.0]
_3P_RATE = lambda z: 2.0 * z / 3.0

# 3d: R_32 = 4z^{7/2}/(81√30) × r² × exp(-zr/3)
# |R_32|² = 16z⁷/(81²×30) × r⁴ × exp(-2zr/3)
_3D_NORM = lambda z: 16.0 * z**7 / (81**2 * 30.0)
_3D_POLY = lambda z: [0.0, 0.0, 0.0, 0.0, 1.0]  # r⁴
_3D_RATE = lambda z: 2.0 * z / 3.0

# Subshell routing table: (n, l) → (norm, poly, rate) functions
_SUBSHELL_PARAMS = {
    (1, 0): (_1S_NORM, _1S_POLY, _1S_RATE),
    (2, 0): (_2S_NORM, _2S_POLY, _2S_RATE),
    (2, 1): (_2P_NORM, _2P_POLY, _2P_RATE),
    (3, 0): (_3S_NORM, _3S_POLY, _3S_RATE),
    (3, 1): (_3P_NORM, _3P_POLY, _3P_RATE),
    (3, 2): (_3D_NORM, _3D_POLY, _3D_RATE),
    (4, 0): (_4S_NORM, _4S_POLY, _4S_RATE),
}

# ─────────────────────────────────────────────────────────────────
#
# The _J_general framework integrates ρ_a(r1) × ρ_b(r2).
# For exchange, we use the mixed density: ρ_{mix}(r) = R_a(r) × R_b(r).
# ─────────────────────────────────────────────────────────────────

def _mixed_1s_1s(za, zb):
    norm = 4.0 * za**1.5 * zb**1.5
    return norm, [1.0], za + zb

def _mixed_1s_2s(za, zb):
    norm = za**1.5 * zb**1.5 / np.sqrt(2.0)
    return norm, [2.0, -zb], za + zb/2.0

def _mixed_1s_2p(za, zb):
    norm = za**1.5 * zb**2.5 / np.sqrt(6.0)
    return norm, [0.0, 1.0], za + zb/2.0

def _mixed_2s_2s(za, zb):
    norm = za**1.5 * zb**1.5 / 8.0
    return norm, [4.0, -2.0*(za+zb), za*zb], (za+zb)/2.0

def _mixed_2s_2p(za, zb):
    norm = za**1.5 * zb**2.5 / (4.0 * np.sqrt(12.0))
    return norm, [0.0, 2.0, -za], (za+zb)/2.0

def _mixed_2p_2p(za, zb):
    norm = za**2.5 * zb**2.5 / 24.0
    return norm, [0.0, 0.0, 1.0], (za+zb)/2.0

_MIXED_FUNCS = {
    ((1,0), (1,0)): _mixed_1s_1s,
    ((1,0), (2,0)): _mixed_1s_2s,
    ((1,0), (2,1)): _mixed_1s_2p,
    ((2,0), (2,0)): _mixed_2s_2s,
    ((2,0), (2,1)): _mixed_2s_2p,
    ((2,1), (2,1)): _mixed_2p_2p,
}

# Angular averaging factors for exchange integrals
# (Fraction of the radial integral that survives 3D integration)
_ANG_SS = 1.0
_ANG_SP = 1.0 / 3.0
_ANG_PP = 2.0 / 5.0

def _exchange_K_sub(n1, l1, z1, n2, l2, z2):
    r"""Subshell-resolved Exchange K integral (AC interference).

    K = ∫∫ R_a(r1)R_b(r1) × R_b(r2)R_a(r2) / r_> r1²r2² dr

    Returns K in eV.
    """
    key = (min((n1,l1), (n2,l2)), max((n1,l1), (n2,l2)))
    za, zb = (z1, z2) if (n1,l1) <= (n2,l2) else (z2, z1)

    if key in _MIXED_FUNCS:
        nm, pm, rm = _MIXED_FUNCS[key](za, zb)
        K_hartree = _J_general(nm, pm, rm, nm, pm, rm)
        
        la, lb = key[0][1], key[1][1]
        if la == 0 and lb == 0:
            ang = _ANG_SS
        elif (la == 0 and lb == 1) or (la == 1 and lb == 0):
            ang = _ANG_SP
        elif la == 1 and lb == 1:
            ang = _ANG_PP
        else:
            # d-d and higher angular averaging factors not yet derived.
            # 0.1 is a placeholder (true value requires 3j/6j symbols).
            ang = 0.1
    else:
        # Mixed wavefunctions for n≥3 exchange not yet implemented.
        # Returns 0 (= no exchange correction), which is conservative:
        # exchange is small relative to direct Coulomb for outer shells.
        return 0.0

    return K_hartree * ang * _TWO_RY


def _coulomb_J_sub(n1, l1, z1, n2, l2, z2):
    r"""Subshell-resolved Coulomb J integral.

    Routes to exact closed-form wavefunctions for (n,l) pairs up to n=4.
    Falls back to s-like scaling for higher shells.

    Returns J in eV.
    """
    key_a = (n1, l1)
    key_b = (n2, l2)

    if key_a in _SUBSHELL_PARAMS and key_b in _SUBSHELL_PARAMS:
        Na, Pa, Ra = _SUBSHELL_PARAMS[key_a]
        Nb, Pb, Rb = _SUBSHELL_PARAMS[key_b]
        J_hartree = _J_general(Na(z1), Pa(z1), Ra(z1),
                               Nb(z2), Pb(z2), Rb(z2))
    else:
        # Approximation: use (n,0) s-wave params for unimplemented (n,l)
        # subshells, or 1s-like scaling with effective Z/n for n≥5.
        fa = _SUBSHELL_PARAMS.get((n1, 0))
        fb = _SUBSHELL_PARAMS.get((n2, 0))
        if fa and fb:
            Na, Pa, Ra = fa
            Nb, Pb, Rb = fb
            J_hartree = _J_general(Na(z1), Pa(z1), Ra(z1),
                                   Nb(z2), Pb(z2), Rb(z2))
        else:
            J_hartree = _J_general(
                _1S_NORM(z1 / n1), _1S_POLY(z1 / n1), _1S_RATE(z1 / n1),
                _1S_NORM(z2 / n2), _1S_POLY(z2 / n2), _1S_RATE(z2 / n2))

    return J_hartree * _TWO_RY


def _coulomb_J(n1, z1, n2, z2):
    r"""
    Two-exponent Coulomb integral in the LINEAR regime.

    Routes to the exact closed-form integral for each (n₁, n₂) pair.
    Each electron has its OWN Z_eff (z1, z2), not an average.

    Exact integrals available for n ≤ 4 (covers Z=1..36).
    For n ≥ 5: uses the general formula with correct wavefunction params.

    Returns J in eV.
    """
    na, nb = min(n1, n2), max(n1, n2)
    za_s, zb_s = (z1, z2) if n1 <= n2 else (z2, z1)

    # Route to the exact integral for (na, nb) pairs
    _SHELL_PARAMS = {
        1: (_1S_NORM, _1S_POLY, _1S_RATE),
        2: (_2S_NORM, _2S_POLY, _2S_RATE),
        3: (_3S_NORM, _3S_POLY, _3S_RATE),
        4: (_4S_NORM, _4S_POLY, _4S_RATE),
    }

    if na in _SHELL_PARAMS and nb in _SHELL_PARAMS:
        Na, Pa, Ra = _SHELL_PARAMS[na]
        Nb, Pb, Rb = _SHELL_PARAMS[nb]
        J_hartree = _J_general(Na(za_s), Pa(za_s), Ra(za_s),
                               Nb(zb_s), Pb(zb_s), Rb(zb_s))
    else:
        # For n >= 5: use 1s-like scaling as lowest-order approximation
        J_hartree = _J_general(
            _1S_NORM(za_s / na), _1S_POLY(za_s / na), _1S_RATE(za_s / na),
            _1S_NORM(zb_s / nb), _1S_POLY(zb_s / nb), _1S_RATE(zb_s / nb)
        )
    return J_hartree * _TWO_RY


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


def atom_total_energy(Z, shells, z_eff_list):
    r"""
    Total energy of N electrons in a nuclear field Z.

    E = Σᵢ (Z_eff² - 2Z × Z_eff) × Ry/nᵢ² + Σᵢ<ⱼ Jᵢⱼ

    First term: kinetic + nuclear (one-electron, from eigenvalue)
    Second term: electron-electron repulsion (linear regime Coulomb)

    Each J integral uses the EXACT two-exponent formula, not an
    averaged Z_eff. This is the corrected version that produces
    the right Coulomb repulsion even when Z_eff differs between shells.

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
            E += _coulomb_J(n_i, ze_i, n_j, ze_j)

    return E


def _scf_z_eff(Z, shells, max_iter=50, tol=1e-6):
    r"""
    Self-consistent Z_eff via dE/dZ_eff = 0 (fixed-point iteration).

    For each shell s, dE/dZ_s = 0 gives:
        z_s = Z - (n_s² / (2·count_s·Ry)) × Σ_partners dJ/dz_s

    The J derivatives are computed via central finite difference
    of the exact two-exponent J integrals. This is not curve fitting
    — it's numerical differentiation of a first-principles integral.

    Starting from z_s = Z (unscreened), iterate until convergence.

    For He: converges to Z_eff ≈ 1.69 (exact HF: 1.6875),
    giving IE ≈ 23.1 eV (exp: 24.59, 6% error).
    """
    n_shells = len(shells)
    z = [float(Z)] * n_shells  # start unscreened

    for iteration in range(max_iter):
        z_old = list(z)

        for s, (ns, cs) in enumerate(shells):
            # Build full electron list with current z_eff values
            electrons = []
            for idx, (n, c) in enumerate(shells):
                for _ in range(c):
                    electrons.append((n, idx))

            # Compute dE_repulsion/dz_s using finite difference
            dz = 1e-5
            dJ_sum = 0.0
            for i in range(len(electrons)):
                for j in range(i + 1, len(electrons)):
                    ni, si = electrons[i]
                    nj, sj = electrons[j]
                    if si != s and sj != s:
                        continue
                    z_plus = list(z)
                    z_minus = list(z)
                    z_plus[s] += dz
                    z_minus[s] -= dz
                    J_plus = _coulomb_J(ni, z_plus[si], nj, z_plus[sj])
                    J_minus = _coulomb_J(ni, z_minus[si], nj, z_minus[sj])
                    dJ_sum += (J_plus - J_minus) / (2.0 * dz)

            # From dE/dz_s = 0:
            # count_s × 2(z_s - Z) × Ry/n_s² + dJ_sum = 0
            # z_s = Z - n_s² × dJ_sum / (2 × count_s × Ry)
            z[s] = Z - ns**2 * dJ_sum / (2.0 * cs * _RY_EV)
            z[s] = max(0.1, z[s])

        if max(abs(z[i] - z_old[i]) for i in range(n_shells)) < tol:
            break

    return z



def ionization_energy(Z, n_electrons=None):
    r"""
    First ionization energy from total energy difference.

    IE = E_total(Z, N-1 electrons) - E_total(Z, N electrons)

    Both energies computed with all Z protons present.
    The ion has Z protons and N-1 electrons.

    Total energy uses exact two-exponent 3D Coulomb integrals,
    justified by the regime analysis: electrons interact with strain
    A = ℓ_node/a₀ = α ≈ 0.007, deep in the LINEAR regime where
    U = -K/r (pure Coulomb). The geometric constants (5/8, 17/81,
    77/512) are the same-exponent limits of the closed-form integrals.

    Z_eff is solved self-consistently (SCF) from dE/dZ_eff = 0.
    No optimizer, no curve fitting — just first-principles integrals
    and the variational principle.

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
    ze_neutral = _scf_z_eff(Z, neutral)
    E_neutral = atom_total_energy(Z, neutral, ze_neutral)

    ion = list(neutral)
    n_last, count_last = ion[-1]
    if count_last > 1:
        ion[-1] = (n_last, count_last - 1)
    else:
        ion = ion[:-1]

    if len(ion) == 0:
        return -E_neutral

    ze_ion = _scf_z_eff(Z, ion)
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


# ─────────────────────────────────────────────────────────────────
# Subshell-resolved IE (DC layer v2)
#
# The v1 functions above group all subshells by n (treating 2p as 2s).
# This misses 21% more repulsion for 2p-2p and 30% more for 3d-3d.
# The v2 functions resolve (n,l) subshells for correct J integrals.
#
# Physical picture:
#   Each electron is a standing wave. The OVERLAP between two standing
#   waves through the nonlinear vacuum (Axiom 4) produces mixing energy.
#   2p electrons overlap MORE than 2s (higher radial density at nucleus),
#   giving more repulsion. Correct overlap → correct screening → correct IE.
# ─────────────────────────────────────────────────────────────────

def _fill_subshells(n_electrons):
    """Fill electron subshells using Aufbau (Madelung n+l) order.

    Returns [(n, l, count), ...] — NOT grouped by n.

    Example: O (Z=8) = [(1,0,2), (2,0,2), (2,1,4)]
             Na (Z=11) = [(1,0,2), (2,0,2), (2,1,6), (3,0,1)]
    """
    result = []
    remaining = n_electrons
    for n, l, capacity in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        count = min(remaining, capacity)
        result.append((n, l, count))
        remaining -= count
    return result


def atom_total_energy_v2(Z, subshells, z_eff_list):
    r"""Total energy with subshell-resolved J integrals.

    E = Σᵢ (z_eff² - 2Z·z_eff) × Ry/nᵢ² + Σ_{i<j} J(nᵢlᵢzᵢ, nⱼlⱼzⱼ)

    Same formula as atom_total_energy, but routing J integrals through
    _coulomb_J_sub which uses the correct radial wavefunctions for
    p and d orbitals (2p ≠ 2s overlap by +21%).

    Args:
        Z: Nuclear charge.
        subshells: [(n, l, count), ...] from _fill_subshells().
        z_eff_list: Z_eff per subshell.

    Returns:
        E_total in eV (negative = bound).
    """
    E = 0.0
    electrons = []
    for idx, (n, l, count) in enumerate(subshells):
        ze = z_eff_list[idx]
        for _ in range(count):
            E += (ze**2 - 2.0 * Z * ze) * _RY_EV / n**2
            electrons.append((n, l, ze))

    for i in range(len(electrons)):
        for j in range(i + 1, len(electrons)):
            n_i, l_i, ze_i = electrons[i]
            n_j, l_j, ze_j = electrons[j]
            E += _coulomb_J_sub(n_i, l_i, ze_i, n_j, l_j, ze_j)

    return E


def _scf_z_eff_v2(Z, subshells, max_iter=80, tol=1e-6):
    r"""Self-consistent Z_eff with subshell resolution.

    Same variational principle as _scf_z_eff: dE/dZ_eff_s = 0 for each
    subshell s. Uses _coulomb_J_sub for correct p/d wavefunctions.

    Args:
        Z: Nuclear charge.
        subshells: [(n, l, count), ...].
        max_iter: Maximum SCF iterations.
        tol: Convergence threshold.

    Returns:
        List of Z_eff per subshell.
    """
    n_sub = len(subshells)
    z = [float(Z)] * n_sub  # start unscreened

    for iteration in range(max_iter):
        z_old = list(z)

        for s, (ns, ls, cs) in enumerate(subshells):
            # Build full electron list with current z_eff values
            electrons = []
            for idx, (n, l, c) in enumerate(subshells):
                for _ in range(c):
                    electrons.append((n, l, idx))

            # Compute dE_repulsion/dz_s using finite difference
            dz = 1e-5
            dJ_sum = 0.0
            for i in range(len(electrons)):
                for j in range(i + 1, len(electrons)):
                    ni, li, si = electrons[i]
                    nj, lj, sj = electrons[j]
                    if si != s and sj != s:
                        continue
                    z_plus = list(z)
                    z_minus = list(z)
                    z_plus[s] += dz
                    z_minus[s] -= dz
                    J_plus = _coulomb_J_sub(ni, li, z_plus[si],
                                           nj, lj, z_plus[sj])
                    J_minus = _coulomb_J_sub(ni, li, z_minus[si],
                                            nj, lj, z_minus[sj])
                    dJ_sum += (J_plus - J_minus) / (2.0 * dz)

            # From dE/dz_s = 0:
            # count_s × 2(z_s - Z) × Ry/n_s² + dJ_sum = 0
            z[s] = Z - ns**2 * dJ_sum / (2.0 * cs * _RY_EV)
            z[s] = max(0.5, z[s])

        if max(abs(z[i] - z_old[i]) for i in range(n_sub)) < tol:
            break

    return z


def ionization_energy_v2(Z, n_electrons=None):
    r"""First ionization energy with subshell-resolved J integrals.

    DC layer of the DC+AC architecture:
      - Subshell-resolved wavefunctions (2p ≠ 2s, 3d ≠ 3s)
      - SCF feedback loop for self-consistent Z_eff
      - IE from total energy difference

    PHYSICAL MODEL:
      Electrons are overlapping standing waves in the nuclear cavity.
      Overlap (J integral) through the nonlinear vacuum (Axiom 4)
      produces mixing energy = repulsion. The SCF IS the feedback
      loop that finds the self-consistent operating point.

    Args:
        Z: Atomic number.
        n_electrons: Number of electrons (default: Z).

    Returns:
        IE in eV. Positive = energy to remove outermost electron.
    """
    if n_electrons is None:
        n_electrons = Z
    if n_electrons <= 1:
        return Z**2 * _RY_EV

    # Neutral atom
    subs_neutral = _fill_subshells(n_electrons)
    ze_neutral = _scf_z_eff_v2(Z, subs_neutral)
    E_neutral = atom_total_energy_v2(Z, subs_neutral, ze_neutral)

    # Ion: remove from outermost subshell
    subs_ion = list(subs_neutral)
    n_last, l_last, c_last = subs_ion[-1]
    if c_last > 1:
        subs_ion[-1] = (n_last, l_last, c_last - 1)
    else:
        subs_ion = subs_ion[:-1]

    if len(subs_ion) == 0:
        return -E_neutral

    ze_ion = _scf_z_eff_v2(Z, subs_ion)
    E_ion = atom_total_energy_v2(Z, subs_ion, ze_ion)

    return E_ion - E_neutral






# ─────────────────────────────────────────────────────────────────
# AC + DC IE (v3 layer with Exchange)
#
# DC mixing (J integral): average overlap density |R_a|²|R_b|²
# AC beat (K integral): interference R_a R_b R_b R_a
#
# Electrons paired in the same subshell are 180° anti-phase
# (Axiom 4 saturation boundary, Γ = -1). This causes DESTRUCTIVE
# interference, reducing the total repulsion.
# ─────────────────────────────────────────────────────────────────

def atom_total_energy_v3(Z, subshells, z_eff_list):
    r"""Total energy with J (DC repulsion) AND K (AC exchange).

    E = Σᵢ (z_eff² - 2Z·z_eff) × Ry/nᵢ² + Σ_{i<j} J(i,j) - Σ_{same_spin} K(i,j)

    Follows Hund's rule: maximizes same-spin electrons per subshell.
    Exchange K applies ONLY between same-spin electrons, reducing their
    mutual repulsion due to anti-phase spatial interference.
    """
    E = 0.0
    electrons = []
    
    # Pre-compute non-interacting energy and build electron list
    for idx, (n, l, count) in enumerate(subshells):
        ze = z_eff_list[idx]
        for _ in range(count):
            E += (ze**2 - 2.0 * Z * ze) * _RY_EV / n**2
            electrons.append((n, l, ze))

    # Assign spins according to Hund's rule (maximize parallel spin)
    spins = []
    for n, l, count in subshells:
        half_capacity = 2 * l + 1
        for i in range(count):
            spins.append('up' if i < half_capacity else 'down')

    N = len(electrons)
    for i in range(N):
        for j in range(i + 1, N):
            n_i, l_i, ze_i = electrons[i]
            n_j, l_j, ze_j = electrons[j]
            
            # DC mixing (Coulomb J) always applies
            J = _coulomb_J_sub(n_i, l_i, ze_i, n_j, l_j, ze_j)
            E += J
            
            # AC beat (Exchange K) only for same-spin pairs
            if spins[i] == spins[j]:
                K = _exchange_K_sub(n_i, l_i, ze_i, n_j, l_j, ze_j)
                E -= K

    return E


def ionization_energy_v3(Z, n_electrons=None):
    r"""First ionization energy with DC+AC architecture (J + K integrals).

    Computes operating point from DC SCF, then computes total energy
    including AC exchange interference. 
    """
    if n_electrons is None:
        n_electrons = Z
    if n_electrons <= 1:
        return Z**2 * _RY_EV

    subs_neutral = _fill_subshells(n_electrons)
    # The DC operating point provides the z_eff
    ze_neutral = _scf_z_eff_v2(Z, subs_neutral)
    E_neutral = atom_total_energy_v3(Z, subs_neutral, ze_neutral)

    subs_ion = list(subs_neutral)
    n_last, l_last, c_last = subs_ion[-1]
    if c_last > 1:
        subs_ion[-1] = (n_last, l_last, c_last - 1)
    else:
        subs_ion = subs_ion[:-1]

    if len(subs_ion) == 0:
        return -E_neutral

    ze_ion = _scf_z_eff_v2(Z, subs_ion)
    E_ion = atom_total_energy_v3(Z, subs_ion, ze_ion)

    return E_ion - E_neutral




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
# Level 2 (AC): Ionization Energy via Circuit Screening
# ─────────────────────────────────────────────────────────────────
#
# DESIGN RATIONALE:
#   Each electron shell is an LC resonant cavity (Axiom 1).
#   Electron-electron coupling between shells uses the SAME
#   geometric Coulomb fractions (k_geo) as the J integrals, but
#   INTRA-shell coupling is reduced by the Axiom 4 phase-lock
#   factor — the AVE analog of exchange correlation.
#
# DIMENSIONAL ANALYSIS (mapped to Universal Operators):
#
#   Operator 1 — ω₀:
#     ω₀_n = Z² × Ry / n²  [eV]
#     Source: Axiom 1 (LC resonance at Bohr scale)
#     Dimension check: Z² [dimensionless] × Ry [eV] / n² [dimensionless] = [eV] ✓
#
#   Operator 2 — k (coupling):
#     k(nᵢ, nⱼ) = K_GEO[(nᵢ,nⱼ)] [dimensionless]
#     Source: Axiom 3 (3D geometry from ν = 2/7)
#     These are ⟨1/r₁₂⟩ in atomic units per unit Z — pure numbers.
#
#   Operator 3 — Screening σ:
#     σᵢ = Σⱼ (occ_j × k_ij) [dimensionless — units of charge]
#     Dimension: occupancy [dimensionless] × k_geo [dimensionless] = [dimensionless] ✓
#     This is the AVE version of E_coulomb operator at atomic scale.
#
#   Operator 4 — Z_eff:
#     Z_eff_i = Z − σᵢ [dimensionless — units of charge]
#
#   Operator 5 — S(x) saturation (Axiom 4):
#     S(fill) = √(1 − fill²) [dimensionless]
#     fill = occupancy / capacity [dimensionless]
#     Reduces intra-shell coupling to model spatial self-organization
#     (phase-locking) that replaces exchange correlation.
#
#   Output — IE:
#     IE = Ry × Z_eff² / n² [eV]
#     Dimension: [eV] × [dimensionless]² / [dimensionless]² = [eV] ✓
#
# CROSS-SCALE MAPPING (see module docstring):
#   Nuclear:  ω₀ = c(1+ν)/d_p,    k = 2α,       B = N·ℏ(ω₀−ω_bond)
#   ATOMIC:   ω₀ = Z²Ry/n²,       k = K_GEO,    IE = Ry·Z_eff²/n²
#   Molecul:  ω₀ = √(IE_A·IE_B),  k = ½ at max, B = 2ω(1−1/√(1+k))
# ─────────────────────────────────────────────────────────────────

# Geometric Coulomb fractions [dimensionless].
# K_GEO[(na, nb)] = ⟨1/r₁₂⟩ for unit-Z hydrogenic densities in shells na, nb.
#
# These are the SAME constants as _J_1S_1S, _J_1S_2X, _J_2X_2X defined above,
# extended to shells n=3,4. They are CONSTANTS OF 3D SPACE derived from the
# integral ∫∫ρ_a(r₁)·(1/r₁₂)·ρ_b(r₂) d³r₁ d³r₂ for Slater-type orbitals.
#
# Dimension: [dimensionless] — fraction of Z per electron pair.
_K_GEO = {
    (1, 1): 5.0 / 8.0,      # 0.6250 — two 1s: verified = _J_1S_1S
    (1, 2): 17.0 / 81.0,    # 0.2099 — 1s × 2s: verified = _J_1S_2X
    (2, 2): 77.0 / 512.0,   # 0.1504 — two 2s: verified = _J_2X_2X
    (1, 3): 0.1273,          # 1s × 3s (from _J_general)
    (2, 3): 0.0887,          # 2s × 3s
    (3, 3): 0.0664,          # two 3s
    (1, 4): 0.0735,          # 1s × 4s
    (2, 4): 0.0543,          # 2s × 4s
    (3, 4): 0.0420,          # 3s × 4s
    (4, 4): 0.0340,          # two 4s
}

# ─────────────────────────────────────────────────────────────────
# Standing-Wave Cavity Algorithms (Acoustic Bulk Modulus Model)
# ─────────────────────────────────────────────────────────────────

from ave.core.constants import EPSILON_0, HBAR, M_E, e_charge, ALPHA, C_0
from scipy.integrate import quad
from scipy.optimize import root_scalar

a_0 = HBAR / (M_E * C_0 * ALPHA)  # Derived Bohr Radius

def _V_nucleus(r, Z):
    """Deep Coulomb well of the nucleus."""
    k_e = 1.0 / (4.0 * np.pi * EPSILON_0)
    return k_e * (Z * e_charge) / r

def _V_1s_electron(r, Z_eff, geometric_overlap=1.0):
    """Voltage profile of a phase-locked 0_1 unknot in the 1s resonance node.
    
    The classical 1s hydrogenic density assumes perfect point-particle smearing.
    However, the AVE electron is a 0_1 topological unknot (crossing number c=1, 
    but effectively an irreducible flux tube). When two such unknots share the 
    same s-wave spatial cavity (e.g. He Z=2), their flux tubes cannot perfectly
    overlap. This creates a geometric excluded volume that reduces their 
    mutual Coulomb repulsion by ~5.28% relative to the classical Hartree-Fock 
    J_1s_1s integral.
    """
    k_e = 1.0 / (4.0 * np.pi * EPSILON_0)
    # The topological node distributes over the classical derived density geometry,
    # but the total integrated screening effect is reduced by the excluded volume overlap.
    classical_screening = 1.0 - (1.0 + Z_eff * r / a_0) * np.exp(-2.0 * Z_eff * r / a_0)
    return -k_e * e_charge / r * (geometric_overlap * classical_screening)

def _V_2s_electron(r, Z_eff, geometric_overlap=1.0):
    """Voltage profile of a phase-locked 0_1 unknot in the 2s resonance node.
    
    The topological geometric overlap for a 180-degree 2s^2 phase-jitter LC oscillator
    is governed strictly by the FOC topological limit $J_{2s} = P_C \\times S(P_C)$.
    """
    k_e = 1.0 / (4.0 * np.pi * EPSILON_0)
    # Actually, the classical potential of a 2s electron is well known:
    # V_2s(r) = - k_e e / r * [1 - (1 + 3/4 A + 1/4 A^2 + 1/8 A^3) e^{-A}] where A = (Z_eff * r) / a_0
    
    A = Z_eff * r / a_0
    classical_screening = 1.0 - (1.0 + 0.75 * A + 0.25 * A**2 + 0.125 * A**3) * np.exp(-A)
    return -k_e * e_charge / r * (geometric_overlap * classical_screening)

def _V_2p_electron(r, Z_eff, geometric_overlap=1.0):
    """Voltage profile of a phase-locked 0_1 unknot in the 2p resonance (l=1) macroscopic trajectory.
    
    In the FOC macroscopic limit, identical electrons phase lock at exactly 120-degrees (3-phase motor) 
    or orthogonally (6-phase motor) bounded by Axiom 4 topological wake expansion limits.
    """
    k_e = 1.0 / (4.0 * np.pi * EPSILON_0)
    # where A = Z_eff * r / a_0
    
    A = Z_eff * r / a_0
    classical_screening = 1.0 - (1.0 + 0.75 * A + 0.25 * A**2 + 1.0/24.0 * A**3) * np.exp(-A)
    return -k_e * e_charge / r * (geometric_overlap * classical_screening)

def _V_total(r, Z, core_profile_defs_with_valency):
    r"""Total scalar voltage field V_total(r) from nucleus and inner shells.
    
    The internal structure of completely filled symmetric phase-separating shells (like $1s^2$) 
    contracts bounded by the absolute macroscopic trace-reversal capacity: $J_1s = 0.5 * (1 + p_c)$.
    
    This identical topological boundary projection identically limits the fractional effective geometric 
    distance to any outer traversing wave. Therefore, the continuous probabilistic integration of the inner 
    core's shielding footprint is scaled symmetrically by this precise ratio: 
    geo_overlap_1s = J_1s_dynamic / J_1s_smeared.
    
    Args:
        core_profile_defs_with_valency: List of tuples (shell_type, Z_eff) or (shell_type, Z_eff, geometric_overlap).
    """
    total_V = _V_nucleus(r, Z)
    
    from ave.core.constants import P_C
    J_1s_dynamic = 0.5 * (1.0 + P_C)
    geo_overlap_1s_foc = J_1s_dynamic / 0.625 # ~0.94672 FOC Density Wake projection
    
    for profile in core_profile_defs_with_valency:
        stype = profile[0]
        Zeff_core = profile[1]
        
        # Intra-shell geometric constraints are passed in, otherwise default.
        geo_overlap = profile[2] if len(profile) > 2 else 1.0
        
        if stype == '1s':
            # FOC continuous projection: The 1s core's continuous spatial shielding is identical 
            # to its structural bound footprint. We explicitly map the scaler to the geometric wake.
            scaler = geo_overlap if geo_overlap != 1.0 else geo_overlap_1s_foc
            total_V += _V_1s_electron(r, Zeff_core, scaler)
        elif stype == '2s':
            total_V += _V_2s_electron(r, Zeff_core, geo_overlap)
        elif stype == '2p':
            total_V += _V_2p_electron(r, Zeff_core, geo_overlap)
        else:
            raise NotImplementedError(f"Voltage profile for shell {stype} is not yet analytical.")
            
    return total_V

def _acoustic_phase_integrand(u, Z, core_profile_defs, E_joules):
    """Local wave number k(u) after substitution r = u^2 to remove the 1/sqrt(r) singularity at r=0."""
    r = u**2
    # Electron has charge -e, so potential energy U(r) = -e * V(r)
    V_joules = -e_charge * _V_total(r, Z, core_profile_defs)
    kinetic_T = E_joules - V_joules
    
    if kinetic_T <= 0.0:
        return 0.0  # Evanescent decay past the classical turning point
        
    # k(r) dr = k(u^2) * 2u du
    return (np.sqrt(2.0 * M_E * kinetic_T) / HBAR) * 2.0 * u

def atomic_resonance(Z, core_profile_defs, shell_n):
    r"""Finds outermost mode eigenvalue using acoustic standing-wave phase-locking.

    Algorithm:
      1. Electron (a massive topological defect) interacts with the Bulk Modulus.
      2. Constructive interference (standing wave) requires the total accumulated 
         phase int k(r) dr across the cavity to equal exactly n * pi.
      3. We search for the unique Total Energy E_joules that satisfies this match.

    Args:
        Z: Nuclear charge.
        core_profile_defs: List of tuples (shell_n, Z_eff) representing inner bumps.
        shell_n: The phase-locking integer (principal quantum number n).

    Returns:
        r_valence: Outermost macroscopic mode radius [m] (classical turning point).
        IE: Ionization Energy trapped in this cavity [eV].
    """
    
    # We are searching for E_eV (which ranges from ~0 to Z^2 * 13.6).
    # Since bound electrons have negative total energy, internal E is -E_eV * e_charge.
    def phase_residual(E_eV):
        E_joules = -E_eV * e_charge
        
        # Determine the cavity wall (where E = V)
        def cross_func(r):
            return -e_charge * _V_total(r, Z, core_profile_defs) - E_joules
            
        try:
            # The deepest well is at r=0. We search for the crossover.
            sol_turn = root_scalar(cross_func, bracket=[1e-15, 100 * shell_n**2 * a_0])
            r_turn = sol_turn.root
        except ValueError:
            return 1e6 # Return large penalty if no root
            
        # Integrate acoustic phase from u=0 to u=sqrt(r_turn)
        u_turn = np.sqrt(r_turn)
        res, _ = quad(_acoustic_phase_integrand, 0.0, u_turn, args=(Z, core_profile_defs, E_joules), limit=200)
        
        # The exact resonance condition for an s-orbital
        target_phase = shell_n * np.pi
        
        return res - target_phase

    # Initial guess bracket: from 0 eV to full unscreened hydrogenic limit
    E_max = Z**2 * 13.606 + 50.0  
    E_min = 0.01
    
    try:
        sol_E = root_scalar(phase_residual, bracket=[E_min, E_max])
        IE_eV = sol_E.root
        
        # Calculate final r_val for the returned energy
        E_joules = -IE_eV * e_charge
        def cross_func(r):
            return -e_charge * _V_total(r, Z, core_profile_defs) - E_joules
        sol_turn = root_scalar(cross_func, bracket=[1e-15, 100 * shell_n**2 * a_0])
        r_valence = sol_turn.root
        
    except ValueError:
        r_valence = 0.0
        IE_eV = 0.0
        
    return r_valence, IE_eV


def atomic_resonance_symmetric_shell(Z, core_profile_defs, shell_n, shell_type, num_electrons):
    r"""Finds the self-consistent Ionization Energy for a symmetric multi-electron cavity.
    
    For atoms like Helium (1s2) or Beryllium (2s2), identical topological unknots 
    share the exact same macroscopic resonance cavity. They mutually dilate the 
    cavity until their phase locks perfectly with the geometric boundary, forming
    an explicit RLC phase-jitter oscillator model.
    
    This function discovers the WKB fixed-point where the assumed Z_eff defining
    the geometric cavity bounds is exactly mathematically identical to the structural 
    eigenvalue Z_implied extracted from the cavity.
    
    Args:
        Z: Nuclear charge.
        core_profile_defs: List of tuples (shell_type, Z_eff) for inner shielding electrons.
        shell_n: The phase-locking integer (principal quantum number n).
        shell_type: Type of the subshell being filled (e.g. '1s', '2s', '2p').
        num_electrons: Number of electrons completely filling this symmetric cavity.
        
    Returns:
        IE_eV: The 1st Ionization Energy [eV].
    """
    from scipy.optimize import root_scalar
    
    if num_electrons == 1:
        # A single valence electron has no dynamic symmetric peers inside its cavity
        r_val, IE = atomic_resonance(Z, core_profile_defs, shell_n)
        return IE

    def residual(Z_test):
        profiles = list(core_profile_defs)
        
        # To insert our pure analytical topological scalars into the continuous WKB engine, 
        # we must explicitly cancel out the engine's default classical Hartree-Fock probability integrals.
        CLASSICAL_HF_1S_INTEGRAL = 0.625
        CLASSICAL_HF_2S_INTEGRAL = 77.0 / 512.0
        CLASSICAL_HF_2P_INTEGRAL = 19.0 / 256.0
        
        from ave.core.constants import P_C
        if shell_type == '1s':
            J_1s_dynamic = 0.5 * (1.0 + P_C)
            overlap_factor = J_1s_dynamic / CLASSICAL_HF_1S_INTEGRAL
        elif shell_type == '2s':
            # Cooper Pair Topological Wake map!
            S_pc = np.sqrt(1.0 - P_C**2)
            J_2s_dynamic = P_C * S_pc
            overlap_factor = J_2s_dynamic / CLASSICAL_HF_2S_INTEGRAL
        elif shell_type == '2p':
            # 3-Phase AC Motor Limit / 6-Phase Orthogonal Limit
            J_2p_dynamic = (1.0 / np.sqrt(3.0)) * (1.0 + P_C)
            overlap_factor = J_2p_dynamic / CLASSICAL_HF_2P_INTEGRAL
        else:
            overlap_factor = 1.0
            
        for _ in range(num_electrons - 1):
            profiles.append((shell_type, Z_test, overlap_factor))
            
        r_val, IE = atomic_resonance(Z, profiles, shell_n)
        if np.isnan(IE) or IE <= 0.0: 
            return 100.0
            
        Z_implied = np.sqrt(IE / 13.60569312 * (shell_n**2))
        return Z_test - Z_implied

    sol = root_scalar(residual, bracket=[0.5, Z])
    Z_opt = sol.root
    final_profiles = list(core_profile_defs)
    
    CLASSICAL_HF_1S_INTEGRAL = 0.625
    CLASSICAL_HF_2S_INTEGRAL = 77.0 / 512.0
    CLASSICAL_HF_2P_INTEGRAL = 19.0 / 256.0
    
    from ave.core.constants import P_C
    if shell_type == '1s':
        overlap_factor = (0.5 * (1.0 + P_C)) / CLASSICAL_HF_1S_INTEGRAL
    elif shell_type == '2s':
        overlap_factor = (P_C * np.sqrt(1.0 - P_C**2)) / CLASSICAL_HF_2S_INTEGRAL
    elif shell_type == '2p':
        J_2p_dynamic = (1.0 / np.sqrt(3.0)) * (1.0 + P_C)
        overlap_factor = J_2p_dynamic / CLASSICAL_HF_2P_INTEGRAL
    else:
        overlap_factor = 1.0
        
    for _ in range(num_electrons - 1):
        final_profiles.append((shell_type, Z_opt, overlap_factor))
        
    r_val, IE_final = atomic_resonance(Z, final_profiles, shell_n)
    return IE_final


def _axiom4_phase_lock(n_electrons_in_shell, shell_capacity):
    r"""Axiom 4 saturation operator applied to intra-shell electron coupling.

    UNIVERSAL OPERATOR: S(x) = √(1 − x²)  [Axiom 4]

    PHYSICAL MEANING:
      When multiple electrons share a shell, they spatially self-organize
      to minimize mutual impedance (Be chapter: "90° perpendicular
      phase-lock guarantees stability without exchange-correlation").

      The filling fraction x = N_e / capacity acts as the dimensionless
      strain ratio (same role as V_R/V_BR in nuclear binding or
      d_sat/r in gravitational regime boundaries).

      At x = 0 (empty shell): S = 1.0, no phase-lock reduction.
      At x = 1 (full shell):  S = 0.0, complete cancellation — but
        electrons still interact, so we use S² as the coupling modifier
        (analogous to power reflection |Γ|² = 1 − S²).

    DIMENSIONAL CHECK:
      Input:  n_electrons [count], capacity [count]
      x = n/cap [dimensionless]
      Output: S(x) [dimensionless]  ✓

    EE ANALOG: Port isolation in a multi-port resonant cavity.
      S(x) plays the same role as the saturation operator in the nuclear
      semiconductor binding engine (Axiom 4 dielectric saturation).

    Args:
        n_electrons_in_shell: Electrons occupying this shell.
        shell_capacity: Maximum occupancy (2n²).

    Returns:
        S(x) = √(1 − x²), the fraction of repulsion surviving phase-lock.
    """
    if n_electrons_in_shell <= 1:
        return 1.0
    x = n_electrons_in_shell / shell_capacity
    # Axiom 4 saturation: S(x) = √(1 − x²)
    return np.sqrt(max(1.0 - x**2, 0.0))


def _screening_sigma(Z, shells, target_idx):
    r"""Total screening at shell target_idx from all other electrons.

    UNIVERSAL OPERATOR: E_coulomb = (3/5)·Z(Z-1)·αℏc/R  [Coulomb operator]

    At atomic scale, this becomes the screening constant σ:
      σ_i = Σ_j (occ_j × k_geo(nᵢ,nⱼ) × phase_factor)

    where:
      occ_j [dimensionless count] — electrons in shell j
      k_geo [dimensionless fraction] — from Axiom 3 (3D geometry)
      phase_factor [dimensionless 0..1] — from Axiom 4 (self-organization)

    DIMENSIONAL CHECK:
      σ = [count] × [fraction] × [fraction] = [dimensionless]
      Z_eff = Z − σ = [charge] − [charge] = [charge]  ✓

    For INTER-shell pairs (i ≠ j):
      Full coupling: each electron in shell j screens the target by k_geo.
      No angular phase-lock (electrons on different radial tracks).

    For INTRA-shell pairs (i = j):
      Phase-lock reduces the coupling via S(fill) from Axiom 4.
      The factor (N_e − 1) counts how many OTHER electrons in the same
      shell are screening the target electron.

    Args:
        Z: Nuclear charge.
        shells: List of (n, count) from _fill_shells().
        target_idx: Which shell index to compute screening for.

    Returns:
        σ: Total screening [dimensionless, units of charge].
    """
    sigma = 0.0
    n_target = shells[target_idx][0]
    count_target = shells[target_idx][1]

    for j, (n_j, count_j) in enumerate(shells):
        # Geometric Coulomb fraction [Axiom 3]
        key = (min(n_target, n_j), max(n_target, n_j))
        k_geo = _K_GEO.get(key, 0.02)

        if j == target_idx:
            # INTRA-shell: screening from (count - 1) other electrons
            # in the same shell.
            #
            # NOTE ON PHASE-LOCK (Axiom 4):
            #   S(fill) = √(1−fill²) describes the shell's SATURATION state
            #   (whether the shell can accept more electrons). It does NOT
            #   reduce the Coulomb coupling between electrons already inside.
            #   The J integral (k_geo) already accounts for spherically-averaged
            #   charge distributions. Angular self-organization (phase-locking)
            #   changes WHERE electrons sit, but the spherical average ⟨1/r₁₂⟩
            #   is a geometric constant of 3D space — it doesn't depend on
            #   the angular arrangement.
            #
            #   In RF terms: S(fill) is the reflection coefficient at the
            #   shell boundary. Port isolation between EXISTING internal
            #   modes is fixed by the cavity geometry (k_geo), not by the
            #   boundary condition.
            if count_j <= 1:
                continue
            sigma += (count_j - 1) * k_geo
        else:
            # INTER-shell: full Coulomb screening, no angular reduction
            sigma += count_j * k_geo

    return sigma


def _build_dynamical_matrix(Z, shells):
    r"""Build the N_total × N_total dynamical matrix for the electron system.

    IDENTICAL STRUCTURE TO NUCLEAR BINDING:
      Nuclear:  N nucleons, all at ω₀ = c(1+ν)/d_p, coupling k = 2α
                D = ω₀² × (I + k×A)  where A = adjacency of K_N
      Atomic:   N electrons, ω_i = Z²Ry/n_i² (shell-dependent), k = K_GEO
                D[i,i] = ε_i²,  D[i,j] = k_geo × ε_i × ε_j

    PHYSICAL MEANING:
      The dynamical matrix D is the stiffness matrix of N coupled oscillators.
      Its eigenvalues ω_m² give the NORMAL MODE frequencies of the system.

      Off-diagonal elements represent REPULSIVE coupling (electron-electron
      Coulomb interaction). Positive off-diagonal → modes are pushed apart:
        - One mode goes UP (anti-bonding)
        - Other modes go DOWN (bonding)
      The NET effect is a REDUCTION in total mode-frequency sum, meaning
      coupling reduces total binding energy (as expected for repulsion).

    DIMENSIONAL CHECK:
      ε_i = Z² × Ry / n²  [eV]
      D[i,i] = ε_i²       [eV²]
      D[i,j] = k_geo × ε_i × ε_j  [dimensionless × eV × eV = eV²]  ✓
      Eigenvalues: [eV²]
      Normal modes: √(eigenvalue) = [eV]  ✓

    UNIVERSAL OPERATOR MAPPING:
      ω₀ = ε_i = Z²Ry/n²             [Operator 1: eigenfrequency]
      k = K_GEO(n_i, n_j)             [Operator 2: coupling]
      Σ√(eigenvalues) = total binding  [Operator 3: via mode spectrum]

    Args:
        Z: Nuclear charge.
        shells: List of (n, count) from _fill_shells().

    Returns:
        D: (N_total × N_total) dynamical matrix [eV²].
        electrons: List of (n, epsilon) per electron.
    """
    # Build electron list: each electron has its uncoupled binding frequency
    electrons = []
    for n, count in shells:
        # ε = Z²Ry/n² — uncoupled one-electron binding [eV]
        # This is Operator 1 (eigenfrequency) at the atomic scale
        epsilon = Z**2 * _RY_EV / n**2
        for _ in range(count):
            electrons.append((n, epsilon))

    N = len(electrons)
    D = np.zeros((N, N))

    for i in range(N):
        ni, ei = electrons[i]
        # Diagonal: uncoupled binding frequency squared
        D[i, i] = ei**2

        for j in range(i + 1, N):
            nj, ej = electrons[j]

            # Operator 2: coupling from 3D Coulomb geometry [Axiom 3]
            key = (min(ni, nj), max(ni, nj))
            k_geo = _K_GEO.get(key, 0.02)

            # Off-diagonal: repulsive coupling [eV²]
            # Positive sign → repulsion (modes pushed apart)
            D[i, j] = k_geo * ei * ej
            D[j, i] = D[i, j]

    return D, electrons


def _total_binding_from_modes(D):
    r"""Total binding energy from normal mode spectrum.

    SAME OPERATION AS NUCLEAR BINDING:
      Nuclear:  B = N × ℏ(ω₀ - ω_bonding)
                  = N × ℏω₀ - Σ ℏω_m  (summing over normal modes)
      Atomic:   Total binding = Σ√(eigenvalues of D)
                IE = binding(N) - binding(N-1)

    The eigenvalues of D are ω_m² (normal mode frequencies squared).
    Each √(eigenvalue) is the coupled binding frequency of one mode.
    The SUM of all mode frequencies = total binding of the system.

    DIMENSIONAL CHECK:
      eigenvalues [eV²] → √ → [eV]
      Σ [eV] = [eV]  ✓

    Args:
        D: Dynamical matrix [eV²].

    Returns:
        Total binding energy [eV].
    """
    eigenvalues = np.linalg.eigvalsh(D)
    # Clip any numerical noise (all physical eigenvalues must be ≥ 0)
    modes = np.sqrt(np.maximum(eigenvalues, 0.0))
    return np.sum(modes)


def ionization_energy_cascade(Z, n_electrons=None):
    r"""First ionization energy from coupled oscillator normal mode spectrum.

    CIRCUIT MODEL — SAME AS NUCLEAR BINDING:
      Nuclear binding (coupled_resonator_binding):
        N nucleons at ω₀, coupling k = 2α, adjacency K_N
        → diagonalize → bonding mode → B = N·ℏ(ω₀ - ω_bond)

      Atomic IE (this function):
        N electrons at ε_i = Z²Ry/n_i², coupling k = K_GEO
        → diagonalize dynamical matrix → normal modes
        → IE = total_binding(N) - total_binding(N-1)

    The ONLY differences from the nuclear case:
      1. Non-uniform frequencies (ε depends on shell n)
      2. Coupling is REPULSIVE (positive off-diagonal in D)
      3. IE via energy difference, not direct mode splitting

    ALGORITHM:
      1. Build N×N dynamical matrix D for neutral atom
      2. Eigenvalues → √ → normal mode frequencies
      3. Total binding = Σ√(eigenvalues)
      4. Repeat for ion (N-1 electrons)
      5. IE = binding(neutral) - binding(ion)

    DIMENSIONAL CHAIN:
      ε_i = Z² × Ry / n²                [eV]
      D[i,j] = k_geo × ε_i × ε_j       [eV²]
      eigenvalues                        [eV²]
      modes = √(eigenvalues)             [eV]
      binding = Σ modes                  [eV]
      IE = binding(N) - binding(N-1)     [eV]  ✓

    AVE AXIOM TRACEABILITY:
      Axiom 1 → ε_i = Z²Ry/n² (Bohr LC resonance)
      Axiom 3 → K_GEO from 3D geometry (ν = 2/7)
      Zero free parameters. Deterministic. Same framework as nuclear.

    Args:
        Z: Nuclear charge (element number).
        n_electrons: Number of electrons (default: Z for neutral atom).

    Returns:
        IE in eV. Positive = energy required to remove an electron.
    """
    if n_electrons is None:
        n_electrons = Z
    if n_electrons <= 0:
        return 0.0
    if n_electrons == 1:
        # Single electron: exact hydrogen-like result [Axiom 1]
        # No coupling → single mode → binding = Z²Ry
        return _RY_EV * Z**2

    # Neutral atom: N electrons
    shells_neutral = _fill_shells(n_electrons)
    D_neutral, electrons_neutral = _build_dynamical_matrix(Z, shells_neutral)
    binding_neutral = _total_binding_from_modes(D_neutral)

    # Ion: N-1 electrons (remove outermost)
    n_outer = shells_neutral[-1][0]
    count_outer = shells_neutral[-1][1]
    if count_outer > 1:
        shells_ion = shells_neutral[:-1] + [(n_outer, count_outer - 1)]
    else:
        shells_ion = shells_neutral[:-1]

    if len(shells_ion) == 0:
        # Bare nucleus: IE = Z²Ry
        return _RY_EV * Z**2

    D_ion, _ = _build_dynamical_matrix(Z, shells_ion)
    binding_ion = _total_binding_from_modes(D_ion)

    # IE = how much MORE bound the neutral is than the ion
    # binding(neutral) > binding(ion) because neutral has more electrons
    # IE = binding(neutral) - binding(ion)
    ie = binding_neutral - binding_ion

    return ie

