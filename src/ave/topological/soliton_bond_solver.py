r"""
Coulomb Bond Force Constant Solver
===================================

Derives covalent bond force constants from the electromagnetic (Coulomb)
interaction on the AVE lattice.

Inputs (all non-spectroscopic):
    ε₀, m_e, ℏ, e  — from AVE axioms
    Z              — atomic number (periodic table)
    Slater rules   — screening constants σ and n* (atomic structure)

Key corrections derived from lattice topology:
    1. Isotropy projection 1/D (D=3 spatial dimensions, Axiom 1)
    2. Three-phase balance factor 1/√3 for terminal atoms (SRS lattice)
    3. σ/π decomposition for double bonds (angular structure)
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from ave.core.constants import EPSILON_0, M_E, HBAR, e_charge

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
_k_coul = e_charge**2 / (4.0 * np.pi * EPSILON_0)
A_BOHR = HBAR**2 / (M_E * _k_coul)


def _slater_z_eff(Z: int) -> float:
    """Slater effective nuclear charge Z_eff = Z - σ."""
    z_eff = {1: 1.00, 6: 3.25, 7: 3.90, 8: 4.55, 16: 5.45}
    return z_eff.get(Z, Z * 0.5)


def _n_star(Z: int) -> float:
    """Effective principal quantum number of valence shell."""
    if Z <= 2: return 1.0
    if Z <= 10: return 2.0
    return 3.0


def _slater_orbital_radius(Z: int) -> float:
    """Most probable radius of Slater orbital [m]: r = n*² · a₀ / Z_eff."""
    return _n_star(Z)**2 * A_BOHR / _slater_z_eff(Z)


def _electronegativity(Z: int) -> float:
    """Pauling electronegativity."""
    chi = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 16: 2.58}
    return chi.get(Z, 2.5)


def _is_terminal(Z: int) -> bool:
    """Hydrogen is a terminal node on the lattice (1 bond, not 3)."""
    return Z <= 2


# ═══════════════════════════════════════════════════════════
# BOND ENERGY MODEL
# ═══════════════════════════════════════════════════════════

def bond_energy(d: float, Z_a: int, Z_b: int, n_shared: int) -> float:
    """
    Total energy of a covalent bond at internuclear distance d [m].

    Uses Slater orbital radii as fixed electron cloud sizes.
    σ/π decomposition for double bonds: π electrons contribute
    with reduced coupling (perpendicular to bond axis).
    """
    if d <= 0:
        return 1e10

    Z_eff_a = _slater_z_eff(Z_a)
    Z_eff_b = _slater_z_eff(Z_b)

    # Electron cloud sizes from Slater orbitals
    r_a = _slater_orbital_radius(Z_a)
    r_b = _slater_orbital_radius(Z_b)
    r_e = (r_a + r_b) / 2.0

    # σ/π decomposition for multiple bonds
    # σ electrons (along bond axis) respond fully to stretching.
    # π electrons (perpendicular) respond with reduced coupling.
    n_sigma = min(n_shared, 2)
    n_pi = n_shared - n_sigma
    PI_COUPLING = 0.5  # π/σ ratio from angular distribution

    # 1. Nuclear-nuclear Coulomb repulsion
    E_nn = _k_coul * Z_eff_a * Z_eff_b / d

    # 2. Electron-nuclear Coulomb attraction
    if Z_a == Z_b:
        center = d / 2
    else:
        chi_a = _electronegativity(Z_a)
        chi_b = _electronegativity(Z_b)
        center = d * chi_b / (chi_a + chi_b)

    r_avg_a = np.sqrt(center**2 + r_e**2)
    r_avg_b = np.sqrt((d - center)**2 + r_e**2)

    n_eff_en = n_sigma + PI_COUPLING * n_pi
    E_en = -_k_coul * n_eff_en * (Z_eff_a / r_avg_a + Z_eff_b / r_avg_b)

    # 3. Kinetic energy (exact STO value, constant in d)
    E_h = _k_coul / A_BOHR
    T_a = Z_eff_a**2 * E_h / (2 * _n_star(Z_a)**2)
    T_b = Z_eff_b**2 * E_h / (2 * _n_star(Z_b)**2)
    E_kin = (n_shared / 2) * (T_a + T_b)

    # 4. Electron-electron repulsion
    n_pairs = n_shared * (n_shared - 1) / 2
    if n_pairs > 0:
        E_ee = _k_coul * n_pairs / r_e
    else:
        E_ee = 0.0

    return E_nn + E_en + E_kin + E_ee


# ═══════════════════════════════════════════════════════════
# FORCE CONSTANT EXTRACTION
# ═══════════════════════════════════════════════════════════

def compute_bond_curve(Z_a, Z_b, n_shared, d_min=0.5e-10, d_max=4.0e-10, n_points=200):
    """Compute E(d) for a given bond. Returns (d [m], E [J])."""
    d_range = np.linspace(d_min, d_max, n_points)
    energies = np.array([bond_energy(d, Z_a, Z_b, n_shared) for d in d_range])
    return d_range, energies


def extract_force_constant(d_array, E_array, Z_a: int = 6, Z_b: int = 6):
    """
    Extract d_eq [m] and k [N/m] from E(d) curve.

    Applies two lattice-topology corrections:

    1. ISOTROPY PROJECTION (1/3): Bond stretching acts along 1 of 3
       equivalent spatial dimensions on the isotropic SRS lattice.

    2. THREE-PHASE BALANCE (1/√3 for terminal atoms):
       On the SRS lattice, each interior node is a 3-connected WYE
       junction — a three-phase power node. When both bond endpoints
       are interior (3-connected), the system is balanced: each phase
       carries 1/3 of the curvature.

       When one endpoint is terminal (hydrogen, 1 bond), the load is
       UNBALANCED — like connecting a single-phase load to a three-
       phase WYE system. In power engineering, the line-to-neutral
       voltage relates to line-to-line by 1/√3. The unbalanced
       current distribution adds a factor of 1/√3.

    Combined: k = (1/3) × (balance) × d²E/dd²
       balanced (heavy-heavy):    balance = 1     → k = k_raw/3
       unbalanced (X-H):          balance = 1/√3  → k = k_raw/(3√3)
    """
    # Three-phase balance factor
    n_terminal = sum(1 for Z in [Z_a, Z_b] if _is_terminal(Z))
    balance_factor = (1.0 / np.sqrt(3.0)) ** n_terminal

    # Combined isotropy correction
    ISOTROPY = 1.0 / 3.0
    correction = ISOTROPY * balance_factor

    i_min = np.argmin(E_array)
    d_eq = d_array[i_min]
    E_min = E_array[i_min]
    dd = d_array[1] - d_array[0]
    if 1 < i_min < len(d_array) - 2:
        k_raw = (-E_array[i_min-2] + 16*E_array[i_min-1] - 30*E_array[i_min]
                 + 16*E_array[i_min+1] - E_array[i_min+2]) / (12 * dd**2)
    elif 0 < i_min < len(d_array) - 1:
        k_raw = (E_array[i_min+1] - 2*E_array[i_min] + E_array[i_min-1]) / dd**2
    else:
        k_raw = 0.0
    return d_eq, abs(k_raw) * correction, E_min


# ═══════════════════════════════════════════════════════════
# BOND DATA
# ═══════════════════════════════════════════════════════════

BOND_DEFS = {
    'C-H': (6, 1, 2),   'C-C': (6, 6, 2),   'C=C': (6, 6, 4),
    'C-N': (6, 7, 2),   'C=O': (6, 8, 4),   'C-O': (6, 8, 2),
    'N-H': (7, 1, 2),   'O-H': (8, 1, 2),   'S-H': (16, 1, 2),
    'C-S': (6, 16, 2),  'S-S': (16, 16, 2),
}

KNOWN_K = {
    'C-H': 494, 'C-C': 354, 'C=C': 965, 'C-N': 461,
    'C=O': 1170, 'C-O': 489, 'N-H': 641, 'O-H': 745,
    'S-H': 390, 'C-S': 253, 'S-S': 236,
}

KNOWN_D = {
    'C-H': 1.09e-10, 'C-C': 1.54e-10, 'C=C': 1.34e-10, 'C-N': 1.47e-10,
    'C=O': 1.23e-10, 'C-O': 1.43e-10, 'N-H': 1.01e-10, 'O-H': 0.96e-10,
    'S-H': 1.34e-10, 'C-S': 1.82e-10, 'S-S': 2.05e-10,
}


if __name__ == "__main__":
    print("=" * 80)
    print("  Coulomb Bond Solver — Three-Phase Lattice Isotropy")
    print("=" * 80)
    print(f"  a₀ = {A_BOHR*1e10:.4f} Å")
    print(f"  Isotropy:  interior-interior = 1/3 = {1/3:.4f}")
    print(f"             interior-terminal = 1/(3√3) = {1/(3*np.sqrt(3)):.4f}")

    print(f"\n  {'Bond':>6}  {'d_eq(Å)':>9}  {'d_known':>8}  "
          f"{'k(N/m)':>9}  {'k_known':>8}  {'k_ratio':>8}  {'d_ratio':>8}")
    print("-" * 80)

    for bond, (za, zb, ne) in BOND_DEFS.items():
        d, E = compute_bond_curve(za, zb, ne, d_min=0.3e-10, d_max=4.0e-10, n_points=300)
        d_eq, k_pred, E_min = extract_force_constant(d, E, za, zb)
        k_known = KNOWN_K[bond]
        d_known = KNOWN_D[bond]
        print(f"  {bond:>6}  {d_eq*1e10:>9.3f}  {d_known*1e10:>8.2f}  "
              f"{k_pred:>9.1f}  {k_known:>8}  {k_pred/k_known:>8.3f}  "
              f"{d_eq/d_known:>8.3f}")
