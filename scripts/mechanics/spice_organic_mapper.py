r"""
SPICE Organic Mapper — Zero-Parameter AVE Derivation
=====================================================
Maps organic chemical topologies (atomic nuclei and covalent bonds)
into absolute Inductance (L) and Capacitance (C) values for SPICE
circuit simulation.

DERIVATION (from AVE Axioms 1–4)
---------------------------------
The vacuum lattice is an LC transmission line with per-unit-length
parameters μ₀ [H/m] and ε₀ [F/m].  The topological conversion
constant ξ_topo ≡ e / ℓ_node [C/m] maps charge dislocation to
spatial dislocation, providing the universal electromechanical
coupling of the lattice.

  1. Mass → Inductance
     An atomic nucleus of mass m is a localized inertial defect.
     Inertia ≡ Inductance.  Dimensional transduction via ξ²:

         L_atom = m / ξ_topo²          [H]

  2. Bond Stiffness → Capacitance
     A covalent bond of stretching force constant k [N/m] is a
     region of dielectric compliance between two massive nodes.
     Compliance ≡ Capacitance:

         C_bond = ξ_topo² / k           [F]

  Self-consistency checks:
    • f_res = 1/(2π√LC) = (1/2π)√(k/m)   — recovers mechanical resonance  ✓
    • Z = √(L/C) = m·√(k/m) / ξ² = √(mk)/ξ²  — mechanical impedance     ✓
    • v = 1/√(LC) = √(k/m)               — bond sound speed              ✓

  NO FREE PARAMETERS.  All values trace to:
    • CODATA atomic masses  (measured)
    • IR-spectroscopic bond force constants  (measured)
    • ξ_topo = e / ℓ_node  (derived from e, ℏ, m_e, c)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ave.core.constants import (
    e_charge, HBAR, M_E, C_0, Z_0, MU_0, EPSILON_0, L_NODE, XI_TOPO
)

# =============================================================================
# TRANSDUCTION CONSTANT  ξ_topo² = (e / ℓ_node)²  [C²/m²]
# =============================================================================
# This is the universal electromechanical coupling of the vacuum lattice.
# It converts mechanical impedance (kg, N/m) into electrical impedance (H, F).
XI_TOPO_SQ: float = XI_TOPO**2   # ≈ 1.721e-13  [C²/m²]

# =============================================================================
# 1.  ATOMIC INDUCTANCE:  L = m / ξ²   [H]
# =============================================================================
# Atomic masses from CODATA 2018 (in kg).
# 1 Da = 1.66053906660e-27 kg
_DA = 1.66053906660e-27  # kg per Dalton

ATOMIC_MASS_DA = {
    'H':   1.00794,
    'C':  12.0107,
    'N':  14.0067,
    'O':  15.9994,
    'S':  32.065,
}

ATOMIC_INDUCTANCE = {
    elem: (mass_da * _DA) / XI_TOPO_SQ
    for elem, mass_da in ATOMIC_MASS_DA.items()
}
# Units: Henries.  Typical scale: H ≈ 9.7 fH, C ≈ 116 fH, O ≈ 154 fH

# =============================================================================
# 2.  BOND CAPACITANCE:  C = ξ² / k   [F]
# =============================================================================
from ave.topological.soliton_bond_solver import (
    compute_bond_curve, extract_force_constant, BOND_DEFS
)

# Stretching force constants k [N/m] are now derived purely from AVE axioms
# (ε₀, m_e, ℏ, e) and lattice topology (isotropy, three-phase balance, 
# and transformer core expansion) using the Soliton Bond Solver.
# No empirical or spectroscopic parameters are used.
BOND_FORCE_CONSTANTS = {}
for _bond, (_za, _zb, _ne) in BOND_DEFS.items():
    # Evaluate 200 points for stable numerical second derivative
    _d_range, _E_array = compute_bond_curve(_za, _zb, _ne, n_points=200)
    _, _k_pred, _ = extract_force_constant(_d_range, _E_array, _za, _zb)
    BOND_FORCE_CONSTANTS[_bond] = _k_pred

COVALENT_CAPACITANCE = {
    bond: XI_TOPO_SQ / k
    for bond, k in BOND_FORCE_CONSTANTS.items()
}
# Units: Farads.  Typical scale: C-H ≈ 348 aF, C-C ≈ 486 aF

# =============================================================================
# 3.  FUNCTIONAL GROUP CONSTANTS
# =============================================================================

# Amino Group (NH₃⁺) → High-frequency source
# The biological power supply is the ambient THz thermal noise floor.
# Wien's law at 310 K: f_peak ≈ 30 THz
AMINO_SOURCE_FREQ = "30THz"
AMINO_SOURCE_VOLT = "1V"       # Normalized driving amplitude

# Carboxyl Group (COO⁻) → Vacuum impedance termination
CARBOXYL_LOAD_R = f"{Z_0:.4f}Ohm"  # ≈ 376.73 Ω (derived, not heuristic)

# =============================================================================
# API
# =============================================================================

def get_inductance(element: str) -> float:
    """Return geometric inductance [H] of an atomic node."""
    if element not in ATOMIC_INDUCTANCE:
        raise ValueError(f"Unknown element: {element}")
    return ATOMIC_INDUCTANCE[element]

def get_capacitance(bond: str) -> float:
    """Return dielectric capacitance [F] of a covalent bond."""
    if bond in COVALENT_CAPACITANCE:
        return COVALENT_CAPACITANCE[bond]
    # Allow reverse lookup  (e.g. 'H-C' → 'C-H')
    rev = f"{bond[-1]}{bond[1:-1]}{bond[0]}"
    if rev in COVALENT_CAPACITANCE:
        return COVALENT_CAPACITANCE[rev]
    raise ValueError(f"Unknown bond: {bond}")

def get_force_constant(bond: str) -> float:
    """Return stretching force constant [N/m] of a covalent bond."""
    if bond in BOND_FORCE_CONSTANTS:
        return BOND_FORCE_CONSTANTS[bond]
    rev = f"{bond[-1]}{bond[1:-1]}{bond[0]}"
    if rev in BOND_FORCE_CONSTANTS:
        return BOND_FORCE_CONSTANTS[rev]
    raise ValueError(f"Unknown bond: {bond}")

# =============================================================================
# DIAGNOSTIC
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  AVE Organic SPICE Mapper — Zero-Parameter Derivation")
    print("=" * 65)
    print(f"\n  Transduction constant  ξ_topo = {XI_TOPO:.6e} C/m")
    print(f"  Transduction squared   ξ²     = {XI_TOPO_SQ:.6e} C²/m²")

    print(f"\n  --- Atomic Inductances  L = m / ξ²  [fH] ---")
    for elem in ['H', 'C', 'N', 'O', 'S']:
        L = ATOMIC_INDUCTANCE[elem]
        print(f"    {elem:2s}:  {L*1e15:10.3f} fH   (m = {ATOMIC_MASS_DA[elem]:.3f} Da)")

    print(f"\n  --- Bond Capacitances   C = ξ² / k  [aF] ---")
    for bond in ['C-H', 'C-C', 'C=C', 'C-N', 'C=O', 'C-O', 'N-H', 'O-H', 'S-H', 'C-S']:
        C = COVALENT_CAPACITANCE[bond]
        k = BOND_FORCE_CONSTANTS[bond]
        print(f"    {bond:4s}: {C*1e18:10.3f} aF   (k = {k:6.0f} N/m)")

    print(f"\n  --- Self-consistency: C-H stretch ---")
    L_C = ATOMIC_INDUCTANCE['C']
    L_H = ATOMIC_INDUCTANCE['H']
    C_CH = COVALENT_CAPACITANCE['C-H']
    # Reduced mass approach: L_red = L_C*L_H/(L_C+L_H)
    L_red = (L_C * L_H) / (L_C + L_H)
    f_res = 1.0 / (2 * np.pi * np.sqrt(L_red * C_CH))
    nu_cm = f_res / (C_0 * 100)
    print(f"    f_res = {f_res:.3e} Hz  ≈ {nu_cm:.0f} cm⁻¹  (expect ~3000 cm⁻¹)")
    print(f"\n  Carboxyl load: {CARBOXYL_LOAD_R}")
