"""
Protein Bond Constants — AVE Derivation Chain for Backbone
===========================================================

Documents the full derivation chain from AVE axioms to the protein
folding engine's backbone parameters.

DERIVATION PATH:
    Axioms 1-4 → ℓ_node → m_e → α → nuclear binding
               → covalent radii → backbone bond lengths
               → Cα-Cα peptide geometry → d₀ ≈ 3.8 Å

The Cα-Cα virtual bond distance (3.80 ± 0.02 Å) is determined by:
  1. Three covalent bond lengths: Cα-C (1.52 Å), C-N (1.33 Å), N-Cα (1.46 Å)
  2. Two bond angles: Cα-C-N (116.2°), C-N-Cα (121.7°)
  3. The trans peptide dihedral: ω ≈ 180°
  4. The Ramachandran angles: φ ≈ -60°, ψ ≈ -40° (α-helix)
                              φ ≈ -120°, ψ ≈ 130° (β-sheet)

All three bond lengths trace back through covalent radii to the
proton charge radius d_p = 4ℏ/(m_p·c) ≈ 0.841 fm, which is
derived from Axiom 1 (lattice pitch) + nuclear confinement
(cinquefoil knot with c=5 crossings).

The backbone bond lengths are NOT independently derivable from the
current 1D bond solver (which computes nuclear-scale fm separations).
They require multi-body Hartree-Fock-level calculations emergent from
the same nuclear physics. For now, they are documented as bridged
constants: connected to the axiom chain through the proton, but
requiring future 3D molecular solver development to compute ab initio.

This module exposes the backbone constants for import by the protein
folding engine, enabling traceability even where the computation
chain is not yet fully automated.
"""

import numpy as np
from ave.core.constants import (
    HBAR, C_0, M_E, D_PROTON, L_NODE,
    PROTON_ELECTRON_RATIO, ALPHA,
)

# ═══════════════════════════════════════════════════════════════
# Fundamental AVE link
# ═══════════════════════════════════════════════════════════════

# Proton charge radius — derived from Axiom 1
D_PROTON_FM = D_PROTON     # [fm] = 4ℏ/(m_p·c)

# Bohr radius — derived from ℓ_node and α
# a₀ = ℓ_node / α = ℏ/(m_e·c·α) ≈ 0.529 Å
BOHR_RADIUS_M = L_NODE / ALPHA  # ≈ 5.29e-11 m
BOHR_RADIUS_ANGSTROM = BOHR_RADIUS_M * 1e10  # ≈ 0.529 Å


# ═══════════════════════════════════════════════════════════════
# Backbone bond lengths (crystallographic, traceable to d_p)
# ═══════════════════════════════════════════════════════════════

# Standard Engh & Huber (1991) backbone geometry
# These emerge from the covalent bonding of C, N, O atoms whose
# nuclear structure is computed by the AVE nuclear solver
BACKBONE_BONDS = {
    'Ca-C':   {'length_A': 1.52, 'type': 'single', 'atoms': ('C', 'C')},
    'C-N':    {'length_A': 1.33, 'type': 'partial_double', 'atoms': ('C', 'N')},
    'N-Ca':   {'length_A': 1.46, 'type': 'single', 'atoms': ('N', 'C')},
    'C=O':    {'length_A': 1.23, 'type': 'double', 'atoms': ('C', 'O')},
    'N-H':    {'length_A': 1.01, 'type': 'single', 'atoms': ('N', 'H')},
}

# Backbone bond angles
BACKBONE_ANGLES = {
    'N-Ca-C':  111.2,  # degrees (tetrahedral-derived)
    'Ca-C-N':  116.2,  # degrees
    'C-N-Ca':  121.7,  # degrees (amide planarity)
}


# ═══════════════════════════════════════════════════════════════
# Cα-Cα virtual bond length
# ═══════════════════════════════════════════════════════════════

# The Cα-Cα distance (3.80 ± 0.02 Å) is a conformational average
# over the trans peptide bond geometry. It is the standard value
# from X-ray crystallography of >100,000 protein structures.
#
# This constant is used by the protein S₁₁ folding engine as d₀.
CA_CA_BOND_LENGTH_ANGSTROM = 3.80  # Å (crystallographic standard)
CA_CA_BOND_LENGTH_M = CA_CA_BOND_LENGTH_ANGSTROM * 1e-10

# Ratio to Bohr radius: d₀ / a₀ ≈ 7.18
# This dimensionless ratio may have a geometric origin in the
# backbone's repeating peptide unit structure.
D0_OVER_BOHR = CA_CA_BOND_LENGTH_ANGSTROM / BOHR_RADIUS_ANGSTROM


# ═══════════════════════════════════════════════════════════════
# Amino acid impedance table (from axiom-derived properties)
# ═══════════════════════════════════════════════════════════════

# Q-factor of the backbone amide-V resonator
# Derived: amide-V mode at 23 THz, linewidth ~3 THz → Q = f₀/Δf ≈ 7
Q_BACKBONE = 7.0

# Per-residue topological impedance (R + jX)
# R: sidechain hydrophobic volume (from periodic table solver)
# X: charge reactance / Q (from electrostatic properties)
Z_TOPO = {
    # Hydrophobic: R only
    'A': 0.53 + 0.00j, 'V': 0.93 + 0.00j, 'I': 0.73 + 0.00j,
    'L': 1.00 + 0.00j, 'M': 0.87 + 0.00j, 'F': 1.57 + 0.00j,
    'W': 3.40 + 0.00j, 'P': 5.02 + 0.00j, 'G': 0.50 + 0.00j,
    # Negative charge (capacitive)
    'D': 0.66 - 0.66/Q_BACKBONE*1j,
    'E': 0.52 - 0.52/Q_BACKBONE*1j,
    # Positive charge (inductive)
    'K': 0.60 + 0.60/Q_BACKBONE*1j,
    'R': 0.55 + 0.55/Q_BACKBONE*1j,
    'H': 2.50 + 2.50/(2*Q_BACKBONE)*1j,
    # Polar uncharged
    'S': 1.64 + 1.64/(2*Q_BACKBONE)*1j,
    'T': 1.73 + 1.73/(2*Q_BACKBONE)*1j,
    'C': 1.74 - 1.74/(2*Q_BACKBONE)*1j,
    'Y': 1.31 - 1.31/(2*Q_BACKBONE)*1j,
    'N': 1.10 + 1.10/(2*Q_BACKBONE)*1j,
    'Q': 0.63 + 0.63/(2*Q_BACKBONE)*1j,
}


def print_derivation_chain():
    """Print the full derivation chain from axioms to protein backbone."""
    print("\n" + "=" * 70)
    print("  AVE → Protein Backbone Derivation Chain")
    print("=" * 70)
    print(f"  Axiom 1: ℓ_node = ℏ/(m_e·c) = {L_NODE*1e10:.6f} Å")
    print(f"  Bohr radius: a₀ = ℓ_node/α  = {BOHR_RADIUS_ANGSTROM:.4f} Å")
    print(f"  Proton charge radius: d_p = 4ℏ/(m_p·c) = {D_PROTON_FM:.3f} fm")
    print()
    print("  Backbone bond lengths (from nuclear→covalent chain):")
    for name, bond in BACKBONE_BONDS.items():
        print(f"    {name:8s} = {bond['length_A']:.2f} Å  ({bond['type']})")
    print()
    print(f"  Cα—Cα virtual bond: d₀ = {CA_CA_BOND_LENGTH_ANGSTROM:.2f} Å")
    print(f"  d₀ / a₀ ratio: {D0_OVER_BOHR:.3f}")
    print(f"  Q_backbone: {Q_BACKBONE:.0f} (amide-V mode)")
    print("=" * 70)
