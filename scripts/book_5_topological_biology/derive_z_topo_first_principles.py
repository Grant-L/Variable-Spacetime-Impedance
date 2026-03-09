#!/usr/bin/env python3
"""
Z_topo First-Principles Derivation
====================================

Derives the topological impedance Z_topo for all 20 amino acids
from the axiom-derived L/C values in spice_organic_mapper.py.

DERIVATION CHAIN:
  Axioms 1-4 → ξ_topo = e/ℓ_node
             → L = m/ξ²  (atomic inductance)
             → C = ξ²/k  (bond capacitance)
             → Z = √(L/C) = √(mk)/ξ²  (characteristic impedance)

Z_topo is defined as the NORMALIZED R-group shunt impedance:

    Z_topo = Z_R-group / Z_backbone

where:
    Z_backbone = √(L_bb / C_bb)  — backbone peptide unit
    Z_R-group  = √(L_R  / C_R )  — sidechain stub

This replaces the bridged constants with ab initio computed values.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mechanics'))

from spice_organic_mapper import (
    get_inductance, get_capacitance, ATOMIC_INDUCTANCE, COVALENT_CAPACITANCE,
    BOND_FORCE_CONSTANTS, XI_TOPO_SQ
)
from ave.solvers.protein_bond_constants import Z_TOPO, Q_BACKBONE
from ave.core.universal_operators import universal_impedance

# ═══════════════════════════════════════════════════════════════
# 1. BACKBONE IMPEDANCE
# ═══════════════════════════════════════════════════════════════
#
# The backbone repeating unit (peptide plane → Cα):
#   Cα — C(=O) — N(-H) — Cα
#
# Three atoms in the unit contribute series inductance:
#   L_bb = L(Cα) + L(C') + L(N)  [all carbon/nitrogen]
#
# Three bonds contribute shunt capacitance (parallel from backbone):
#   C_bb = C(C-C) + C(C-N) + C(C-N)   [Cα-C', C'-N, N-Cα]
#
# Characteristic impedance:
#   Z_bb = √(L_bb / C_bb)

L_Ca = get_inductance('C')   # Cα (carbon)
L_Cp = get_inductance('C')   # C' (carbonyl carbon) — same mass as Cα
L_N  = get_inductance('N')   # Backbone nitrogen

C_CaC = get_capacitance('C-C')   # Cα—C' bond
C_CN  = get_capacitance('C-N')   # C'—N bond (partial double)
C_NCa = get_capacitance('C-N')   # N—Cα bond

L_backbone = L_Ca + L_Cp + L_N
C_backbone = C_CaC + C_CN + C_NCa

Z_backbone = universal_impedance(L_backbone, C_backbone)

print("=" * 70)
print("  Z_topo First-Principles Derivation")
print("=" * 70)
print(f"\n  Backbone unit: Cα—C'(=O)—N(—H)—Cα")
print(f"    L_backbone = L(Cα) + L(C') + L(N)")
print(f"             = {L_Ca*1e15:.3f} + {L_Cp*1e15:.3f} + {L_N*1e15:.3f} fH")
print(f"             = {L_backbone*1e15:.3f} fH")
print(f"    C_backbone = C(C-C) + C(C-N) + C(C-N)")
print(f"             = {C_CaC*1e18:.3f} + {C_CN*1e18:.3f} + {C_NCa*1e18:.3f} aF")
print(f"             = {C_backbone*1e18:.3f} aF")
print(f"    Z_backbone = √(L/C) = {Z_backbone:.3f} Ω")


# ═══════════════════════════════════════════════════════════════
# 2. R-GROUP IMPEDANCE FOR ALL 20 AMINO ACIDS
# ═══════════════════════════════════════════════════════════════
#
# Each sidechain is a stub of atoms connected by bonds.
# L_R = Σ L(atom_i) for all atoms in the R-group
# C_R = Σ C(bond_j) for all bonds in the R-group
# Z_R = √(L_R / C_R)
#
# The atoms/bonds are defined by the molecular structure of each AA.
# We count only the SIDECHAIN atoms (not backbone, not Hα).

# R-group definitions: {'atoms': {element: count}, 'bonds': {bond_type: count}}
# This is the molecular topology of each sidechain
RGROUP_TOPOLOGY = {
    'G': {  # Glycine: R = H (single hydrogen)
        'atoms': {'H': 1},
        'bonds': {'C-H': 1},  # Cα—H bond
    },
    'A': {  # Alanine: R = CH₃
        'atoms': {'C': 1, 'H': 3},
        'bonds': {'C-C': 1, 'C-H': 3},
    },
    'V': {  # Valine: R = CH(CH₃)₂
        'atoms': {'C': 3, 'H': 7},
        'bonds': {'C-C': 3, 'C-H': 7},
    },
    'L': {  # Leucine: R = CH₂CH(CH₃)₂
        'atoms': {'C': 4, 'H': 9},
        'bonds': {'C-C': 4, 'C-H': 9},
    },
    'I': {  # Isoleucine: R = CH(CH₃)(CH₂CH₃)
        'atoms': {'C': 4, 'H': 9},
        'bonds': {'C-C': 4, 'C-H': 9},
    },
    'P': {  # Proline: R = —CH₂CH₂CH₂— (cyclic, bonded to N)
        'atoms': {'C': 3, 'H': 6},
        'bonds': {'C-C': 2, 'C-N': 1, 'C-H': 6},  # ring includes C-N bond back to backbone
    },
    'F': {  # Phenylalanine: R = CH₂—C₆H₅
        'atoms': {'C': 7, 'H': 7},
        'bonds': {'C-C': 4, 'C=C': 3, 'C-H': 7},  # benzene ring: 3 C-C + 3 C=C
    },
    'W': {  # Tryptophan: R = CH₂—(indole ring: C₈NH₅)
        'atoms': {'C': 9, 'N': 1, 'H': 8},
        'bonds': {'C-C': 4, 'C=C': 4, 'C-N': 1, 'N-H': 1, 'C-H': 6},
    },
    'M': {  # Methionine: R = CH₂CH₂SCH₃
        'atoms': {'C': 3, 'H': 7, 'S': 1},
        'bonds': {'C-C': 2, 'C-S': 2, 'C-H': 7},
    },
    'S': {  # Serine: R = CH₂OH
        'atoms': {'C': 1, 'H': 3, 'O': 1},
        'bonds': {'C-C': 1, 'C-H': 2, 'C-O': 1, 'O-H': 1},
    },
    'T': {  # Threonine: R = CH(OH)CH₃
        'atoms': {'C': 2, 'H': 5, 'O': 1},
        'bonds': {'C-C': 2, 'C-H': 4, 'C-O': 1, 'O-H': 1},
    },
    'C': {  # Cysteine: R = CH₂SH
        'atoms': {'C': 1, 'H': 3, 'S': 1},
        'bonds': {'C-C': 1, 'C-H': 2, 'C-S': 1, 'S-H': 1},
    },
    'Y': {  # Tyrosine: R = CH₂—C₆H₄—OH
        'atoms': {'C': 7, 'H': 7, 'O': 1},
        'bonds': {'C-C': 4, 'C=C': 3, 'C-H': 6, 'C-O': 1, 'O-H': 1},
    },
    'H': {  # Histidine: R = CH₂—(imidazole: C₃H₃N₂)
        'atoms': {'C': 4, 'H': 5, 'N': 2},
        'bonds': {'C-C': 2, 'C=C': 1, 'C-N': 2, 'C-H': 4, 'N-H': 1},
    },
    'D': {  # Aspartate: R = CH₂COOH (ionised: CH₂COO⁻)
        'atoms': {'C': 2, 'H': 2, 'O': 2},
        'bonds': {'C-C': 2, 'C-H': 2, 'C=O': 1, 'C-O': 1},
    },
    'E': {  # Glutamate: R = CH₂CH₂COOH (ionised: CH₂CH₂COO⁻)
        'atoms': {'C': 3, 'H': 4, 'O': 2},
        'bonds': {'C-C': 3, 'C-H': 4, 'C=O': 1, 'C-O': 1},
    },
    'N': {  # Asparagine: R = CH₂CONH₂
        'atoms': {'C': 2, 'H': 4, 'N': 1, 'O': 1},
        'bonds': {'C-C': 2, 'C-H': 2, 'C=O': 1, 'C-N': 1, 'N-H': 2},
    },
    'Q': {  # Glutamine: R = CH₂CH₂CONH₂
        'atoms': {'C': 3, 'H': 6, 'N': 1, 'O': 1},
        'bonds': {'C-C': 3, 'C-H': 4, 'C=O': 1, 'C-N': 1, 'N-H': 2},
    },
    'K': {  # Lysine: R = (CH₂)₄NH₃⁺
        'atoms': {'C': 4, 'H': 11, 'N': 1},
        'bonds': {'C-C': 4, 'C-H': 8, 'C-N': 1, 'N-H': 3},
    },
    'R': {  # Arginine: R = (CH₂)₃NHC(=NH)NH₂
        'atoms': {'C': 4, 'H': 11, 'N': 3},
        'bonds': {'C-C': 3, 'C-H': 6, 'C-N': 3, 'N-H': 5},
    },
}


def compute_z_rgroup(topology):
    """Compute characteristic impedance of an R-group from its atom/bond counts."""
    # Total inductance: sum over all atoms
    L_total = 0.0
    for elem, count in topology['atoms'].items():
        L_total += count * get_inductance(elem)
    
    # Total capacitance: sum over all bonds (parallel stubs)
    C_total = 0.0
    for bond, count in topology['bonds'].items():
        C_total += count * get_capacitance(bond)
    
    Z_R = universal_impedance(L_total, C_total)
    return L_total, C_total, Z_R


# ═══════════════════════════════════════════════════════════════
# 3. COMPUTE Z_topo FOR ALL 20 AMINO ACIDS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  R-Group Impedances and Derived Z_topo")
print(f"{'='*70}")
print(f"\n  {'AA':<4} {'L_R (fH)':>10} {'C_R (aF)':>10} {'Z_R (Ω)':>10}"
      f" {'Z_topo':>8} {'Current':>8} {'Δ%':>6}")
print(f"  {'-'*60}")

derived_z_topo = {}
for aa in 'GAVLIMFWPSTCYHDNEQKR':
    topo = RGROUP_TOPOLOGY[aa]
    L_R, C_R, Z_R = compute_z_rgroup(topo)
    z_topo_derived = Z_R / Z_backbone
    
    # Current bridged value
    z_current = abs(Z_TOPO[aa])
    pct_diff = 100 * (z_topo_derived - z_current) / z_current if z_current > 0 else 0
    
    derived_z_topo[aa] = z_topo_derived
    
    print(f"  {aa:2s}   {L_R*1e15:10.3f}  {C_R*1e18:10.3f}  {Z_R:10.3f}"
          f"  {z_topo_derived:8.4f}  {z_current:8.4f}  {pct_diff:+6.1f}%")

# ═══════════════════════════════════════════════════════════════
# 4. GENERATE UPDATED Z_TOPO DICTIONARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  Derived Z_TOPO dictionary (for protein_bond_constants.py)")
print(f"{'='*70}")
print(f"\n  Q_BACKBONE = {Q_BACKBONE}")
print(f"\n  Z_TOPO = {{")

# Charge classification for X component
neg_charge = {'D', 'E'}     # X = -R/Q
pos_charge = {'K', 'R'}     # X = +R/Q
half_prot  = {'H'}          # X = +R/(2Q)
polar_pos  = {'S', 'T', 'N', 'Q'}  # X = +R/(2Q)
polar_neg  = {'C', 'Y'}     # X = -R/(2Q)

for aa in 'GAVLIMFWPDEKRHSTCYNQ':
    R = derived_z_topo[aa]
    if aa in neg_charge:
        X = -R / Q_BACKBONE
        tag = "Neg. charge"
    elif aa in pos_charge:
        X = R / Q_BACKBONE
        tag = "Pos. charge"
    elif aa in half_prot:
        X = R / (2 * Q_BACKBONE)
        tag = "Half-protonated"
    elif aa in polar_pos:
        X = R / (2 * Q_BACKBONE)
        tag = "Polar uncharged"
    elif aa in polar_neg:
        X = -R / (2 * Q_BACKBONE)
        tag = "Polar uncharged"
    else:
        X = 0.0
        tag = "Hydrophobic"
    
    z_mag = abs(complex(R, X))
    print(f"      '{aa}': {R:.4f} {'+' if X >= 0 else ''}{X:.4f}j,  # |Z|={z_mag:.4f}  {tag}")

print(f"  }}")

print(f"\n  Axiom chain: Axioms 1-4 → ξ_topo → L=m/ξ², C=ξ²/k → Z=√(L/C) → Z_topo=Z_R/Z_bb")
print(f"  Zero free parameters. Zero empirical fits.")
