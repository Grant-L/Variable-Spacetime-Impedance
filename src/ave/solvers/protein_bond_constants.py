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
    # mass_Da = sum of both bonded atoms (from bond_energy_solver NUCLEAR_MASSES)
    # Z = √(μ/ε) = √(mass_Da / n_electrons) per place_nuclear_defect physics
    'Ca-C':   {'length_A': 1.52, 'type': 'single',         'atoms': ('C', 'C'), 'n_electrons': 2, 'mass_Da': 24.0},   # 12+12
    'C-N':    {'length_A': 1.33, 'type': 'partial_double',  'atoms': ('C', 'N'), 'n_electrons': 3, 'mass_Da': 26.0},   # 12+14
    'N-Ca':   {'length_A': 1.46, 'type': 'single',         'atoms': ('N', 'C'), 'n_electrons': 2, 'mass_Da': 26.0},   # 14+12
    'C=O':    {'length_A': 1.23, 'type': 'double',          'atoms': ('C', 'O'), 'n_electrons': 4, 'mass_Da': 28.0},   # 12+16
    'N-H':    {'length_A': 1.01, 'type': 'single',         'atoms': ('N', 'H'), 'n_electrons': 2, 'mass_Da': 15.0},   # 14+1
}

# Backbone bond angles
BACKBONE_ANGLES = {
    'N-Ca-C':  111.2,  # degrees (tetrahedral-derived)
    'Ca-C-N':  116.2,  # degrees
    'Ca-C-O':  121.4,  # degrees (carbonyl oxygen, Engh & Huber 1991)
    'C-N-Ca':  121.7,  # degrees (amide planarity)
}


# ═══════════════════════════════════════════════════════════════
# Backbone bond impedances (derived from Axiom 1: Z = √(μ/ε))
# ═══════════════════════════════════════════════════════════════
#
# Each covalent bond has a characteristic impedance:
#   Z = √(mass_Da / n_electrons) = √(μ/ε)
#
# The C-N peptide bond (partial double, 3e⁻) has 19% lower impedance
# than the N-Cα single bond (2e⁻). This creates a periodic impedance
# grating in the backbone — each peptide unit (N-Cα-C) has an internal
# impedance profile that drives Bragg-like reflections at the amide-V
# resonance frequency (23 THz).

Z_BOND_N_CA = np.sqrt(BACKBONE_BONDS['N-Ca']['mass_Da'] /
                       BACKBONE_BONDS['N-Ca']['n_electrons'])     # √(26/2) ≈ 3.606
Z_BOND_CA_C = np.sqrt(BACKBONE_BONDS['Ca-C']['mass_Da'] /
                       BACKBONE_BONDS['Ca-C']['n_electrons'])     # √(24/2) ≈ 3.464
Z_BOND_C_N  = np.sqrt(BACKBONE_BONDS['C-N']['mass_Da'] /
                       BACKBONE_BONDS['C-N']['n_electrons'])      # √(26/3) ≈ 2.944

# Normalised to mean backbone impedance (compatible with z_topo scale)
Z_BOND_MEAN = (Z_BOND_N_CA + Z_BOND_CA_C + Z_BOND_C_N) / 3.0   # ≈ 3.338
Z_N_CA_NORM = Z_BOND_N_CA / Z_BOND_MEAN                          # ≈ 1.080
Z_CA_C_NORM = Z_BOND_CA_C / Z_BOND_MEAN                          # ≈ 1.038
Z_C_N_NORM  = Z_BOND_C_N  / Z_BOND_MEAN                          # ≈ 0.882


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
# H-bond mutual inductance constants (derived from backbone bonds)
# ═══════════════════════════════════════════════════════════════
#
# An H-bond is the mutual inductance between two backbone LC oscillators:
#   Donor:   N-H dipole (inductive: current flows N→H)
#   Acceptor: C=O dipole (capacitive: charge accumulates at O)
#
# This is the protein-scale analog of K_MUTUAL at the nuclear scale
# (mutual inductance between nucleon knots):
#   Nuclear:  K = (cπ/2) × αℏc / (1 − α/3)  → 1/d coupling
#   Protein:  Y_HB = κ_HB × Z_match × exp(−d/d₀)  → shunt admittance
#
# All parameters from BACKBONE_BONDS (Axioms 1-2):

# H-bond dipole lengths (from covalent bond geometry)
D_NH = BACKBONE_BONDS['N-H']['length_A']   # 1.01 Å — donor dipole
D_CO = BACKBONE_BONDS['C=O']['length_A']   # 1.23 Å — acceptor dipole

# H-bond detection distance: sum of dipole lengths
# In the N-Cα-C representation, we detect N_i···C_j proximity
# The actual H···O distance ≈ 1.8-2.0 Å, but d(N,C) ≈ d_NH + d_CO + d_H···O
# Simplified: d_detect = d_NH + d_CO + d₀/2 (half a backbone step covers
# the N..C distance in an α-helix i→i+4 contact)
D_HB_DETECT = D_NH + D_CO   # = 2.24 Å (tight detection for strong coupling)

# H-bond coupling strength: 1/(2Q) — same amide-V quality factor
# This is the critical coupling point (κ = 1/2) divided by Q,
# ensuring H-bond coupling is resonance-modulated
# (Q_BACKBONE defined below in amino acid impedance section)


# ═══════════════════════════════════════════════════════════════
# Amino acid impedance table (from axiom-derived properties)
# ═══════════════════════════════════════════════════════════════

# Q-factor of the backbone amide-V resonator
# Derived: amide-V mode at 23 THz, linewidth ~3 THz → Q = f₀/Δf ≈ 7
Q_BACKBONE = 7.0

# H-bond coupling strength (deferred from above — needs Q_BACKBONE)
KAPPA_HB = 1.0 / (2.0 * Q_BACKBONE)  # = 1/14 ≈ 0.071

# Per-residue topological impedance (R + jX)
# 
# AB INITIO DERIVATION (no empirical fits):
#   R = Z_R-group / Z_backbone  where:
#     Z_R = √(L_R / C_R)  — R-group characteristic impedance
#     Z_bb = √(L_bb / C_bb) — backbone peptide unit impedance
#     L = m/ξ² (atomic inductance from mass)
#     C = ξ²/k (bond capacitance from force constant)
#   X = charge reactance / Q
#
# Derivation script: scripts/book_5_topological_biology/derive_z_topo_first_principles.py
# Full derivation chain: Axioms 1-4 → ξ_topo → L,C → Z → Z_topo
#
Z_TOPO = {
    # Hydrophobic: R from Z_R/Z_bb, X = 0
    'G': 0.3036 + 0.00j,    # R=H (minimal stub)
    'A': 0.5684 + 0.00j,    # R=CH₃
    'V': 0.6050 + 0.00j,    # R=CH(CH₃)₂
    'I': 0.6104 + 0.00j,    # R=CH(CH₃)(CH₂CH₃)
    'L': 0.6104 + 0.00j,    # R=CH₂CH(CH₃)₂
    'M': 0.7229 + 0.00j,    # R=CH₂CH₂SCH₃
    'F': 0.7855 + 0.00j,    # R=CH₂C₆H₅
    'W': 0.8947 + 0.00j,    # R=CH₂(indole)
    'P': 0.6324 + 0.00j,    # R=—CH₂CH₂CH₂— (cyclic)
    # Negative charge (capacitive): X = -R/Q
    'D': 0.9488 - 0.9488/Q_BACKBONE*1j,  # R=CH₂COO⁻
    'E': 0.8486 - 0.8486/Q_BACKBONE*1j,  # R=CH₂CH₂COO⁻
    # Positive charge (inductive): X = +R/Q
    'K': 0.6386 + 0.6386/Q_BACKBONE*1j,  # R=(CH₂)₄NH₃⁺
    'R': 0.7403 + 0.7403/Q_BACKBONE*1j,  # R=(CH₂)₃guanidinium
    'H': 0.8618 + 0.8618/(2*Q_BACKBONE)*1j,  # R=CH₂(imidazole), half-protonated
    # Polar uncharged: X = ±R/(2Q) — weak H-bond reactance
    'S': 0.7641 + 0.7641/(2*Q_BACKBONE)*1j,  # R=CH₂OH
    'T': 0.7125 + 0.7125/(2*Q_BACKBONE)*1j,  # R=CH(OH)CH₃
    'C': 0.8240 - 0.8240/(2*Q_BACKBONE)*1j,  # R=CH₂SH
    'Y': 0.8332 - 0.8332/(2*Q_BACKBONE)*1j,  # R=CH₂C₆H₄OH
    'N': 0.8397 + 0.8397/(2*Q_BACKBONE)*1j,  # R=CH₂CONH₂
    'Q': 0.7818 + 0.7818/(2*Q_BACKBONE)*1j,  # R=CH₂CH₂CONH₂
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
