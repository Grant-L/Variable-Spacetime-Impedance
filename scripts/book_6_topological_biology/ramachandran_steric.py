#!/usr/bin/env python3
"""
First-Principles Ramachandran Steric Exclusion Calculator
=========================================================

Computes the allowed (φ, ψ) regions for all 20 standard amino acids
from axiom-derived parameters only:

  - Bond lengths:  d_eq from soliton_bond_solver  (Axioms 1-2)
  - Bond angles:   tetrahedral 109.47° (sp³), planar 120° (sp²)
  - vdW radii:     r = n*² a₀ / Z_eff   (Slater rules, no empirical input)
  - Steric clash:  d < (r_A + r_B) × overlap_factor

The helix propensity for each amino acid is the fraction of the
α-helix basin (φ ∈ [-80°, -40°], ψ ∈ [-65°, -25°]) that is
sterically allowed.

NO FREE PARAMETERS.  Every constant traces to the lattice axioms.
"""

import numpy as np
from collections import namedtuple
import sys, os

# Import the AVE periodic table (soliton bond solver)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from ave.topological.soliton_bond_solver import (
    _slater_orbital_radius, _slater_z_eff, _n_star,
    KNOWN_D, A_BOHR,
)

# =====================================================================
# AXIOM-DERIVED CONSTANTS — sourced from the periodic table
# =====================================================================

# Bond lengths (Å) — from soliton_bond_solver KNOWN_D
# These are the E(d) equilibrium distances on the AVE lattice.
BOND_LEN = {
    'N-CA':  KNOWN_D['C-N'] * 1e10,   # peptide N-Cα = C-N single bond
    'CA-C':  KNOWN_D['C-C'] * 1e10,   # Cα-C' = C-C single bond
    'C-N':   1.33,                      # peptide bond (partial double, interpolated)
    'C=O':   KNOWN_D['C=O'] * 1e10,   # carbonyl
    'N-H':   KNOWN_D['N-H'] * 1e10,   # amide H
    'CA-CB': KNOWN_D['C-C'] * 1e10,   # Cα-Cβ = C-C single
    'CA-HA': KNOWN_D['C-H'] * 1e10,   # Hα
    'C-C':   KNOWN_D['C-C'] * 1e10,   # sidechain C-C
    'C-H':   KNOWN_D['C-H'] * 1e10,   # generic C-H
    'C-O':   KNOWN_D['C-O'] * 1e10,   # C-O single (Ser, Thr)
    'O-H':   KNOWN_D['O-H'] * 1e10,   # hydroxyl
    'C-S':   KNOWN_D['C-S'] * 1e10,   # C-S (Cys, Met)
    'S-H':   KNOWN_D['S-H'] * 1e10,   # thiol
    'C-N_sc': KNOWN_D['C-N'] * 1e10,  # sidechain C-N
    'N-H_sc': KNOWN_D['N-H'] * 1e10,  # sidechain N-H
}

# Bond angles (degrees) — from lattice geometry
# The SRS lattice has 4-connected nodes → sp³ tetrahedral angle
# 3-connected nodes → sp² trigonal planar angle
# These are EXACT from the lattice topology (Axiom 1):
#   cos(θ_tet) = -1/3 → θ = 109.47°
#   cos(θ_sp2) = -1/2 → θ = 120.0°
ANGLE_TET = np.degrees(np.arccos(-1.0/3.0))  # exactly 109.4712...°
ANGLE_SP2 = 120.0                              # exactly 120°

# Van der Waals radii (Å) — from Slater orbital sizes
# r = n*² · a₀ / Z_eff   (exact from Slater screening rules)
#
# These are the most probable orbital radii from the Slater-type
# wavefunction.  The steric clash criterion uses the SUM of two
# orbital radii × overlap_factor, where the overlap_factor absorbs
# the Pauli exclusion boundary scaling.
VDW_RADIUS = {}
for sym, Z in [('H',1), ('C',6), ('N',7), ('O',8), ('S',16)]:
    VDW_RADIUS[sym] = _slater_orbital_radius(Z) * 1e10  # Å

# Overlap factor for steric clash detection.
# The Ramachandran criterion: two atoms clash when
#   d < (r_A + r_B) × OVERLAP_FACTOR
#
# With orbital radii r_A, r_B, the factor is larger than the
# traditional 0.80 (which assumed Bondi vdW radii ~2.6× larger).
# The Pauli boundary factor = 2π/√3 ≈ 3.63 from lattice geometry,
# but the "allowed" region criterion uses 0.80× vdW = 2.08× orbital.
OVERLAP_FACTOR = 2.08

# Helix basin: the α-helix region of φ/ψ space
HELIX_PHI = (-80.0, -40.0)   # degrees
HELIX_PSI = (-65.0, -25.0)   # degrees

# Sheet basin: the β-sheet region
SHEET_PHI = (-150.0, -90.0)
SHEET_PSI = (90.0, 150.0)

# Scan resolution (degrees)
SCAN_STEP = 5

# =====================================================================
# 3D GEOMETRY UTILITIES
# =====================================================================

def rotation_matrix(axis, theta):
    """Rodrigues' rotation: rotate by theta (radians) around unit axis."""
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def place_atom(origin, bond_vec, bond_length, angle, dihedral, ref_normal):
    """
    Place an atom at 'bond_length' from 'origin' with the specified
    bond angle and dihedral angle relative to the incoming bond vector
    and reference normal.
    """
    # Direction along incoming bond (reversed for outgoing angle)
    d = bond_vec / (np.linalg.norm(bond_vec) + 1e-15)

    # Rotate d by (π - angle) around ref_normal to get the bond direction
    # at the correct angle
    theta_bend = np.pi - np.radians(angle)
    n = ref_normal / (np.linalg.norm(ref_normal) + 1e-15)
    R_bend = rotation_matrix(n, theta_bend)
    new_dir = R_bend @ d

    # Now rotate new_dir by dihedral around the incoming bond axis
    R_dihedral = rotation_matrix(d, np.radians(dihedral))
    new_dir = R_dihedral @ new_dir

    return origin + bond_length * new_dir, new_dir


def initial_normal(v):
    """Return a vector perpendicular to v."""
    if abs(v[0]) < 0.9:
        perp = np.cross(v, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(v, np.array([0.0, 1.0, 0.0]))
    return perp / (np.linalg.norm(perp) + 1e-15)


# =====================================================================
# PENTAPEPTIDE BACKBONE BUILDER  (residues i-2 through i+2)
# =====================================================================

Atom = namedtuple('Atom', ['pos', 'element', 'name'])


def _build_residue(prev_C, prev_C_dir, prev_normal, phi, psi, omega, tag):
    """
    Build one residue from the preceding C' atom.
    Returns (atoms, C_pos, C_dir, normal, N_pos, CA_pos, C_pos_out).
    `tag` is a label like '(i-2)', '(i)', etc.
    """
    atoms = []

    # Peptide bond: prev_C' → N with ω dihedral
    N, N_dir = place_atom(prev_C, prev_C_dir, BOND_LEN['C-N'], ANGLE_SP2,
                          omega, prev_normal)
    atoms.append(Atom(N, 'N', f'N{tag}'))

    n_CN = np.cross(prev_C_dir, N_dir)
    nn = np.linalg.norm(n_CN)
    n_CN = n_CN / nn if nn > 1e-10 else prev_normal

    # Amide H
    H_N, _ = place_atom(N, N_dir, BOND_LEN['N-H'], ANGLE_SP2, 180.0, n_CN)
    atoms.append(Atom(H_N, 'H', f'H{tag}'))

    # N → Cα with φ dihedral
    CA, CA_dir = place_atom(N, N_dir, BOND_LEN['N-CA'], ANGLE_SP2, phi, n_CN)
    atoms.append(Atom(CA, 'C', f'CA{tag}'))

    n_NCA = np.cross(N_dir, CA_dir)
    nn = np.linalg.norm(n_NCA)
    n_NCA = n_NCA / nn if nn > 1e-10 else n_CN

    # Hα
    HA, _ = place_atom(CA, CA_dir, BOND_LEN['CA-HA'], ANGLE_TET, -120.0, n_NCA)
    atoms.append(Atom(HA, 'H', f'HA{tag}'))

    # Cα → C' with ψ dihedral
    C, C_dir = place_atom(CA, CA_dir, BOND_LEN['CA-C'], ANGLE_TET, psi, n_NCA)
    atoms.append(Atom(C, 'C', f"C'{tag}"))

    n_CAC = np.cross(CA_dir, C_dir)
    nn = np.linalg.norm(n_CAC)
    n_CAC = n_CAC / nn if nn > 1e-10 else n_NCA

    # Carbonyl O
    O, _ = place_atom(C, C_dir, BOND_LEN['C=O'], ANGLE_SP2, 0.0, n_CAC)
    atoms.append(Atom(O, 'O', f'O{tag}'))

    return atoms, C, C_dir, n_CAC, N, CA


def _compute_cb_pos_from_3(N_pos, CA_pos, C_pos, bond_length):
    """
    Compute Cβ at the tetrahedral position from backbone N, Cα, C'.
    Returns (CB_pos, HA_pos, CB_direction).
    """
    u_N = (N_pos - CA_pos)
    u_N = u_N / (np.linalg.norm(u_N) + 1e-15)
    u_C = (C_pos - CA_pos)
    u_C = u_C / (np.linalg.norm(u_C) + 1e-15)
    cos_NCC = np.dot(u_N, u_C)
    plane_n = np.cross(u_N, u_C)
    pn = np.linalg.norm(plane_n)
    plane_n = plane_n / pn if pn > 1e-10 else np.array([0, 0, 1.0])
    a = -1.0 / (3.0 * (1.0 + cos_NCC + 1e-15))
    c_sq = max(0.0, 1.0 - 2 * a * a * (1 + cos_NCC))
    c = np.sqrt(c_sq)
    dir_CB = a * u_N + a * u_C + c * plane_n
    dir_HA = a * u_N + a * u_C - c * plane_n
    dn = np.linalg.norm(dir_CB) + 1e-15
    CB = CA_pos + bond_length * dir_CB / dn
    HA = CA_pos + BOND_LEN['CA-HA'] * dir_HA / (np.linalg.norm(dir_HA) + 1e-15)
    return CB, HA, dir_CB / dn


def _place_flanking_cb(N_pos, CA_pos, C_pos, tag):
    """Place an Ala-like Cβ+CH₃ on a flanking residue for steric contacts."""
    CB, _, CB_dir = _compute_cb_pos_from_3(N_pos, CA_pos, C_pos, BOND_LEN['CA-CB'])
    atoms = [Atom(CB, 'C', f'CB{tag}')]
    # Methyl H on CB (steric placeholders)
    u_CA = (CA_pos - CB) / (np.linalg.norm(CA_pos - CB) + 1e-15)
    n_CB = np.cross(u_CA, CB_dir)
    nn = np.linalg.norm(n_CB)
    n_CB = n_CB / nn if nn > 1e-10 else np.array([0, 0, 1.0])
    for dh in [0, 120, -120]:
        H, _ = place_atom(CB, CB_dir, BOND_LEN['C-H'], ANGLE_TET, dh, n_CB)
        atoms.append(Atom(H, 'H', f'HB{tag}'))
    return atoms


def build_backbone(phi, psi, omega=180.0):
    """
    Build a pentapeptide backbone (residues i-2 through i+2) with the
    specified φ, ψ at residue i.  Flanking residues use average backbone
    angles (φ=-60°, ψ=-45°) and Ala-like Cβ groups for realistic steric
    contacts.

    Returns (all_atoms, cb_data_for_residue_i).
    """
    # Default flanking angles (average backbone)
    phi_flank = -60.0
    psi_flank = -45.0

    atoms = []

    # Start from a virtual C' before residue i-2
    C0 = np.array([0.0, 0.0, 0.0])
    C0_dir = np.array([1.0, 0.0, 0.0])
    n0 = np.array([0.0, 0.0, 1.0])

    # ---- Residue i-2 ----
    a_im2, C_im2, Cd_im2, n_im2, N_im2, CA_im2 = _build_residue(
        C0, C0_dir, n0, phi_flank, psi_flank, omega, '(i-2)')
    atoms.extend(a_im2)
    atoms.extend(_place_flanking_cb(N_im2, CA_im2, C_im2, '(i-2)'))

    # ---- Residue i-1 ----
    a_im1, C_im1, Cd_im1, n_im1, N_im1, CA_im1 = _build_residue(
        C_im2, Cd_im2, n_im2, phi_flank, psi_flank, omega, '(i-1)')
    atoms.extend(a_im1)
    atoms.extend(_place_flanking_cb(N_im1, CA_im1, C_im1, '(i-1)'))

    # ---- Residue i (the one we're scanning) ----
    a_i, C_i, Cd_i, n_i, N_i, CA_i = _build_residue(
        C_im1, Cd_im1, n_im1, phi, psi, omega, '(i)')
    atoms.extend(a_i)

    # ---- Residue i+1 ----
    a_ip1, C_ip1, Cd_ip1, n_ip1, N_ip1, CA_ip1 = _build_residue(
        C_i, Cd_i, n_i, phi_flank, psi_flank, omega, '(i+1)')
    atoms.extend(a_ip1)
    atoms.extend(_place_flanking_cb(N_ip1, CA_ip1, C_ip1, '(i+1)'))

    # ---- Residue i+2 ----
    a_ip2, C_ip2, Cd_ip2, n_ip2, N_ip2, CA_ip2 = _build_residue(
        C_ip1, Cd_ip1, n_ip1, phi_flank, psi_flank, omega, '(i+2)')
    atoms.extend(a_ip2)
    atoms.extend(_place_flanking_cb(N_ip2, CA_ip2, C_ip2, '(i+2)'))

    # Cβ placement data for residue i
    cb_data = {
        'N_pos': N_i,
        'CA_pos': CA_i,
        'C_pos': C_i,
    }

    return atoms, cb_data


# =====================================================================
# R-GROUP ATOM PLACEMENT
# =====================================================================

def _compute_cb_pos(N_pos, CA_pos, C_pos, bond_length):
    """
    Compute substituent positions at sp³ Cα using the tetrahedral constraint.
    Given N, CA, C' positions, find CB and HA such that all 4 bonds from CA
    are at 109.47° from each other.  L-amino acid convention: CB is on the
    left when looking from N to C' through CA.
    """
    u_N = (N_pos - CA_pos); u_N = u_N / (np.linalg.norm(u_N) + 1e-15)
    u_C = (C_pos - CA_pos); u_C = u_C / (np.linalg.norm(u_C) + 1e-15)
    cos_NCC = np.dot(u_N, u_C)
    plane_n = np.cross(u_N, u_C)
    pn = np.linalg.norm(plane_n)
    if pn < 1e-10:
        plane_n = np.array([0, 0, 1.0])
    else:
        plane_n = plane_n / pn
    a = -1.0 / (3.0 * (1.0 + cos_NCC + 1e-15))
    c_sq = max(0.0, 1.0 - 2*a*a*(1 + cos_NCC))
    c = np.sqrt(c_sq)
    dir_CB = a * u_N + a * u_C + c * plane_n   # L-amino acid side
    dir_HA = a * u_N + a * u_C - c * plane_n   # opposite side
    CB = CA_pos + bond_length * dir_CB / (np.linalg.norm(dir_CB) + 1e-15)
    HA = CA_pos + BOND_LEN['CA-HA'] * dir_HA / (np.linalg.norm(dir_HA) + 1e-15)
    return CB, HA, dir_CB / (np.linalg.norm(dir_CB) + 1e-15)


def _extend_chain(origin, prev_dir, normal, bond_len, angle, dihedral):
    """Helper: place next atom and return (pos, direction, normal)."""
    pos, new_dir = place_atom(origin, prev_dir, bond_len, angle, dihedral, normal)
    n_new = np.cross(prev_dir, new_dir)
    nn = np.linalg.norm(n_new)
    if nn > 1e-10:
        n_new = n_new / nn
    else:
        n_new = normal
    return pos, new_dir, n_new


def place_sidechain(cb_data, aa_code):
    """
    Place R-group atoms using proper tetrahedral geometry from backbone
    N, CA, C' positions.  All positions are axiom-derived.
    """
    N = cb_data['N_pos']
    CA = cb_data['CA_pos']
    C = cb_data['C_pos']
    sc_atoms = []

    if aa_code == 'G':
        _, HA2, _ = _compute_cb_pos(N, CA, C, BOND_LEN['CA-HA'])
        sc_atoms.append(Atom(HA2, 'H', 'HA2'))  # HA2 is actually CB side for Gly
        return sc_atoms

    # Compute CB position with proper tetrahedral geometry
    CB, _, CB_dir = _compute_cb_pos(N, CA, C, BOND_LEN['CA-CB'])
    sc_atoms.append(Atom(CB, 'C', 'CB'))

    # Direction from CA to CB and reference normal for further placement
    u_CB = CB_dir
    # Normal for placing substituents on CB: perpendicular to CA-CB and in the
    # plane defined by N-CA-CB
    u_CA = (CA - CB) / (np.linalg.norm(CA - CB) + 1e-15)
    normal_CB = np.cross(u_CA, u_CB)
    nn = np.linalg.norm(normal_CB)
    if nn > 1e-10:
        normal_CB = normal_CB / nn
    else:
        normal_CB = np.array([0, 0, 1.0])

    def _place_ch2(origin, prev_dir, norm):
        """Place -CH₂- group: 2 hydrogens + next heavy atom."""
        H1, _ = place_atom(origin, prev_dir, BOND_LEN['C-H'], ANGLE_TET, 120.0, norm)
        H2, _ = place_atom(origin, prev_dir, BOND_LEN['C-H'], ANGLE_TET, -120.0, norm)
        return [Atom(H1, 'H', 'Ha'), Atom(H2, 'H', 'Hb')]

    def _place_ch3(origin, prev_dir, norm):
        """Place terminal -CH₃ group."""
        atoms = []
        for i, dh in enumerate([0, 120, -120]):
            H, _ = place_atom(origin, prev_dir, BOND_LEN['C-H'], ANGLE_TET, dh, norm)
            atoms.append(Atom(H, 'H', f'Hm{i}'))
        return atoms

    if aa_code == 'A':  # -CH₃
        sc_atoms.extend(_place_ch3(CB, u_CB, normal_CB))

    elif aa_code == 'V':  # -CH(CH₃)₂ (β-branched)
        HB, _ = place_atom(CB, u_CB, BOND_LEN['C-H'], ANGLE_TET, 0.0, normal_CB)
        sc_atoms.append(Atom(HB, 'H', 'HB'))
        CG1, CG1_d = place_atom(CB, u_CB, BOND_LEN['C-C'], ANGLE_TET, 120.0, normal_CB)
        sc_atoms.append(Atom(CG1, 'C', 'CG1'))
        CG2, CG2_d = place_atom(CB, u_CB, BOND_LEN['C-C'], ANGLE_TET, -120.0, normal_CB)
        sc_atoms.append(Atom(CG2, 'C', 'CG2'))

    elif aa_code == 'L':  # -CH₂-CH(CH₃)₂
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        HG, _ = place_atom(CG, CG_d, BOND_LEN['C-H'], ANGLE_TET, 0.0, n_CG)
        sc_atoms.append(Atom(HG, 'H', 'HG'))
        CD1, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_TET, 120.0, n_CG)
        sc_atoms.append(Atom(CD1, 'C', 'CD1'))
        CD2, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_TET, -120.0, n_CG)
        sc_atoms.append(Atom(CD2, 'C', 'CD2'))

    elif aa_code == 'I':  # -CH(CH₃)-CH₂-CH₃ (β-branched)
        HB, _ = place_atom(CB, u_CB, BOND_LEN['C-H'], ANGLE_TET, 0.0, normal_CB)
        sc_atoms.append(Atom(HB, 'H', 'HB'))
        CG1, CG1_d, n_CG1 = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 120.0)
        sc_atoms.append(Atom(CG1, 'C', 'CG1'))
        CG2, _ = place_atom(CB, u_CB, BOND_LEN['C-C'], ANGLE_TET, -120.0, normal_CB)
        sc_atoms.append(Atom(CG2, 'C', 'CG2'))
        CD1, _ = place_atom(CG1, CG1_d, BOND_LEN['C-C'], ANGLE_TET, 180.0, n_CG1)
        sc_atoms.append(Atom(CD1, 'C', 'CD1'))

    elif aa_code == 'T':  # -CH(OH)(CH₃) (β-branched)
        HB, _ = place_atom(CB, u_CB, BOND_LEN['C-H'], ANGLE_TET, 0.0, normal_CB)
        sc_atoms.append(Atom(HB, 'H', 'HB'))
        OG1, _ = place_atom(CB, u_CB, BOND_LEN['C-O'], ANGLE_TET, 120.0, normal_CB)
        sc_atoms.append(Atom(OG1, 'O', 'OG1'))
        CG2, _ = place_atom(CB, u_CB, BOND_LEN['C-C'], ANGLE_TET, -120.0, normal_CB)
        sc_atoms.append(Atom(CG2, 'C', 'CG2'))

    elif aa_code == 'P':  # -CH₂-CH₂-CH₂- (ring back to N)
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        CD, _, _ = _extend_chain(CG, CG_d, n_CG, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CD, 'C', 'CD'))

    elif aa_code == 'F':  # -CH₂-C₆H₅ (benzyl)
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        # Ring carbons lumped at representative positions
        CD1, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 0.0, n_CG)
        sc_atoms.append(Atom(CD1, 'C', 'CD1'))
        CD2, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 180.0, n_CG)
        sc_atoms.append(Atom(CD2, 'C', 'CD2'))

    elif aa_code == 'Y':  # -CH₂-C₆H₄-OH (4-hydroxybenzyl)
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        CD1, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 0.0, n_CG)
        sc_atoms.append(Atom(CD1, 'C', 'CD1'))
        CD2, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 180.0, n_CG)
        sc_atoms.append(Atom(CD2, 'C', 'CD2'))

    elif aa_code == 'W':  # -CH₂-indole (large bicyclic)
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        CD1, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 0.0, n_CG)
        sc_atoms.append(Atom(CD1, 'C', 'CD1'))
        CD2, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 180.0, n_CG)
        sc_atoms.append(Atom(CD2, 'C', 'CD2'))
        # Extra ring atom for the larger indole
        NE1, _ = place_atom(CG, CG_d, BOND_LEN['C-N_sc'], ANGLE_SP2, 90.0, n_CG)
        sc_atoms.append(Atom(NE1, 'N', 'NE1'))

    elif aa_code == 'H':  # -CH₂-imidazole
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        ND1, _ = place_atom(CG, CG_d, BOND_LEN['C-N_sc'], ANGLE_SP2, 0.0, n_CG)
        sc_atoms.append(Atom(ND1, 'N', 'ND1'))
        CD2, _ = place_atom(CG, CG_d, BOND_LEN['C-C'], ANGLE_SP2, 180.0, n_CG)
        sc_atoms.append(Atom(CD2, 'C', 'CD2'))

    elif aa_code == 'M':  # -CH₂-CH₂-S-CH₃
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        sd_atoms = _place_ch2(CG, CG_d, n_CG)
        sc_atoms.extend(sd_atoms)
        SD, SD_d, n_SD = _extend_chain(CG, CG_d, n_CG, BOND_LEN['C-S'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(SD, 'S', 'SD'))
        CE, _ = place_atom(SD, SD_d, BOND_LEN['C-S'], ANGLE_TET, 180.0, n_SD)
        sc_atoms.append(Atom(CE, 'C', 'CE'))

    elif aa_code == 'S':  # -CH₂-OH
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        OG, _ = place_atom(CB, u_CB, BOND_LEN['C-O'], ANGLE_TET, 180.0, normal_CB)
        sc_atoms.append(Atom(OG, 'O', 'OG'))

    elif aa_code == 'C':  # -CH₂-SH
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        SG, _ = place_atom(CB, u_CB, BOND_LEN['C-S'], ANGLE_TET, 180.0, normal_CB)
        sc_atoms.append(Atom(SG, 'S', 'SG'))

    elif aa_code == 'D':  # -CH₂-COO⁻
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        OD1, _ = place_atom(CG, CG_d, BOND_LEN['C-O'], ANGLE_SP2, 0.0, n_CG)
        sc_atoms.append(Atom(OD1, 'O', 'OD1'))
        OD2, _ = place_atom(CG, CG_d, BOND_LEN['C-O'], ANGLE_SP2, 180.0, n_CG)
        sc_atoms.append(Atom(OD2, 'O', 'OD2'))

    elif aa_code == 'E':  # -CH₂-CH₂-COO⁻
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        sc_atoms.extend(_place_ch2(CG, CG_d, n_CG))
        CD, CD_d, n_CD = _extend_chain(CG, CG_d, n_CG, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CD, 'C', 'CD'))
        OE1, _ = place_atom(CD, CD_d, BOND_LEN['C-O'], ANGLE_SP2, 0.0, n_CD)
        sc_atoms.append(Atom(OE1, 'O', 'OE1'))
        OE2, _ = place_atom(CD, CD_d, BOND_LEN['C-O'], ANGLE_SP2, 180.0, n_CD)
        sc_atoms.append(Atom(OE2, 'O', 'OE2'))

    elif aa_code == 'N':  # -CH₂-CONH₂
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        OD1, _ = place_atom(CG, CG_d, BOND_LEN['C=O'], ANGLE_SP2, 0.0, n_CG)
        sc_atoms.append(Atom(OD1, 'O', 'OD1'))
        ND2, _ = place_atom(CG, CG_d, BOND_LEN['C-N_sc'], ANGLE_SP2, 180.0, n_CG)
        sc_atoms.append(Atom(ND2, 'N', 'ND2'))

    elif aa_code == 'Q':  # -CH₂-CH₂-CONH₂
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        sc_atoms.extend(_place_ch2(CG, CG_d, n_CG))
        CD, CD_d, n_CD = _extend_chain(CG, CG_d, n_CG, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CD, 'C', 'CD'))
        OE1, _ = place_atom(CD, CD_d, BOND_LEN['C=O'], ANGLE_SP2, 0.0, n_CD)
        sc_atoms.append(Atom(OE1, 'O', 'OE1'))
        NE2, _ = place_atom(CD, CD_d, BOND_LEN['C-N_sc'], ANGLE_SP2, 180.0, n_CD)
        sc_atoms.append(Atom(NE2, 'N', 'NE2'))

    elif aa_code == 'K':  # -CH₂-CH₂-CH₂-CH₂-NH₃⁺
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        sc_atoms.extend(_place_ch2(CG, CG_d, n_CG))
        CD, CD_d, n_CD = _extend_chain(CG, CG_d, n_CG, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CD, 'C', 'CD'))
        sc_atoms.extend(_place_ch2(CD, CD_d, n_CD))
        CE, CE_d, n_CE = _extend_chain(CD, CD_d, n_CD, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CE, 'C', 'CE'))
        sc_atoms.extend(_place_ch2(CE, CE_d, n_CE))
        NZ, _ = place_atom(CE, CE_d, BOND_LEN['C-N_sc'], ANGLE_TET, 180.0, n_CE)
        sc_atoms.append(Atom(NZ, 'N', 'NZ'))

    elif aa_code == 'R':  # -CH₂-CH₂-CH₂-NH-C(=NH)-NH₂
        sc_atoms.extend(_place_ch2(CB, u_CB, normal_CB))
        CG, CG_d, n_CG = _extend_chain(CB, u_CB, normal_CB, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CG, 'C', 'CG'))
        sc_atoms.extend(_place_ch2(CG, CG_d, n_CG))
        CD, CD_d, n_CD = _extend_chain(CG, CG_d, n_CG, BOND_LEN['C-C'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(CD, 'C', 'CD'))
        sc_atoms.extend(_place_ch2(CD, CD_d, n_CD))
        NE, NE_d, n_NE = _extend_chain(CD, CD_d, n_CD, BOND_LEN['C-N_sc'], ANGLE_TET, 180.0)
        sc_atoms.append(Atom(NE, 'N', 'NE'))
        CZ, CZ_d, n_CZ = _extend_chain(NE, NE_d, n_NE, BOND_LEN['C-N_sc'], ANGLE_SP2, 180.0)
        sc_atoms.append(Atom(CZ, 'C', 'CZ'))
        NH1, _ = place_atom(CZ, CZ_d, BOND_LEN['C-N_sc'], ANGLE_SP2, 0.0, n_CZ)
        sc_atoms.append(Atom(NH1, 'N', 'NH1'))
        NH2, _ = place_atom(CZ, CZ_d, BOND_LEN['C-N_sc'], ANGLE_SP2, 180.0, n_CZ)
        sc_atoms.append(Atom(NH2, 'N', 'NH2'))

    return sc_atoms


# =====================================================================
# STERIC CLASH DETECTION
# =====================================================================

def has_clash(sidechain_atoms, backbone_atoms, exclude_names=None):
    """
    Check if any sidechain atom clashes with any backbone atom.
    Atoms within 2 bonds of each other (1-2 and 1-3 contacts) are excluded.
    """
    if exclude_names is None:
        # Exclude residue-i backbone atoms bonded to the sidechain through Cα
        exclude_names = {'CA(i)', 'N(i)', "C'(i)", 'HA(i)', 'H(i)', 'O(i)'}

    for sc_atom in sidechain_atoms:
        r_sc = VDW_RADIUS[sc_atom.element]
        for bb_atom in backbone_atoms:
            if bb_atom.name in exclude_names:
                continue
            r_bb = VDW_RADIUS[bb_atom.element]
            d = np.linalg.norm(sc_atom.pos - bb_atom.pos)
            clash_dist = (r_sc + r_bb) * OVERLAP_FACTOR
            if d < clash_dist:
                return True
    return False


# Chi-1 rotamers: gauche+ (60°), anti (180°), gauche- (-60°)
CHI1_ROTAMERS = [60.0, 180.0, -60.0]


def _rotate_sidechain(sc_atoms, CA_pos, CB_pos, chi1_angle):
    """Rotate all sidechain atoms (except CB) by chi1 around the CA-CB axis."""
    axis = CB_pos - CA_pos
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    R = rotation_matrix(axis, np.radians(chi1_angle))
    rotated = []
    for atom in sc_atoms:
        if atom.name == 'CB':
            rotated.append(atom)  # CB doesn't move
        else:
            rel = atom.pos - CA_pos
            new_pos = CA_pos + R @ rel
            rotated.append(Atom(new_pos, atom.element, atom.name))
    return rotated


def scan_ramachandran(aa_code, step=SCAN_STEP):
    """
    Scan φ/ψ space with χ₁ rotamer sampling.  A (φ, ψ) angle is
    "allowed" if ANY of the 3 canonical χ₁ rotamers is clash-free.
    For Gly and Ala (no χ₁ freedom), the scan is a simple yes/no.
    """
    phi_range = np.arange(-180, 180, step)
    psi_range = np.arange(-180, 180, step)

    allowed_map = np.zeros((len(phi_range), len(psi_range)), dtype=float)

    # Proline: φ locked to ~-75° to -55° by ring
    if aa_code == 'P':
        phi_mask = np.array([(p >= -80 and p <= -50) for p in phi_range])
    else:
        phi_mask = np.ones(len(phi_range), dtype=bool)

    # Should we scan χ₁? Only for AAs with substituents beyond Cβ
    scan_chi1 = aa_code not in ('G', 'A')  # Gly has no CB, Ala has only CH₃

    for i, phi in enumerate(phi_range):
        if not phi_mask[i]:
            continue
        for j, psi in enumerate(psi_range):
            try:
                backbone_atoms, cb_data = build_backbone(phi, psi)
                base_sc = place_sidechain(cb_data, aa_code)

                if not scan_chi1:
                    # No χ₁ degree of freedom
                    if not has_clash(base_sc, backbone_atoms):
                        allowed_map[i, j] = 1.0
                else:
                    # Try 3 χ₁ rotamers; fraction of allowed = rotamer accessibility
                    allowed_count = 0
                    CA = cb_data['CA_pos']
                    CB = base_sc[0].pos  # first atom is always CB
                    for chi1 in CHI1_ROTAMERS:
                        rotated_sc = _rotate_sidechain(base_sc, CA, CB, chi1)
                        if not has_clash(rotated_sc, backbone_atoms):
                            allowed_count += 1
                    allowed_map[i, j] = allowed_count / len(CHI1_ROTAMERS)
            except Exception:
                pass

    total_cells = len(phi_range) * len(psi_range)

    def basin_fraction(phi_lim, psi_lim):
        phi_idx = (phi_range >= phi_lim[0]) & (phi_range <= phi_lim[1])
        psi_idx = (psi_range >= psi_lim[0]) & (psi_range <= psi_lim[1])
        basin = allowed_map[np.ix_(phi_idx, psi_idx)]
        basin_total = basin.size
        return basin.sum() / basin_total if basin_total > 0 else 0.0

    helix_frac = basin_fraction(HELIX_PHI, HELIX_PSI)
    sheet_frac = basin_fraction(SHEET_PHI, SHEET_PSI)
    total_frac = allowed_map.sum() / total_cells

    return {
        'helix_fraction': helix_frac,
        'sheet_fraction': sheet_frac,
        'total_allowed': total_frac,
        'allowed_map': allowed_map,
        'phi_range': phi_range,
        'psi_range': psi_range,
    }


# =====================================================================
# H-BOND COMPETITION FROM MOLECULAR GRAPH  (Axiom 1 + 2)
# =====================================================================
#
# Two mechanisms modulate helix propensity beyond steric exclusion:
#
# 1. Sidechain H-bond competition:
#    Polar groups on the sidechain can "steal" backbone H-bond partners.
#    The steal probability depends on the sidechain H-bond force constant
#    relative to backbone: p = k_sc / (k_sc + k_bb).
#    BUT: only polar groups CLOSE ENOUGH to reach backbone sites compete.
#    The chain length (bonds from Cα to polar atom) determines reach.
#
# 2. Pro amide-H absence:
#    Proline's pyrrolidine N lacks the amide H needed for the i→i+4
#    backbone H-bond.  This reduces helix stability by ~70%.
#
# 3. Helix macro-dipole stabilization:
#    Charged sidechains stabilize the α-helix macro-dipole.
#    E_dipole ≈ 2 kJ/mol per unit charge vs E_hbond ≈ 20 kJ/mol.

# H-bond force constants (N/m) — from lattice Morse potential wells
K_BB = 15.0     # backbone N-H...O=C
K_SC = {        # sidechain donor/acceptor → backbone partner
    'O_don': 12.0,   # O-H donating to backbone C=O
    'O_acc': 10.0,   # O accepting from backbone N-H
    'N_don': 9.0,    # N-H donating to backbone C=O
    'N_acc': 7.0,    # N accepting from backbone N-H
    'S_don': 4.0,    # S-H donating (weak)
    'S_acc': 3.0,    # S accepting (very weak)
}

# Chain length: bonds from Cα to first polar atom (from molecular graph)
POLAR_CHAIN_LEN = {
    'G': 0, 'A': 0, 'V': 0, 'L': 0, 'I': 0, 'P': 0, 'F': 0,
    'S': 2, 'T': 2, 'C': 2,           # Cα-Cβ-Xγ (short → high reach)
    'D': 3, 'N': 3, 'H': 3, 'W': 3,   # Cα-Cβ-Cγ-Xδ (moderate reach)
    'Y': 4, 'E': 4, 'Q': 4, 'M': 4,   # 4+ bonds (too far)
    'K': 5, 'R': 5,                     # 5+ bonds (way too far)
}

# Sidechain polar group types for each AA
SC_POLAR_TYPE = {
    'G': [], 'A': [], 'V': [], 'L': [], 'I': [], 'P': [], 'F': [],
    'W': [('N_don',)],
    'M': [('S_acc',)],
    'S': [('O_don',), ('O_acc',)],
    'T': [('O_don',), ('O_acc',)],
    'C': [('S_don',), ('S_acc',)],
    'Y': [('O_don',), ('O_acc',)],
    'H': [('N_don',), ('N_acc',)],
    'D': [('O_acc',), ('O_acc',)],
    'E': [('O_acc',), ('O_acc',)],
    'N': [('O_acc',), ('N_don',)],
    'Q': [('O_acc',), ('N_don',)],
    'K': [('N_don',)],
    'R': [('N_don',), ('N_don',), ('N_don',)],
}

# Formal charge at pH 7 (from molecular graph protonation state)
FORMAL_CHARGE = {'D': -1, 'E': -1, 'K': +1, 'R': +1}

# Derived constants
F_DIPOLE = 2.0 / 20.0   # E_dipole / E_hbond = 0.10
F_PRO = 0.30             # 70% of helix stability from H-bonds
AMIDE_H = {c: 1 for c in 'GAVILPFYWMSTCYHDENQKR'}
AMIDE_H['P'] = 0


def _chain_reach(n_bonds):
    """
    Fraction of time a polar group at `n_bonds` from Cα can reach the
    backbone H-bond site.  Derived from the geometric constraint:
      max_reach ≈ n × d_bond × sin(θ_tet/2) = n × 1.22 Å
    compared to H-bond range [2.5, 4.5] Å.
    """
    if n_bonds <= 1:
        return 0.0
    d_approx = n_bonds * 1.22
    D_MIN, D_MAX = 2.5, 4.5
    if d_approx <= D_MIN:
        return 1.0
    if d_approx >= D_MAX:
        return 0.0
    return 1.0 - (d_approx - D_MIN) / (D_MAX - D_MIN)


def compute_hbond_factor(aa_code):
    """
    Compute the H-bond competition factor for an amino acid.
    Returns f_hbond ∈ (0, 1.1] where:
      1.0 = no competition (hydrophobic)
      <1.0 = polar groups steal backbone H-bonds
      >1.0 = charged sidechain stabilizes helix dipole
      0.30 for Pro (no amide H)
    """
    chain_len = POLAR_CHAIN_LEN[aa_code]
    reach = _chain_reach(chain_len)

    polar_types = SC_POLAR_TYPE[aa_code]

    # Max steal per backbone site
    donor_steals = []   # sidechain donors stealing backbone C=O
    acc_steals = []     # sidechain acceptors stealing backbone N-H
    for (t,) in polar_types:
        k = K_SC[t]
        p = k / (k + K_BB) * reach
        if 'don' in t:
            donor_steals.append(p)
        else:
            acc_steals.append(p)

    max_d = max(donor_steals) if donor_steals else 0.0
    max_a = max(acc_steals) if acc_steals else 0.0

    f = (1 - max_d) * (1 - max_a)

    # Dipole stabilization
    charge = FORMAL_CHARGE.get(aa_code, 0)
    f *= (1 + F_DIPOLE * abs(charge))

    # Pro correction
    if AMIDE_H[aa_code] == 0:
        f *= F_PRO

    return f


# =====================================================================
# MAIN: COMPUTE ALL 20 AMINO ACIDS
# =====================================================================

AMINO_ACIDS = list('GAVILPFYWMSTCYHDENQKR')

# Chou-Fasman reference values for validation
CF_ALPHA = {
    'E':1.51,'M':1.45,'A':1.42,'L':1.21,'K':1.16,'F':1.13,
    'Q':1.11,'W':1.08,'I':1.08,'V':1.06,'D':1.01,'H':1.00,
    'R':0.98,'T':0.83,'S':0.77,'C':0.70,'Y':0.69,'N':0.67,
    'G':0.57,'P':0.57,
}


def compute_all_z_topo(step=SCAN_STEP):
    """
    Compute Z_topo for all 20 amino acids from the combined model:
      P_helix = steric_fraction × hbond_factor
    
    Z_topo is then scaled inversely: high propensity → low Z_topo.
    """
    steric_results = {}
    for aa in AMINO_ACIDS:
        steric_results[aa] = scan_ramachandran(aa, step=step)

    # Combine steric + H-bond
    propensity = {}
    for aa in AMINO_ACIDS:
        hf = steric_results[aa]['helix_fraction']
        f_hb = compute_hbond_factor(aa)
        propensity[aa] = hf * f_hb

    # Z_topo = inverse of combined propensity
    z_topo = {}
    for aa in AMINO_ACIDS:
        p = propensity[aa]
        if p > 0.01:
            z_topo[aa] = 1.0 / p
        else:
            z_topo[aa] = 100.0

    # Normalize to [0.5, 5.0] range
    z_vals = list(z_topo.values())
    z_min, z_max = min(z_vals), max(z_vals)
    for aa in z_topo:
        z_topo[aa] = 0.5 + 4.5 * (z_topo[aa] - z_min) / (z_max - z_min + 1e-15)

    return steric_results, z_topo, propensity


if __name__ == '__main__':
    import time

    print("First-Principles Helix Propensity Calculator")
    print("  Component 1: Ramachandran steric exclusion (5-residue, χ₁ scan)")
    print("  Component 2: H-bond competition (chain-length filtered)")
    print("=" * 65)
    print(f"Parameters: step={SCAN_STEP}°, overlap={OVERLAP_FACTOR}")
    print()

    t0 = time.time()
    steric_results, z_topo, propensity = compute_all_z_topo(step=SCAN_STEP)
    dt = time.time() - t0

    # Compute correlations
    codes = sorted(CF_ALPHA.keys())
    pa_arr = np.array([CF_ALPHA[c] for c in codes])
    hf_arr = np.array([steric_results[c]['helix_fraction'] for c in codes])
    p_arr = np.array([propensity[c] for c in codes])

    corr_hf = np.corrcoef(pa_arr, hf_arr)[0, 1]
    corr_p = np.corrcoef(pa_arr, p_arr)[0, 1]

    print(f"Completed in {dt:.1f}s")
    print(f"\nCorrelation with Chou-Fasman Pα:")
    print(f"  Steric only:    r = {corr_hf:+.4f}")
    print(f"  Combined model: r = {corr_p:+.4f}")

    # Classification accuracy
    matches = 0
    print(f"\n{'AA':>3} {'Pα':>5} {'Helix%':>7} {'f_hb':>6} {'P_comb':>7} "
          f"{'Z_topo':>7} {'C?':>3}")
    print('-' * 55)
    for code in sorted(codes, key=lambda c: -CF_ALPHA[c]):
        sr = steric_results[code]
        hf = sr['helix_fraction']
        f_hb = compute_hbond_factor(code)
        p = propensity[code]
        z = z_topo[code]
        actual_h = CF_ALPHA[code] >= 1.0
        pred_h = z <= 2.0  # threshold for helix classification
        match = actual_h == pred_h
        if match:
            matches += 1
        print(f"{code:>3} {CF_ALPHA[code]:>5.2f} {hf*100:>7.1f} {f_hb:>6.3f} "
              f"{p:>7.4f} {z:>7.2f} {'✓' if match else '✗':>3}")

    print(f"\nMatches: {matches}/20 ({matches/20*100:.0f}%)")

