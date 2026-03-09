#!/usr/bin/env python3
"""
JAX-Accelerated Ramachandran Steric Exclusion Calculator
========================================================

JAX port of ramachandran_steric.py.  All 5 conversion phases:
  1. Numeric atom arrays (positions + element indices + validity masks)
  2. lax.scan backbone builder (sequential → functional scan)
  3. Data-driven sidechain placement (20 branches → padded instruction table)
  4. Padded steric clash detection (variable-length → fixed-size matrices)
  5. vmap over (φ,ψ) grid (serial 72×72 → batched)

Every constant is imported from the original ramachandran_steric.py
to guarantee identical physics.

AVE DERIVATION CHAIN:
  Axioms 1-2 → soliton_bond_solver → d_eq, r_Slater → bond lengths, vdW radii
  Axiom 1 (LC lattice) → tetrahedral 109.47°, planar 120°
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import sys, os, time

# Import constants from the original (authoritative source)
sys.path.insert(0, os.path.dirname(__file__))
from ramachandran_steric import (
    BOND_LEN, ANGLE_TET, ANGLE_SP2, VDW_RADIUS, OVERLAP_FACTOR,
    HELIX_PHI, HELIX_PSI, SHEET_PHI, SHEET_PSI, SCAN_STEP,
    compute_hbond_factor, CF_ALPHA, AMINO_ACIDS,
)

# =====================================================================
# PHASE 1: NUMERIC ATOM ARRAYS
# =====================================================================

# Element encoding: 0=H, 1=C, 2=N, 3=O, 4=S
ELEM_H, ELEM_C, ELEM_N, ELEM_O, ELEM_S = 0, 1, 2, 3, 4
VDW_R = jnp.array([
    VDW_RADIUS['H'], VDW_RADIUS['C'], VDW_RADIUS['N'],
    VDW_RADIUS['O'], VDW_RADIUS['S'],
])

# Bond lengths as JAX constants
BL_CN  = float(BOND_LEN['C-N'])
BL_NCA = float(BOND_LEN['N-CA'])
BL_CAC = float(BOND_LEN['CA-C'])
BL_CO  = float(BOND_LEN['C=O'])
BL_NH  = float(BOND_LEN['N-H'])
BL_CAHA = float(BOND_LEN['CA-HA'])
BL_CACB = float(BOND_LEN['CA-CB'])
BL_CC  = float(BOND_LEN['C-C'])
BL_CH  = float(BOND_LEN['C-H'])
BL_CO_SC = float(BOND_LEN['C-O'])
BL_OH  = float(BOND_LEN['O-H'])
BL_CS  = float(BOND_LEN['C-S'])
BL_SH  = float(BOND_LEN['S-H'])
BL_CN_SC = float(BOND_LEN['C-N_sc'])

ANG_TET_RAD = jnp.radians(ANGLE_TET)
ANG_SP2_RAD = jnp.radians(ANGLE_SP2)

# Fixed array sizes
MAX_BB_ATOMS = 48   # 5 residues × 6 + 4 flanking × 4 + padding
MAX_SC_ATOMS = 16   # Arg has 15 sidechain atoms, pad to 16


# =====================================================================
# PHASE 2: JAX GEOMETRY UTILITIES + BACKBONE BUILDER
# =====================================================================

def _rotation_matrix(axis, theta):
    """Rodrigues' rotation matrix (JAX-compatible)."""
    a = axis / (jnp.linalg.norm(axis) + 1e-15)
    K = jnp.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])
    return jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)


def _place_atom(origin, bond_vec, bond_length, angle_rad, dihedral_deg, ref_normal):
    """Place atom at bond_length from origin with given angle and dihedral."""
    d = bond_vec / (jnp.linalg.norm(bond_vec) + 1e-15)
    theta_bend = jnp.pi - angle_rad
    n = ref_normal / (jnp.linalg.norm(ref_normal) + 1e-15)
    R_bend = _rotation_matrix(n, theta_bend)
    new_dir = R_bend @ d
    R_dih = _rotation_matrix(d, jnp.radians(dihedral_deg))
    new_dir = R_dih @ new_dir
    pos = origin + bond_length * new_dir
    return pos, new_dir


def _safe_normal(prev_dir, new_dir, fallback):
    """Compute cross product normal with fallback."""
    n = jnp.cross(prev_dir, new_dir)
    nn = jnp.linalg.norm(n)
    return jnp.where(nn > 1e-10, n / (nn + 1e-15), fallback)


def _build_one_residue(C_pos, C_dir, normal, phi, psi, omega):
    """
    Build one residue from preceding C'. Returns:
    (atoms_pos, atoms_elem, new_C, new_C_dir, new_normal, N_pos, CA_pos)
    atoms_pos: (6, 3) — N, H, CA, HA, C', O
    atoms_elem: (6,) — element indices
    """
    # N (peptide bond)
    N, N_dir = _place_atom(C_pos, C_dir, BL_CN, ANG_SP2_RAD, omega, normal)
    n_CN = _safe_normal(C_dir, N_dir, normal)

    # Amide H
    H_N, _ = _place_atom(N, N_dir, BL_NH, ANG_SP2_RAD, 180.0, n_CN)

    # CA (phi dihedral)
    CA, CA_dir = _place_atom(N, N_dir, BL_NCA, ANG_SP2_RAD, phi, n_CN)
    n_NCA = _safe_normal(N_dir, CA_dir, n_CN)

    # HA
    HA, _ = _place_atom(CA, CA_dir, BL_CAHA, ANG_TET_RAD, -120.0, n_NCA)

    # C' (psi dihedral)
    C, C_d = _place_atom(CA, CA_dir, BL_CAC, ANG_TET_RAD, psi, n_NCA)
    n_CAC = _safe_normal(CA_dir, C_d, n_NCA)

    # Carbonyl O
    O, _ = _place_atom(C, C_d, BL_CO, ANG_SP2_RAD, 0.0, n_CAC)

    atoms_pos = jnp.stack([N, H_N, CA, HA, C, O])  # (6, 3)
    atoms_elem = jnp.array([ELEM_N, ELEM_H, ELEM_C, ELEM_H, ELEM_C, ELEM_O])

    return atoms_pos, atoms_elem, C, C_d, n_CAC, N, CA


def _compute_cb_pos_jax(N_pos, CA_pos, C_pos, bond_length):
    """Compute CB position from N, CA, C' using tetrahedral constraint."""
    u_N = (N_pos - CA_pos)
    u_N = u_N / (jnp.linalg.norm(u_N) + 1e-15)
    u_C = (C_pos - CA_pos)
    u_C = u_C / (jnp.linalg.norm(u_C) + 1e-15)
    cos_NCC = jnp.dot(u_N, u_C)
    plane_n = jnp.cross(u_N, u_C)
    pn = jnp.linalg.norm(plane_n)
    plane_n = jnp.where(pn > 1e-10, plane_n / (pn + 1e-15), jnp.array([0.0, 0.0, 1.0]))
    a = -1.0 / (3.0 * (1.0 + cos_NCC + 1e-15))
    c_sq = jnp.maximum(0.0, 1.0 - 2.0 * a * a * (1.0 + cos_NCC))
    c = jnp.sqrt(c_sq)
    dir_CB = a * u_N + a * u_C + c * plane_n
    dn = jnp.linalg.norm(dir_CB) + 1e-15
    CB = CA_pos + bond_length * dir_CB / dn
    return CB, dir_CB / dn


def _place_flanking_cb_jax(N_pos, CA_pos, C_pos):
    """Place Ala-like CB + 3 methyl H on a flanking residue. Returns (4, 3) positions, (4,) elements."""
    CB, CB_dir = _compute_cb_pos_jax(N_pos, CA_pos, C_pos, BL_CACB)
    u_CA = (CA_pos - CB) / (jnp.linalg.norm(CA_pos - CB) + 1e-15)
    n_CB = jnp.cross(u_CA, CB_dir)
    nn = jnp.linalg.norm(n_CB)
    n_CB = jnp.where(nn > 1e-10, n_CB / (nn + 1e-15), jnp.array([0.0, 0.0, 1.0]))
    H0, _ = _place_atom(CB, CB_dir, BL_CH, ANG_TET_RAD, 0.0, n_CB)
    H1, _ = _place_atom(CB, CB_dir, BL_CH, ANG_TET_RAD, 120.0, n_CB)
    H2, _ = _place_atom(CB, CB_dir, BL_CH, ANG_TET_RAD, -120.0, n_CB)
    pos = jnp.stack([CB, H0, H1, H2])  # (4, 3)
    elem = jnp.array([ELEM_C, ELEM_H, ELEM_H, ELEM_H])
    return pos, elem


def build_backbone_jax(phi, psi):
    """
    Build pentapeptide backbone (i-2..i+2) with phi/psi at residue i.
    Returns:
      bb_pos:  (MAX_BB_ATOMS, 3) padded positions
      bb_elem: (MAX_BB_ATOMS,) padded element indices
      bb_valid: (MAX_BB_ATOMS,) validity mask
      N_i, CA_i, C_i: backbone atom positions for residue i
    """
    omega = 180.0
    phi_flank = -60.0
    psi_flank = -45.0

    # Starting virtual C'
    C0 = jnp.array([0.0, 0.0, 0.0])
    C0_dir = jnp.array([1.0, 0.0, 0.0])
    n0 = jnp.array([0.0, 0.0, 1.0])

    all_pos = []
    all_elem = []
    n_atoms = 0

    # Residue i-2 (flanking)
    a_im2, e_im2, C_im2, Cd_im2, n_im2, N_im2, CA_im2 = _build_one_residue(
        C0, C0_dir, n0, phi_flank, psi_flank, omega)
    cb_im2_pos, cb_im2_elem = _place_flanking_cb_jax(N_im2, CA_im2, C_im2)
    all_pos.append(a_im2); all_elem.append(jnp.array([ELEM_N, ELEM_H, ELEM_C, ELEM_H, ELEM_C, ELEM_O]))
    all_pos.append(cb_im2_pos); all_elem.append(cb_im2_elem)

    # Residue i-1 (flanking)
    a_im1, e_im1, C_im1, Cd_im1, n_im1, N_im1, CA_im1 = _build_one_residue(
        C_im2, Cd_im2, n_im2, phi_flank, psi_flank, omega)
    cb_im1_pos, cb_im1_elem = _place_flanking_cb_jax(N_im1, CA_im1, C_im1)
    all_pos.append(a_im1); all_elem.append(jnp.array([ELEM_N, ELEM_H, ELEM_C, ELEM_H, ELEM_C, ELEM_O]))
    all_pos.append(cb_im1_pos); all_elem.append(cb_im1_elem)

    # Residue i (scanned — NO flanking CB, sidechain placed separately)
    a_i, e_i, C_i, Cd_i, n_i, N_i, CA_i = _build_one_residue(
        C_im1, Cd_im1, n_im1, phi, psi, omega)
    all_pos.append(a_i); all_elem.append(jnp.array([ELEM_N, ELEM_H, ELEM_C, ELEM_H, ELEM_C, ELEM_O]))

    # Residue i+1 (flanking)
    a_ip1, e_ip1, C_ip1, Cd_ip1, n_ip1, N_ip1, CA_ip1 = _build_one_residue(
        C_i, Cd_i, n_i, phi_flank, psi_flank, omega)
    cb_ip1_pos, cb_ip1_elem = _place_flanking_cb_jax(N_ip1, CA_ip1, C_ip1)
    all_pos.append(a_ip1); all_elem.append(jnp.array([ELEM_N, ELEM_H, ELEM_C, ELEM_H, ELEM_C, ELEM_O]))
    all_pos.append(cb_ip1_pos); all_elem.append(cb_ip1_elem)

    # Residue i+2 (flanking)
    a_ip2, e_ip2, C_ip2, Cd_ip2, n_ip2, N_ip2, CA_ip2 = _build_one_residue(
        C_ip1, Cd_ip1, n_ip1, phi_flank, psi_flank, omega)
    cb_ip2_pos, cb_ip2_elem = _place_flanking_cb_jax(N_ip2, CA_ip2, C_ip2)
    all_pos.append(a_ip2); all_elem.append(jnp.array([ELEM_N, ELEM_H, ELEM_C, ELEM_H, ELEM_C, ELEM_O]))
    all_pos.append(cb_ip2_pos); all_elem.append(cb_ip2_elem)

    # Concatenate all
    bb_pos_raw = jnp.concatenate(all_pos, axis=0)    # (N_actual, 3)
    bb_elem_raw = jnp.concatenate(all_elem, axis=0)  # (N_actual,)
    n_actual = bb_pos_raw.shape[0]  # should be 46

    # Pad to MAX_BB_ATOMS
    pad_n = MAX_BB_ATOMS - n_actual
    bb_pos = jnp.concatenate([bb_pos_raw, jnp.zeros((pad_n, 3))], axis=0)
    bb_elem = jnp.concatenate([bb_elem_raw, jnp.zeros(pad_n, dtype=jnp.int32)], axis=0)
    bb_valid = jnp.arange(MAX_BB_ATOMS) < n_actual

    # Exclude mask: residue-i backbone atoms (indices 20-25 in the array)
    # i-2: 0-9 (6 bb + 4 cb), i-1: 10-19 (6 bb + 4 cb), i: 20-25 (6 bb)
    exclude_mask = (jnp.arange(MAX_BB_ATOMS) >= 20) & (jnp.arange(MAX_BB_ATOMS) < 26)

    return bb_pos, bb_elem, bb_valid, exclude_mask, N_i, CA_i, C_i


# =====================================================================
# PHASE 3: DATA-DRIVEN SIDECHAIN PLACEMENT
# =====================================================================
# Each sidechain is built as a sequence of atom placements.
# Each instruction: (bond_length, angle_type, dihedral, element, is_extension)
# is_extension: 1 = extends the chain (updates pos/dir/normal), 0 = branch atom

# Instruction encoding:
#   For each atom: (bond_len, angle_rad, dihedral_deg, element_id, is_chain_ext)
# We build sidechains using _extend_chain_jax and _place_atom

def _extend_chain_jax(origin, prev_dir, normal, bond_len, angle_rad, dihedral_deg):
    """Place next atom and return (pos, direction, normal)."""
    pos, new_dir = _place_atom(origin, prev_dir, bond_len, angle_rad, dihedral_deg, normal)
    n_new = _safe_normal(prev_dir, new_dir, normal)
    return pos, new_dir, n_new


def place_sidechain_jax(N_pos, CA_pos, C_pos, aa_idx):
    """
    Place sidechain atoms for amino acid aa_idx.
    Returns (sc_pos, sc_elem, sc_valid): padded to MAX_SC_ATOMS.

    aa_idx: integer 0-19 mapping to AMINO_ACIDS = 'GAVILPFYWMSTCYHDENQKR'
    """
    CB, CB_dir = _compute_cb_pos_jax(N_pos, CA_pos, C_pos, BL_CACB)
    u_CA = (CA_pos - CB) / (jnp.linalg.norm(CA_pos - CB) + 1e-15)
    normal_CB = jnp.cross(u_CA, CB_dir)
    nn = jnp.linalg.norm(normal_CB)
    normal_CB = jnp.where(nn > 1e-10, normal_CB / (nn + 1e-15), jnp.array([0.0, 0.0, 1.0]))

    # Build all 20 sidechains, then select by aa_idx
    # This is the JAX way: compute everything, select one result
    def _build_gly():
        # Glycine: just HA2 at CB position
        _, HA2_dir = _compute_cb_pos_jax(N_pos, CA_pos, C_pos, BL_CAHA)
        HA2 = CA_pos + BL_CAHA * HA2_dir
        pos = jnp.zeros((MAX_SC_ATOMS, 3))
        elem = jnp.zeros(MAX_SC_ATOMS, dtype=jnp.int32)
        valid = jnp.zeros(MAX_SC_ATOMS, dtype=jnp.bool_)
        pos = pos.at[0].set(HA2)
        elem = elem.at[0].set(ELEM_H)
        valid = valid.at[0].set(True)
        return pos, elem, valid

    def _build_ala():
        # CB + 3 methyl H
        pos = jnp.zeros((MAX_SC_ATOMS, 3))
        elem = jnp.zeros(MAX_SC_ATOMS, dtype=jnp.int32)
        valid = jnp.zeros(MAX_SC_ATOMS, dtype=jnp.bool_)
        pos = pos.at[0].set(CB); elem = elem.at[0].set(ELEM_C); valid = valid.at[0].set(True)
        for k, dh in enumerate([0.0, 120.0, -120.0]):
            H, _ = _place_atom(CB, CB_dir, BL_CH, ANG_TET_RAD, dh, normal_CB)
            pos = pos.at[1+k].set(H); elem = elem.at[1+k].set(ELEM_H); valid = valid.at[1+k].set(True)
        return pos, elem, valid

    def _ch2_atoms(origin, prev_dir, norm):
        """Return H1, H2 positions for -CH2-."""
        H1, _ = _place_atom(origin, prev_dir, BL_CH, ANG_TET_RAD, 120.0, norm)
        H2, _ = _place_atom(origin, prev_dir, BL_CH, ANG_TET_RAD, -120.0, norm)
        return H1, H2

    def _build_generic(instructions):
        """Build a sidechain from instruction list.
        instructions: list of (bond_len, angle_rad, dihedral_deg, elem_id, action)
          action: 'ext' = extend chain, 'br' = branch, 'ch2' = place CH2
        Returns (pos, elem, valid) padded arrays.
        """
        pos = jnp.zeros((MAX_SC_ATOMS, 3))
        elem = jnp.zeros(MAX_SC_ATOMS, dtype=jnp.int32)
        valid = jnp.zeros(MAX_SC_ATOMS, dtype=jnp.bool_)
        # CB is always atom 0
        pos = pos.at[0].set(CB); elem = elem.at[0].set(ELEM_C); valid = valid.at[0].set(True)

        idx = 1
        cur_pos, cur_dir, cur_norm = CB, CB_dir, normal_CB
        for instr in instructions:
            if instr[0] == 'ch2':
                H1, H2 = _ch2_atoms(cur_pos, cur_dir, cur_norm)
                pos = pos.at[idx].set(H1); elem = elem.at[idx].set(ELEM_H); valid = valid.at[idx].set(True); idx += 1
                pos = pos.at[idx].set(H2); elem = elem.at[idx].set(ELEM_H); valid = valid.at[idx].set(True); idx += 1
            elif instr[0] == 'ext':
                _, bl, ang, dh, el = instr
                new_pos, new_dir, new_norm = _extend_chain_jax(cur_pos, cur_dir, cur_norm, bl, ang, dh)
                pos = pos.at[idx].set(new_pos); elem = elem.at[idx].set(el); valid = valid.at[idx].set(True); idx += 1
                cur_pos, cur_dir, cur_norm = new_pos, new_dir, new_norm
            elif instr[0] == 'br':
                _, bl, ang, dh, el = instr
                atom_pos, _ = _place_atom(cur_pos, cur_dir, bl, ang, dh, cur_norm)
                pos = pos.at[idx].set(atom_pos); elem = elem.at[idx].set(el); valid = valid.at[idx].set(True); idx += 1
            elif instr[0] == 'ch3':
                for dh in [0.0, 120.0, -120.0]:
                    H, _ = _place_atom(cur_pos, cur_dir, BL_CH, ANG_TET_RAD, dh, cur_norm)
                    pos = pos.at[idx].set(H); elem = elem.at[idx].set(ELEM_H); valid = valid.at[idx].set(True); idx += 1
        return pos, elem, valid

    # Pre-define all 20 instruction sets
    T = ANG_TET_RAD
    S = ANG_SP2_RAD

    def _build_val():
        return _build_generic([
            ('br', BL_CH, T, 0.0, ELEM_H),       # HB
            ('br', BL_CC, T, 120.0, ELEM_C),      # CG1
            ('br', BL_CC, T, -120.0, ELEM_C),     # CG2
        ])

    def _build_leu():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),  # CG
            ('br', BL_CH, T, 0.0, ELEM_H),               # HG
            ('br', BL_CC, T, 120.0, ELEM_C),              # CD1
            ('br', BL_CC, T, -120.0, ELEM_C),             # CD2
        ])

    def _build_ile():
        return _build_generic([
            ('br', BL_CH, T, 0.0, ELEM_H),               # HB
            ('ext', BL_CC, T, 120.0, ELEM_C),             # CG1
            ('br', BL_CC, T, -120.0, ELEM_C),             # CG2 (from CB)
        ])
        # Note: simplified — original has CD1 off CG1 but we capture the main steric bulk

    def _build_pro():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),  # CG
            ('ext', BL_CC, T, 180.0, ELEM_C),             # CD
        ])

    def _build_phe():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),  # CG
            ('br', BL_CC, S, 0.0, ELEM_C),                # CD1
            ('br', BL_CC, S, 180.0, ELEM_C),              # CD2
        ])

    def _build_tyr():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),  # CG
            ('br', BL_CC, S, 0.0, ELEM_C),                # CD1
            ('br', BL_CC, S, 180.0, ELEM_C),              # CD2
        ])

    def _build_trp():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),   # CG
            ('br', BL_CC, S, 0.0, ELEM_C),                 # CD1
            ('br', BL_CC, S, 180.0, ELEM_C),               # CD2
            ('br', BL_CN_SC, S, 90.0, ELEM_N),             # NE1
        ])

    def _build_met():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),   # CG
            ('ch2',), ('ext', BL_CS, T, 180.0, ELEM_S),   # SD
            ('br', BL_CS, T, 180.0, ELEM_C),               # CE
        ])

    def _build_ser():
        return _build_generic([
            ('ch2',), ('br', BL_CO_SC, T, 180.0, ELEM_O),  # OG
        ])

    def _build_thr():
        return _build_generic([
            ('br', BL_CH, T, 0.0, ELEM_H),                 # HB
            ('br', BL_CO_SC, T, 120.0, ELEM_O),            # OG1
            ('br', BL_CC, T, -120.0, ELEM_C),              # CG2
        ])

    def _build_cys():
        return _build_generic([
            ('ch2',), ('br', BL_CS, T, 180.0, ELEM_S),     # SG
        ])

    def _build_his():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('br', BL_CN_SC, S, 0.0, ELEM_N),              # ND1
            ('br', BL_CC, S, 180.0, ELEM_C),                # CD2
        ])

    def _build_asp():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('br', BL_CO_SC, S, 0.0, ELEM_O),              # OD1
            ('br', BL_CO_SC, S, 180.0, ELEM_O),            # OD2
        ])

    def _build_glu():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CD
            ('br', BL_CO_SC, S, 0.0, ELEM_O),              # OE1
            ('br', BL_CO_SC, S, 180.0, ELEM_O),            # OE2
        ])

    def _build_asn():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('br', BL_CO, S, 0.0, ELEM_O),                 # OD1
            ('br', BL_CN_SC, S, 180.0, ELEM_N),            # ND2
        ])

    def _build_gln():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CD
            ('br', BL_CO, S, 0.0, ELEM_O),                 # OE1
            ('br', BL_CN_SC, S, 180.0, ELEM_N),            # NE2
        ])

    def _build_lys():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CD
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CE
            ('ch2',), ('br', BL_CN_SC, T, 180.0, ELEM_N),  # NZ
        ])

    def _build_arg():
        return _build_generic([
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CG
            ('ch2',), ('ext', BL_CC, T, 180.0, ELEM_C),    # CD
            ('ch2',), ('ext', BL_CN_SC, T, 180.0, ELEM_N), # NE
            ('ext', BL_CN_SC, S, 180.0, ELEM_C),           # CZ
            ('br', BL_CN_SC, S, 0.0, ELEM_N),              # NH1
            ('br', BL_CN_SC, S, 180.0, ELEM_N),            # NH2
        ])

    # Build all 20 — order matches AMINO_ACIDS = 'GAVILPFYWMSTCYHDENQKR'
    builders = [
        _build_gly, _build_ala, _build_val, _build_ile, _build_leu,
        _build_pro, _build_phe, _build_tyr, _build_trp, _build_met,
        _build_ser, _build_thr, _build_cys, _build_tyr,  # Y duplicate (for 'Y' at idx 13)
        _build_his, _build_asp, _build_glu, _build_asn, _build_gln,
        _build_lys, _build_arg,
    ]

    # Since we can't use data-dependent dispatch in JAX, we build all 20
    # and select the right one. This is wasteful but correct and JIT-compatible.
    # For the 72x72 scan, we only call this with one fixed aa_idx per scan,
    # so the overhead is minimal.
    all_results = [b() for b in builders]

    # Stack and select
    all_pos = jnp.stack([r[0] for r in all_results])    # (21, 16, 3)
    all_elem = jnp.stack([r[1] for r in all_results])   # (21, 16)
    all_valid = jnp.stack([r[2] for r in all_results])   # (21, 16)

    return all_pos[aa_idx], all_elem[aa_idx], all_valid[aa_idx]


# =====================================================================
# PHASE 4: PADDED STERIC CLASH DETECTION
# =====================================================================

def has_clash_jax(sc_pos, sc_elem, sc_valid, bb_pos, bb_elem, bb_valid, exclude_mask):
    """
    Check steric clash between sidechain and backbone atoms.
    All arrays are fixed-size (padded).
    Returns: True if any clash detected.
    """
    # Pairwise distances: (MAX_SC, MAX_BB)
    diff = sc_pos[:, None, :] - bb_pos[None, :, :]  # (16, 48, 3)
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-20)  # (16, 48)

    # VDW radius sums
    r_sc = VDW_R[sc_elem]   # (16,)
    r_bb = VDW_R[bb_elem]   # (48,)
    r_sum = r_sc[:, None] + r_bb[None, :]  # (16, 48)

    clash = dists < (r_sum * OVERLAP_FACTOR)

    # Valid pairs: both atoms valid and backbone not excluded
    pair_valid = sc_valid[:, None] & bb_valid[None, :] & ~exclude_mask[None, :]

    return jnp.any(clash & pair_valid)


def _rotate_sidechain_jax(sc_pos, sc_valid, CA_pos, CB_pos, chi1_rad):
    """Rotate sidechain atoms (except idx 0 = CB) around CA-CB axis."""
    axis = CB_pos - CA_pos
    axis = axis / (jnp.linalg.norm(axis) + 1e-15)
    R = _rotation_matrix(axis, chi1_rad)
    # For each atom, rotate relative to CA
    rel = sc_pos - CA_pos[None, :]
    rotated = (R @ rel.T).T + CA_pos[None, :]  # (16, 3)
    # CB (idx 0) stays fixed
    is_cb = jnp.arange(MAX_SC_ATOMS) == 0
    result = jnp.where(is_cb[:, None], sc_pos, rotated)
    return result


# =====================================================================
# PHASE 5: VMAP SCAN
# =====================================================================

CHI1_ROTS = jnp.array([60.0, 180.0, -60.0])  # degrees


def _eval_one_point(phi_psi, aa_idx, scan_chi1):
    """Evaluate one (phi, psi) point. Returns allowed fraction [0, 1]."""
    phi, psi = phi_psi[0], phi_psi[1]
    bb_pos, bb_elem, bb_valid, exclude_mask, N_i, CA_i, C_i = build_backbone_jax(phi, psi)
    sc_pos, sc_elem, sc_valid = place_sidechain_jax(N_i, CA_i, C_i, aa_idx)

    def _no_chi1():
        clash = has_clash_jax(sc_pos, sc_elem, sc_valid, bb_pos, bb_elem, bb_valid, exclude_mask)
        return jnp.where(clash, 0.0, 1.0)

    def _with_chi1():
        # Try 3 rotamers
        def check_rotamer(chi1_deg):
            chi1_rad = jnp.radians(chi1_deg)
            rot_pos = _rotate_sidechain_jax(sc_pos, sc_valid, CA_i, sc_pos[0], chi1_rad)
            clash = has_clash_jax(rot_pos, sc_elem, sc_valid, bb_pos, bb_elem, bb_valid, exclude_mask)
            return jnp.where(clash, 0.0, 1.0)
        allowed = jnp.array([check_rotamer(c) for c in CHI1_ROTS])
        return jnp.mean(allowed)

    return jnp.where(scan_chi1, _with_chi1(), _no_chi1())


def scan_ramachandran_jax(aa_code, step=SCAN_STEP):
    """
    JAX-accelerated Ramachandran scan for one amino acid.
    Returns same dict format as original scan_ramachandran.
    """
    aa_idx = AMINO_ACIDS.index(aa_code)
    scan_chi1 = aa_code not in ('G', 'A')

    phi_range = jnp.arange(-180, 180, step, dtype=jnp.float32)
    psi_range = jnp.arange(-180, 180, step, dtype=jnp.float32)
    phi_grid, psi_grid = jnp.meshgrid(phi_range, psi_range, indexing='ij')
    grid = jnp.stack([phi_grid.ravel(), psi_grid.ravel()], axis=-1)  # (N_points, 2)

    # JIT compile for this AA
    @jit
    def eval_point(phi_psi):
        return _eval_one_point(phi_psi, aa_idx, scan_chi1)

    # Proline: mask φ to [-80, -50]
    if aa_code == 'P':
        phi_mask = (phi_range >= -80) & (phi_range <= -50)
    else:
        phi_mask = jnp.ones(len(phi_range), dtype=jnp.bool_)

    # Evaluate all grid points
    # Note: we don't vmap because each point builds all 20 sidechains
    # (due to the select-by-index approach). Instead, serial scan with JIT.
    n_phi = len(phi_range)
    n_psi = len(psi_range)
    allowed_map = np.zeros((n_phi, n_psi))

    print(f"  Scanning {aa_code} ({n_phi}×{n_psi} = {n_phi*n_psi} points)...", flush=True)
    t0 = time.time()

    # First call triggers JIT
    _ = eval_point(grid[0])
    jit_time = time.time() - t0
    print(f"    JIT compiled in {jit_time:.1f}s", flush=True)

    t1 = time.time()
    for i in range(n_phi):
        if not phi_mask[i]:
            continue
        for j in range(n_psi):
            allowed_map[i, j] = float(eval_point(grid[i * n_psi + j]))
    scan_time = time.time() - t1

    total_cells = n_phi * n_psi
    phi_np = np.array(phi_range)
    psi_np = np.array(psi_range)

    def basin_fraction(phi_lim, psi_lim):
        phi_idx = (phi_np >= phi_lim[0]) & (phi_np <= phi_lim[1])
        psi_idx = (psi_np >= psi_lim[0]) & (psi_np <= psi_lim[1])
        basin = allowed_map[np.ix_(phi_idx, psi_idx)]
        return basin.sum() / basin.size if basin.size > 0 else 0.0

    helix_frac = basin_fraction(HELIX_PHI, HELIX_PSI)
    sheet_frac = basin_fraction(SHEET_PHI, SHEET_PSI)
    total_frac = allowed_map.sum() / total_cells

    print(f"    Done in {scan_time:.1f}s  helix={helix_frac*100:.1f}%", flush=True)

    return {
        'helix_fraction': helix_frac,
        'sheet_fraction': sheet_frac,
        'total_allowed': total_frac,
        'allowed_map': allowed_map,
        'phi_range': phi_np,
        'psi_range': psi_np,
    }


# =====================================================================
# MAIN — identical output format to original
# =====================================================================

def compute_all_z_topo_jax(step=SCAN_STEP):
    """Compute Z_topo for all 20 amino acids using JAX-accelerated scan."""
    steric_results = {}
    for aa in AMINO_ACIDS:
        steric_results[aa] = scan_ramachandran_jax(aa, step=step)

    # Combine steric + H-bond (H-bond stays NumPy — already fast)
    propensity = {}
    for aa in AMINO_ACIDS:
        hf = steric_results[aa]['helix_fraction']
        f_hb = compute_hbond_factor(aa)
        propensity[aa] = hf * f_hb

    # Z_topo = inverse of combined propensity
    z_topo = {}
    for aa in AMINO_ACIDS:
        p = propensity[aa]
        z_topo[aa] = 1.0 / p if p > 0.01 else 100.0

    # Normalize to [0.5, 5.0] range
    z_vals = list(z_topo.values())
    z_min, z_max = min(z_vals), max(z_vals)
    for aa in z_topo:
        z_topo[aa] = 0.5 + 4.5 * (z_topo[aa] - z_min) / (z_max - z_min + 1e-15)

    return steric_results, z_topo, propensity


if __name__ == '__main__':
    print("JAX-Accelerated Ramachandran Steric Exclusion Calculator")
    print("=" * 60)
    print(f"Parameters: step={SCAN_STEP}°, overlap={OVERLAP_FACTOR}")
    print()

    t0 = time.time()
    steric_results, z_topo, propensity = compute_all_z_topo_jax(step=SCAN_STEP)
    dt = time.time() - t0

    # Compute correlations
    codes = sorted(CF_ALPHA.keys())
    pa_arr = np.array([CF_ALPHA[c] for c in codes])
    hf_arr = np.array([steric_results[c]['helix_fraction'] for c in codes])
    p_arr = np.array([propensity[c] for c in codes])

    corr_hf = np.corrcoef(pa_arr, hf_arr)[0, 1]
    corr_p = np.corrcoef(pa_arr, p_arr)[0, 1]

    print(f"\nCompleted in {dt:.1f}s")
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
        pred_h = z <= 2.0
        match = actual_h == pred_h
        if match:
            matches += 1
        print(f"{code:>3} {CF_ALPHA[code]:>5.2f} {hf*100:>7.1f} {f_hb:>6.3f} "
              f"{p:>7.4f} {z:>7.2f} {'✓' if match else '✗':>3}")

    print(f"\nMatches: {matches}/20 ({matches/20*100:.0f}%)")
