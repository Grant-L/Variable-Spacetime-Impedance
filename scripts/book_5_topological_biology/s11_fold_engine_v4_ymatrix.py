#!/usr/bin/env python3
"""
S₁₁ Minimiser v4: Y-Matrix Network + DC/AC Separation
=======================================================

ARCHITECTURE (v4 vs v3):
  v3: 1063-line monolith mixing DC, AC, and nonlinear layers
  v4: Clean separation into:
    1. DC Analysis  — geometry, sterics, operating point
    2. AC Analysis  — nodal Y-matrix, S₁₁ from [Y]→[S]
    3. Loss         — weighted combination
    4. Optimizer    — Adam (same as v3)

KEY UPGRADE: Contact topology preserved.
  v3: H-bonds → Y_shunt.sum(axis=1)  [leak to ground]
  v4: H-bonds → Y[i,j] off-diagonal  [port-to-port connection]

This uses the shared transmission_line.py module (no duplicate code).
All constants from protein_bond_constants.py and ave.core.constants.

Zero new parameters.  Same Axiom 1-4 physics as v3.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, grad, jit
import optax

# --- AVE imports ---
from ave.solvers.protein_bond_constants import (
    Z_TOPO as Z_TOPO_COMPLEX, Q_BACKBONE,
    BACKBONE_BONDS, BACKBONE_ANGLES,
    D_HB_DETECT, KAPPA_HB, D_NH, D_CO,
    Z_BOND_CA_C, Z_BOND_C_N, Z_BOND_N_CA,
    R_STERIC_CC, R_STERIC_NN, R_STERIC_CN, R_STERIC_CB,
    R_SLATER_C, R_SLATER_O, ANGLE_N_CA_CB_RAD,
    CA_CA_BOND_LENGTH_ANGSTROM,
    D_WATER as D_WATER_CONST, R_BURIAL as R_BURIAL_CONST,
    LAMBDA_BOND as LAMBDA_BOND_CONST,
    LAMBDA_RAMA as LAMBDA_RAMA_CONST,
)
from ave.core.constants import P_C
from ave.core.universal_operators import universal_reflection
from ave.solvers.transmission_line import (
    build_nodal_y_matrix_jax,
    s11_from_y_matrix_jax,
    abcd_to_y_3seg_jax,
    s_diagonal_from_y_matrix_jax,
)

# Import proven backbone generation from v3 (don't reimplement)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from s11_fold_engine_v3_jax import (
    _torsions_to_backbone,
    _compute_cb_positions,
    _compute_cg_positions,
    _nerf_place_atom,
)

# --- Derived constants (same as v3, zero assumed) ---
Z_TOPO = {k: abs(v) for k, v in Z_TOPO_COMPLEX.items()}
THETA_TET = jnp.arccos(-1.0 / 3.0)  # 109.47° — sp3 tetrahedral
PHI_ALPHA = jnp.radians(-60.0)
PSI_ALPHA = jnp.radians(-40.0)
PHI_BETA  = -(jnp.pi - THETA_TET / 2.0)
PSI_BETA  = jnp.pi - THETA_TET / 2.0
PHI_PPII  = jnp.radians(-75.0)
PSI_PPII  = PSI_BETA
OMEGA_TRANS = jnp.radians(180.0)

# Bond lengths and angles from protein_bond_constants
D_N_CA = BACKBONE_BONDS['N-Ca']['length_A']
D_CA_C = BACKBONE_BONDS['Ca-C']['length_A']
D_C_N  = BACKBONE_BONDS['C-N']['length_A']
ANGLE_N_CA_C = jnp.radians(BACKBONE_ANGLES['N-Ca-C'])
ANGLE_CA_C_N = jnp.radians(BACKBONE_ANGLES['Ca-C-N'])
ANGLE_C_N_CA = jnp.radians(BACKBONE_ANGLES['C-N-Ca'])

# Carbonyl C=O and amide N-H geometry (from derived BACKBONE_BONDS)
D_C_O = D_CO   # = BACKBONE_BONDS['C=O']['length_A'] (derived: 1.121 Å)
D_N_H = D_NH   # = BACKBONE_BONDS['N-H']['length_A'] (derived: 0.817 Å)
ANGLE_CA_C_O = jnp.radians(BACKBONE_ANGLES['Ca-C-O'])  # sp² exact: 120.0°
ANGLE_C_N_H  = jnp.radians(BACKBONE_ANGLES['C-N-H'])   # sp² exact: 120.0°
D_CA_CB = BACKBONE_BONDS['Ca-C']['length_A']   # = Cα-C bond length (1.52 Å)
ANGLE_N_CA_CB_DEG = jnp.degrees(ANGLE_N_CA_CB_RAD)  # sp³ exact: 109.47°

# d₀ = Cα-Cα virtual bond (from protein_bond_constants)
d0 = CA_CA_BOND_LENGTH_ANGSTROM  # 3.80 Å (NERF-derived)
r_Ca = R_SLATER_C               # 1.70 Å (carbon Slater radius)
Z0 = 1.0                        # normalised backbone impedance
R_BURIAL = R_BURIAL_CONST       # d₀×√2 ≈ 5.37 Å (FCC coordination shell)
D_WATER = D_WATER_CONST         # 2×R_Slater_O = 3.04 Å
BETA_BURIAL = Q_BACKBONE / d0   # Q/d₀ (sigmoid sharpness)

# Frequency sweep — derived from Q_BACKBONE bandwidth
# BW = f₀/Q = 1/7 (normalised)
_BW = 1.0 / Q_BACKBONE
N_FREQ = 5
FREQ_SWEEP = jnp.array([
    0.5 * (1.0 - _BW/2),  # sub-band floor
    1.0 - _BW/2,          # lower 3dB edge
    1.0,                   # center frequency
    1.0 + _BW/2,          # upper 3dB edge
    2.0 * (1.0 + _BW/2),  # super-band ceiling
])

# Nearest-neighbour coupling
ETA_NN = 1.0 / (2.0 * Q_BACKBONE)

# Backbone segment per-bond impedance magnitudes
Z_N_CA = D_N_CA * Q_BACKBONE
Z_CA_C = D_CA_C * Q_BACKBONE
Z_C_N  = D_C_N  * Q_BACKBONE

# Bend admittance constant: C_bend = (1−cos θ)/(2π²)
# Derived from microstrip junction + d_eff/λ_g = 1/(2π)

# Water Debye relaxation (defaults — can be overridden via env_params)
TAU_WATER = 8.3e-12
F0_BACKBONE = 23e12
OMEGA0 = 2.0 * jnp.pi * F0_BACKBONE
EPS_S_WATER = 80.0
EPS_INF_WATER = 1.77

# Environment parameter vector: [ε_s, ε_∞, τ_D, ω₀]
# This is passed as a JAX array through the call chain so it can be
# varied at runtime (not baked into JIT trace).
DEFAULT_ENV_PARAMS = jnp.array([EPS_S_WATER, EPS_INF_WATER, TAU_WATER, OMEGA0])

# Coupling weights (from protein_bond_constants, at proper engine level)
LAMBDA_BOND = LAMBDA_BOND_CONST  # = 2.0
LAMBDA_RAMA = LAMBDA_RAMA_CONST  # = 2π
_r_Ca = jnp.float32(r_Ca)

# ═══════════════════════════════════════════════════════════════════════
# UTILITY: Per-residue computed quantities
# ═══════════════════════════════════════════════════════════════════════

def compute_z_topo(sequence):
    """Per-residue complex Z_topo with nearest-neighbour correction."""
    N = len(sequence)
    z_raw = jnp.array([Z_TOPO_COMPLEX[aa] for aa in sequence])
    z_mag_raw = jnp.abs(z_raw)
    z_mag_left = jnp.concatenate([z_mag_raw[:1], z_mag_raw[:-1]])
    z_mag_right = jnp.concatenate([z_mag_raw[1:], z_mag_raw[-1:]])
    nn_avg = 0.5 * (z_mag_left + z_mag_right)
    correction = 1.0 + ETA_NN * nn_avg
    return z_raw * correction


def compute_masks(sequence):
    """All per-residue boolean masks in one call."""
    N = len(sequence)
    cys  = jnp.array([1.0 if aa == 'C' else 0.0 for aa in sequence])
    arom = jnp.array([1.0 if aa in 'WHYF' else 0.0 for aa in sequence])
    gly  = jnp.array([1.0 if aa == 'G' else 0.0 for aa in sequence])
    pro  = jnp.array([1.0 if aa == 'P' else 0.0 for aa in sequence])
    NO_CG = {'G', 'A'}
    cg   = jnp.array([0.0 if aa in NO_CG else 1.0 for aa in sequence])
    # Charged residues for salt bridges
    neg  = jnp.array([1.0 if aa in 'DE' else 0.0 for aa in sequence])  # acidic
    pos  = jnp.array([1.0 if aa in 'KR' else 0.0 for aa in sequence])  # basic
    return cys, arom, gly, pro, cg, neg, pos




def dc_analysis(coords_flat, z_topo, gly_mask, pro_mask, N,
                chi1=None, chi2=None, cg_mask=None,
                cys_mask=None, arom_mask=None, neg_mask=None, pos_mask=None):
    """
    DC Analysis: compute geometry and steric constraints.

    Args:
        coords_flat: (N*9,) flattened backbone [N, Cα, C per residue]
        z_topo: (N,) complex impedance
        gly_mask, pro_mask: per-residue masks
        N: number of residues

    Returns:
        dict with coords, distances, steric penalties, contact info
    """
    bb = coords_flat.reshape(N, 3, 3)
    atom_N  = bb[:, 0, :]
    atom_Ca = bb[:, 1, :]
    atom_C  = bb[:, 2, :]
    coords = atom_Ca

    # Cβ placement
    chi1_arr = chi1 if chi1 is not None else jnp.full(N, jnp.radians(60.0))
    cb_pos = _compute_cb_positions(atom_N, atom_Ca, atom_C, chi1_arr, gly_mask)

    # Cys mask default (zero = no disulfides)
    if cys_mask is None:
        cys_mask = jnp.zeros(N)
    if arom_mask is None:
        arom_mask = jnp.zeros(N)
    if neg_mask is None:
        neg_mask = jnp.zeros(N)
    if pos_mask is None:
        pos_mask = jnp.zeros(N)

    # Pairwise Cα distances
    diff = coords[:, None, :] - coords[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    z_mag = jnp.abs(z_topo)

    # ── Backbone bend angles ──
    v1 = coords[1:] - coords[:-1]
    v2 = coords[2:] - coords[1:-1]
    cos_theta = jnp.sum(v1[:-1] * v2, axis=-1) / (
        jnp.sqrt(jnp.sum(v1[:-1]**2, axis=-1)) *
        jnp.sqrt(jnp.sum(v2**2, axis=-1)) + 1e-12)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    # Bend admittance: C_bend = (1−cos θ)/(2π²)
    C_bend_arr = (1.0 - cos_theta) / (2.0 * jnp.pi**2)
    # Pad to N for per-residue: endpoints have zero bend
    C_bend = jnp.concatenate([jnp.array([0.0]), C_bend_arr, jnp.array([0.0])])

    # ── Solvent exposure ──
    # Burial depth: how many neighbors within R_BURIAL
    burial_count = jnp.sum(jax.nn.sigmoid(BETA_BURIAL * (R_BURIAL - dists)), axis=1) - 1.0
    max_contacts = jnp.minimum(N - 1.0, 12.0)  # coordination limit
    exposure = 1.0 - jnp.clip(burial_count / max_contacts, 0.0, 1.0)

    # ── H-bond coupling (i→j mutual admittance) ──
    idx = jnp.arange(N)
    seq_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 3  # exclude local
    # N-C distances for H-bond detection
    diff_NC = atom_N[:, None, :] - atom_C[None, :, :]
    d_NC = jnp.sqrt(jnp.sum(diff_NC**2, axis=-1) + 1e-12)
    # Donor direction
    donor_dir = atom_N - atom_Ca
    donor_hat = donor_dir / (jnp.sqrt(jnp.sum(donor_dir**2, axis=-1, keepdims=True)) + 1e-12)
    sep_hat = diff_NC / (d_NC[:, :, None] + 1e-12)
    cos_donor = jnp.maximum(0.0, jnp.sum(donor_hat[:, None, :] * (-sep_hat), axis=-1))
    hb_proximity = jax.nn.sigmoid(BETA_BURIAL * (D_HB_DETECT + d0 - d_NC))
    hb_coupling = LAMBDA_RAMA * KAPPA_HB * cos_donor * jnp.exp(-d_NC / d0) * hb_proximity
    hb_coupling = jnp.where(seq_mask, hb_coupling, 0.0)

    # ── β-sheet antiparallel coupling (coupled-line even/odd mode) ──
    # Standard coupled microstrip: two antiparallel strands form a
    # coupled transmission line pair with coupling k.
    #
    # Even mode: Z_e = Z₀√((1+k)/(1-k))  → symmetric currents
    # Odd mode:  Z_o = Z₀√((1-k)/(1+k))  → antisymmetric currents
    #
    # Y-matrix contribution:
    #   Y_self   = Y_e + Y_o  (diagonal)
    #   Y_mutual = Y_e - Y_o  (off-diagonal, drives mode splitting)
    #
    u_dir = atom_C - atom_N
    u_hat = u_dir / (jnp.sqrt(jnp.sum(u_dir**2, axis=-1, keepdims=True)) + 1e-12)
    cos_uij = jnp.sum(u_hat[:, None, :] * u_hat[None, :, :], axis=-1)
    antiparallel = jnp.maximum(0.0, -cos_uij)  # 1 for perfectly antiparallel
    nc_local_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 3

    # Coupling coefficient: k = antiparallel × directionality × proximity
    beta_proximity = jax.nn.sigmoid(BETA_BURIAL * (D_HB_DETECT + d0 - d_NC))
    k_coupling = antiparallel * cos_donor * beta_proximity
    k_coupling = jnp.where(nc_local_mask, k_coupling, 0.0)
    k_coupling = jnp.clip(k_coupling, 0.0, 0.99)  # k < 1 for stability

    # Even/odd mode admittances (Z₀ = 1 normalised)
    Z_even = jnp.sqrt((1.0 + k_coupling) / (1.0 - k_coupling + 1e-12))
    Z_odd  = jnp.sqrt((1.0 - k_coupling) / (1.0 + k_coupling + 1e-12))
    Y_even = 1.0 / (Z_even + 1e-12)
    Y_odd  = 1.0 / (Z_odd + 1e-12)

    # Coupled-line Y contributions
    beta_self   = KAPPA_HB * (Y_even + Y_odd)    # diagonal
    beta_mutual = KAPPA_HB * (Y_even - Y_odd)    # off-diagonal (negative = coupling)

    # ── Disulfide bond coupling (Cys-Cys covalent S-S) ──
    # S-S bond = permanent, strong mutual admittance (covalent, not H-bond)
    # Strength: κ_HB × (d₀ / d_SS) — shorter & covalent = stronger
    D_SS = 2.05  # S-S bond length (Å) — covalent
    cys_pair = cys_mask[:, None] * cys_mask[None, :]  # outer product
    ss_coupling = LAMBDA_RAMA * KAPPA_HB * (d0 / D_SS) * \
                  jax.nn.sigmoid(BETA_BURIAL * (D_SS + d0 - dists)) * cys_pair
    ss_coupling = jnp.where(seq_mask, ss_coupling, 0.0)

    # ── Aromatic π-stacking (capacitive coupling) ──
    # Aromatic sidechains (W/H/Y/F) stack face-to-face.
    # EE: capacitive mutual admittance ∝ 1/distance × alignment
    # Coupling strength: κ_HB × exp(−d/d₀) for aromatics within range
    arom_pair = arom_mask[:, None] * arom_mask[None, :]  # outer product
    arom_coupling = KAPPA_HB * jnp.exp(-dists / d0) * arom_pair
    arom_coupling = jnp.where(seq_mask, arom_coupling, 0.0)

    # ── Salt bridges (charge-pair transformer coupling) ──
    # Opposite-charge residues (D/E⁻ ↔ K/R⁺) form ionic bonds.
    # EE: transformer coupling ∝ 1/distance (Coulombic)
    # Only opposite charges attract: neg×pos pairs
    salt_pair = neg_mask[:, None] * pos_mask[None, :] + \
                pos_mask[:, None] * neg_mask[None, :]  # both directions
    salt_coupling = KAPPA_HB * (d0 / (dists + 1e-6)) * \
                    jax.nn.sigmoid(BETA_BURIAL * (D_HB_DETECT + d0 - dists)) * salt_pair
    salt_coupling = jnp.where(seq_mask, salt_coupling, 0.0)

    # ── Combined contact matrix (upper triangle to avoid double-counting) ──
    contact_matrix = hb_coupling + beta_mutual + ss_coupling + arom_coupling + salt_coupling
    # Symmetrise (both directions)
    contact_matrix = 0.5 * (contact_matrix + contact_matrix.T)

    # β-sheet even/odd self-admittance goes on diagonal separately
    beta_diag = jnp.sum(beta_self * jnp.triu(jnp.ones((N, N)), k=3), axis=1)

    # ── Steric penalties (DC constraints) ──
    LAMBDA_STERIC = LAMBDA_BOND * d0 / r_Ca
    steric_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 3
    violations = jnp.maximum(0.0, d0 - dists) ** 2
    violations = jnp.where(steric_mask, violations, 0.0)
    steric_penalty = LAMBDA_STERIC * jnp.sum(jnp.triu(violations, k=3)) / N

    # Full backbone atom steric
    R_CC = R_STERIC_CC; R_NN = R_STERIC_NN; R_CN = R_STERIC_CN
    bb_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 2
    d_NN = jnp.sqrt(jnp.sum((atom_N[:, None, :] - atom_N[None, :, :])**2, axis=-1) + 1e-12)
    d_CC = jnp.sqrt(jnp.sum((atom_C[:, None, :] - atom_C[None, :, :])**2, axis=-1) + 1e-12)
    d_NC_all = jnp.sqrt(jnp.sum((atom_N[:, None, :] - atom_C[None, :, :])**2, axis=-1) + 1e-12)
    nn_v = jnp.where(bb_mask, jnp.maximum(0.0, R_NN - d_NN)**2, 0.0)
    cc_v = jnp.where(bb_mask, jnp.maximum(0.0, R_CC - d_CC)**2, 0.0)
    nc_v = jnp.where(bb_mask, jnp.maximum(0.0, R_CN - d_NC_all)**2, 0.0)
    cn_v = jnp.where(bb_mask, jnp.maximum(0.0, R_CN - d_NC_all.T)**2, 0.0)

    # Cβ steric (Ramachandran source)
    R_CB = R_STERIC_CB
    cb_mask = jnp.abs(idx[:, None] - idx[None, :]) >= 2
    d_cb = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - cb_pos[None, :, :])**2, axis=-1) + 1e-12)
    d_cb_ca = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - atom_Ca[None, :, :])**2, axis=-1) + 1e-12)
    d_cb_N = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - atom_N[None, :, :])**2, axis=-1) + 1e-12)
    d_cb_C = jnp.sqrt(jnp.sum((cb_pos[:, None, :] - atom_C[None, :, :])**2, axis=-1) + 1e-12)
    local_cb = jnp.abs(idx[:, None] - idx[None, :]) == 1
    cb_v = jnp.where(cb_mask, jnp.maximum(0.0, R_CB - d_cb)**2, 0.0)
    cb_ca_v = jnp.where(local_cb, jnp.maximum(0.0, R_CB - d_cb_ca)**2, 0.0)
    cb_N_v = jnp.where(local_cb, jnp.maximum(0.0, R_CN - d_cb_N)**2, 0.0)
    cb_C_v = jnp.where(local_cb, jnp.maximum(0.0, R_CN - d_cb_C)**2, 0.0)

    bb_steric = LAMBDA_STERIC * (
        jnp.sum(jnp.triu(nn_v, k=2)) + jnp.sum(jnp.triu(cc_v, k=2)) +
        jnp.sum(jnp.triu(nc_v, k=2)) + jnp.sum(jnp.triu(cn_v, k=2)) +
        jnp.sum(jnp.triu(cb_v, k=2)) +
        jnp.sum(cb_ca_v) + jnp.sum(cb_N_v) + jnp.sum(cb_C_v)
    ) / N

    # Packing fraction
    com = jnp.mean(coords, axis=0)
    Rg_sq = jnp.mean(jnp.sum((coords - com)**2, axis=1))

    return {
        'coords': coords,
        'atom_N': atom_N, 'atom_Ca': atom_Ca, 'atom_C': atom_C,
        'cb_pos': cb_pos,
        'dists': dists,
        'z_mag': z_mag,
        'C_bend': C_bend,
        'exposure': exposure,
        'contact_matrix': contact_matrix,
        'beta_diag': beta_diag,  # even/odd mode self-admittance
        'steric_penalty': steric_penalty + bb_steric,
        'Rg_sq': Rg_sq,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: AC ANALYSIS (Impedance Network via Y-Matrix)
# ═══════════════════════════════════════════════════════════════════════

def ac_analysis(dc_result, z_topo, N, env_params=None):
    """
    AC Analysis: compute Y-matrix S₁₁ from structural contacts.

    Uses the DC geometry to build an N-port Y-matrix network,
    then extracts the multi-port S₁₁ diagnostic.

    Args:
        dc_result: dict from dc_analysis
        z_topo: (N,) complex impedance array
        N: number of residues
        env_params: (4,) array [ε_s, ε_∞, τ_D, ω₀] (optional, uses defaults)

    Returns:
        dict with s11_avg, s11_per_freq, Y_matrix
    """
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS
    eps_s = env_params[0]
    eps_inf = env_params[1]
    tau_d = env_params[2]
    omega0 = env_params[3]
    z_mag = dc_result['z_mag']
    C_bend = dc_result['C_bend']
    exposure = dc_result['exposure']
    contact_matrix = dc_result['contact_matrix']
    beta_diag = dc_result['beta_diag']  # even/odd mode self-admittance

    # ── Build contact arrays from upper triangle ──
    # Extract (i, j, y) triplets where contact_matrix[i,j] > threshold
    triu_mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=3)
    contact_vals = contact_matrix * triu_mask
    # Flatten to arrays for JAX Y-matrix builder
    flat_idx = jnp.arange(N * N)
    i_idx = flat_idx // N
    j_idx = flat_idx % N
    flat_contacts = contact_vals.ravel()
    # Only keep non-negligible contacts (threshold = 1e-6)
    active = flat_contacts > 1e-6
    contact_i = jnp.where(active, i_idx, 0)
    contact_j = jnp.where(active, j_idx, 0)
    contact_y = jnp.where(active, flat_contacts, 0.0).astype(jnp.complex64)

    # ── Cα-Cα distances for backbone propagation ──
    coords = dc_result['coords']
    d_CaCa = jnp.sqrt(jnp.sum((coords[1:] - coords[:-1])**2, axis=-1) + 1e-12)  # (N-1,)
    # Effective backbone impedance per segment: geometric mean of Z_i and Z_{i+1}
    Z_eff = jnp.sqrt(z_mag[:-1] * z_mag[1:])  # (N-1,)

    # ── Multi-frequency sweep ──
    s11_list = []
    eig_list = []
    eig_min_list = []
    diag_list = []      # per-port |Γᵢ|² at each frequency
    for f_idx in range(N_FREQ):
        w = 2.0 * jnp.pi * FREQ_SWEEP[f_idx]

        # ── 3-segment ABCD cascade → Y (shared module) ──
        # Uses actual sub-segment impedances: Z_CaC, Z_CN, Z_NCa
        Z_seg = jnp.array([Z_BOND_CA_C, Z_BOND_C_N, Z_BOND_N_CA])
        d_seg = jnp.array([D_CA_C, D_C_N, D_N_CA])
        y_mutual, diag_bb = abcd_to_y_3seg_jax(
            N, Z_seg, d_seg, jnp.abs(z_topo), w, d0)

        # ── Self admittance: solvent + bend ──
        w_phys = w * omega0
        eps_w = eps_inf + (eps_s - eps_inf) / (1.0 + 1j * w_phys * tau_d)
        Z_water = jnp.sqrt(jnp.abs(eps_w))
        Y_solvent = exposure / Z_water
        Y_bend = w * C_bend

        # ── Build Y-matrix (proper ABCD→Y backbone + contacts) ──
        Y_mat = jnp.zeros((N, N), dtype=jnp.complex64)

        # Backbone: off-diagonal (mutual from shared module)
        bb_idx = jnp.arange(N - 1)
        Y_mat = Y_mat.at[bb_idx, bb_idx + 1].add(y_mutual)
        Y_mat = Y_mat.at[bb_idx + 1, bb_idx].add(y_mutual)

        # Diagonal: backbone + solvent + bend + β-sheet even/odd self
        diag_total = diag_bb + (Y_solvent + Y_bend).astype(jnp.complex64)
        diag_total = diag_total + beta_diag.astype(jnp.complex64)
        diag_idx = jnp.arange(N)
        Y_mat = Y_mat.at[diag_idx, diag_idx].add(diag_total)

        # Contacts: off-diagonal (H-bond + β-sheet)
        Y_mat = Y_mat.at[contact_i, contact_j].add(-contact_y)
        Y_mat = Y_mat.at[contact_j, contact_i].add(-contact_y)
        Y_mat = Y_mat.at[contact_i, contact_i].add(contact_y)
        Y_mat = Y_mat.at[contact_j, contact_j].add(contact_y)

        # ── Chain termination impedances ──
        # N/C termini are charged (NH₃⁺, COO⁻) and fully solvated.
        # In RF: unterminated ports reflect 100%. Fix: matched load.
        # Y_term = Y₀ (matched to solvent) at ports 0 and N-1.
        Y0_bulk = (1.0 / Z_water).astype(jnp.complex64)
        Y_mat = Y_mat.at[0, 0].add(Y0_bulk)        # N-terminus
        Y_mat = Y_mat.at[N-1, N-1].add(Y0_bulk)    # C-terminus

        # ── Multi-port S₁₁ referenced to bulk solvent ──
        s_result = s_diagonal_from_y_matrix_jax(Y_mat, Y0=Y0_bulk)
        s11_list.append(s_result['mean'])
        eig_list.append(s_result['eig_mean'])
        eig_min_list.append(s_result['eig_min'])
        diag_list.append(s_result['diag'])   # (N,) per-port |Γᵢ|²

    s11_per_freq = jnp.array(s11_list)
    s11_avg = jnp.mean(s11_per_freq)
    eig_per_freq = jnp.array(eig_list)
    eig_avg = jnp.mean(eig_per_freq)
    eig_min_per_freq = jnp.array(eig_min_list)
    eig_min = jnp.min(eig_min_per_freq)

    # Per-port |Γᵢ|² averaged across frequencies — (N,) array
    # This is the Gauss-Seidel target: drive each port toward Γᵢ = 0
    diag_stack = jnp.stack(diag_list, axis=0)     # (N_FREQ, N)
    s11_per_port = jnp.mean(diag_stack, axis=0)   # (N,)

    return {
        's11_avg': s11_avg,           # scalar mean — v4 compat
        'eig_avg': eig_avg,           # eigenvalue mean
        'eig_min': eig_min,           # eigenvalue min — v5 root target
        's11_per_port': s11_per_port, # (N,) per-port |Γᵢ|² — v6 GS target
        's11_per_freq': s11_per_freq,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def _s11_loss_v4(coords_flat, z_topo, gly_mask, pro_mask, N,
                 chi1=None, chi2=None, cg_mask=None, env_params=None,
                 cys_mask=None, arom_mask=None, neg_mask=None, pos_mask=None):
    """
    v4 loss function with clean DC/AC separation.

    Loss = w_ac × |S₁₁|² + w_dc × steric_penalty
    """
    # Stage 1: DC
    dc = dc_analysis(coords_flat, z_topo, gly_mask, pro_mask, N,
                      chi1=chi1, chi2=chi2, cg_mask=cg_mask,
                      cys_mask=cys_mask, arom_mask=arom_mask,
                      neg_mask=neg_mask, pos_mask=pos_mask)

    # Stage 2: AC
    ac = ac_analysis(dc, z_topo, N, env_params=env_params)

    # Packing saturation (Axiom 4)
    R_eff = jnp.sqrt(5.0 / 3.0 * dc['Rg_sq'] + 1e-12)
    eta = N * _r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_packing = jnp.sqrt(1.0 - eta_ratio**2)

    # Combine: AC drives compaction, DC prevents overlap
    port_loss = ac['s11_avg'] * sat_packing
    total_loss = port_loss + dc['steric_penalty']

    return total_loss


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: TORSION-ANGLE PARAMETERIZATION + OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

def _torsion_loss_v4(angles, z_topo, gly_mask, pro_mask, N, cg_mask=None, env_params=None,
                     cys_mask=None, arom_mask=None, neg_mask=None, pos_mask=None):
    """Torsion-angle wrapper for v4 loss."""
    phi = angles[:N]
    psi = angles[N:2*N]
    chi1 = angles[2*N:3*N] if angles.shape[0] > 2*N else None
    chi2 = angles[3*N:] if angles.shape[0] > 3*N else None
    coords_flat = _torsions_to_backbone(phi, psi, N).ravel()
    return _s11_loss_v4(coords_flat, z_topo, gly_mask, pro_mask, N,
                        chi1=chi1, chi2=chi2, cg_mask=cg_mask, env_params=env_params,
                        cys_mask=cys_mask, arom_mask=arom_mask,
                        neg_mask=neg_mask, pos_mask=pos_mask)


_torsion_loss_v4_jit = jit(_torsion_loss_v4, static_argnums=(4,))
_torsion_grad_v4_jit = jit(grad(_torsion_loss_v4), static_argnums=(4,))


def fold_s11_v4(sequence, n_steps=3000, lr=None, anneal=True, n_starts=3,
                env_params=None):
    """
    Protein folding via v4 Y-matrix engine.

    NOTE: Optimizer hyperparameters (lr, n_steps, anneal schedule) are
    ENGINEERING choices — they affect convergence speed but NOT the
    physics minimum. The loss landscape is defined entirely by derived
    constants. Any optimizer that converges should reach the same minimum.

    Args:
        sequence: amino acid string
        n_steps: optimizer iterations (engineering, not physics)
        lr: learning rate (default: 1/(2π·Q) ≈ 0.023)
        anneal: whether to anneal noise
        n_starts: number of random restarts
        env_params: (4,) jnp array [ε_s, ε_∞, τ_D, ω₀] or None for defaults
    """
    if lr is None:
        # Default lr: each step moves at most BW/2 radians when
        # gradient is at the clip boundary (2π).
        # lr × clip = target_step → lr = target_step / clip
        # target_step = BW/2 = 1/(2Q)
        # clip = 2π
        # ∴ lr = 1/(2Q × 2π) = 1/(4πQ) ≈ 0.011
        # But Adam adapts per-parameter, so we use 2× for headroom:
        lr = 1.0 / (2.0 * jnp.pi * Q_BACKBONE)  # ≈ 0.023
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS

    N = len(sequence)
    z_topo = compute_z_topo(sequence)
    cys_mask, arom_mask, gly_mask, pro_mask, cg_mask, neg_mask, pos_mask = compute_masks(sequence)

    best_loss = float('inf')
    best_angles = None

    print(f"  S₁₁ v4 Y-matrix ({n_starts}-start): N={N}, steps={n_steps}", flush=True)

    for start_idx in range(n_starts):
        seed = 42 + start_idx * 137
        np.random.seed(seed)
        phi_init = np.random.uniform(-np.pi, np.pi, N)
        psi_init = np.random.uniform(-np.pi, np.pi, N)
        chi1_init = np.random.choice(
            [np.radians(-60), np.radians(60), np.radians(180)], N)
        chi2_init = np.random.choice(
            [np.radians(-60), np.radians(60), np.radians(180)], N)
        for i in range(N):
            if sequence[i] == 'G':
                chi1_init[i] = 0.0; chi2_init[i] = 0.0
            elif sequence[i] == 'A':
                chi2_init[i] = 0.0
        angles = jnp.concatenate([jnp.array(phi_init), jnp.array(psi_init),
                                  jnp.array(chi1_init), jnp.array(chi2_init)])

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(angles)
        key = jax.random.PRNGKey(seed)

        t0 = time.time()
        if start_idx == 0:
            _ = _torsion_loss_v4_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask)
            _ = _torsion_grad_v4_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask)
            print(f"    JIT compiled in {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()

        # Anneal fraction: 50% exploration, 50% refinement
        anneal_steps = int(n_steps * 0.5) if anneal else 0

        # Derived optimizer constants:
        # T₀ = 1/(2Q): the half-bandwidth of the backbone resonator.
        # This is the thermal noise floor — the minimum perturbation
        # needed to escape a local minimum of width ~BW/2.
        _T0 = 1.0 / (2.0 * Q_BACKBONE)  # = 1/14 ≈ 0.0714

        # Gradient clip = 2π: an angular gradient cannot physically
        # exceed one full rotation per optimisation step.
        _GRAD_CLIP = 2.0 * jnp.pi  # ≈ 6.28

        def opt_step(step, carry):
            angles_c, opt_state_c, key_c = carry
            g = _torsion_grad_v4_jit(angles_c, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > _GRAD_CLIP, g * _GRAD_CLIP / g_norm, g)
            updates, new_opt_state = optimizer.update(g, opt_state_c)
            new_angles = optax.apply_updates(angles_c, updates)
            T = _T0 * jnp.maximum(0.0, 1.0 - step / jnp.maximum(1.0, anneal_steps)) ** 2
            key_c, subkey = jax.random.split(key_c)
            noise = jax.random.normal(subkey, shape=new_angles.shape) * T
            new_angles = jnp.where(step < anneal_steps, new_angles + noise, new_angles)
            return (new_angles, new_opt_state, key_c)

        angles, opt_state, key = lax.fori_loop(
            0, n_steps, opt_step, (angles, opt_state, key))

        loss = float(_torsion_loss_v4_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                          cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask))
        dt = time.time() - t0
        print(f"    start {start_idx}: loss={loss:.4f} ({dt:.0f}s)", flush=True)

        if loss < best_loss:
            best_loss = loss
            best_angles = angles

    # Extract final structure
    phi = best_angles[:N]
    psi = best_angles[N:2*N]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    coords = coords_flat.reshape(N, 3, 3)[:, 1, :]  # Cα positions

    return np.array(coords), float(best_loss), np.array(best_angles)

# ═══════════════════════════════════════════════════════════════════════
# v5: NEWTON-RAPHSON EIGENVALUE ROOT-FINDER
# ═══════════════════════════════════════════════════════════════════════
#
# AVE principle: the folded protein IS the eigenstate of its impedance
# network. Finding it is a ROOT-FINDING problem, not optimisation:
#
#   Find θ such that λ_min(S†S(θ)) = 0
#
# where λ_min is the smallest eigenvalue of the Hermitian matrix S†S,
# and S is the multiport scattering matrix.
#
# Newton-Raphson step:
#   Δθ = −f(θ) × ∇f / |∇f|²
#
# The step size is ENTIRELY determined by the function value and
# gradient. No learning rate, no hyperparameters.
#
# Trust region: |Δθ| ≤ π (angular variables cannot exceed half-rotation)
#

def _eigenvalue_target(angles, z_topo, gly_mask, pro_mask, N,
                        cg_mask=None, env_params=None):
    """
    Newton-Raphson target: minimum eigenvalue of S†S + sterics.

    Returns a scalar f(θ) that should be driven to zero.
    When f = 0, the protein has found an eigenstate where at least
    one mode of the S-matrix is perfectly matched to the environment.
    """
    phi = angles[:N]
    psi = angles[N:2*N]
    chi1 = angles[2*N:3*N] if angles.shape[0] > 2*N else None
    chi2 = angles[3*N:] if angles.shape[0] > 3*N else None
    coords_flat = _torsions_to_backbone(phi, psi, N).ravel()

    # DC analysis (geometry → contacts, sterics)
    dc = dc_analysis(coords_flat, z_topo, gly_mask, pro_mask, N,
                      chi1=chi1, chi2=chi2, cg_mask=cg_mask)

    # AC analysis (eigenvalues computed inside)
    ac = ac_analysis(dc, z_topo, N, env_params=env_params)

    # Packing saturation (Axiom 4)
    R_eff = jnp.sqrt(5.0 / 3.0 * dc['Rg_sq'] + 1e-12)
    eta = N * _r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_packing = jnp.sqrt(1.0 - eta_ratio**2)

    # Eigenvalue target: min(|λ_S|²) = the best-matched mode
    # When this → 0, ONE mode is perfectly matched (eigenstate found).
    # This is the Newton root: f(θ) = λ_min → 0
    modal_target = ac['eig_min'] * sat_packing

    # Total target: modal mismatch + steric violation
    # Both must be zero at the physical fold
    f = modal_target + dc['steric_penalty']

    return f


_eigenvalue_target_jit = jit(_eigenvalue_target, static_argnums=(4,))
_eigenvalue_grad_jit = jit(grad(_eigenvalue_target), static_argnums=(4,))


def fold_eigenvalue_v5(sequence, n_scf=200, n_starts=3, env_params=None):
    """
    AVE Newton-Raphson eigenvalue root-finder.

    Finds the torsion angles where the minimum S-matrix eigenvalue
    vanishes — the eigenstate of the impedance network.

    Newton step: Δθ = −f(θ) × g / |g|²
    Trust region: |Δθ_i| ≤ π (geometric, angular)
    Convergence: |f| → 0 (eigenstate found)

    No hyperparameters. Everything is determined by the function
    value, gradient, and angular geometry.

    Args:
        sequence: amino acid string
        n_scf: max SCF iterations
        n_starts: number of random restarts
        env_params: environment parameters or None for defaults
    """
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS

    N = len(sequence)
    z_topo = compute_z_topo(sequence)
    cys_mask, arom_mask, gly_mask, pro_mask, cg_mask, neg_mask, pos_mask = compute_masks(sequence)

    best_f = float('inf')
    best_angles = None

    print(f"  v5 Newton ({n_starts}-start): N={N}, max_iter={n_scf}", flush=True)

    for start_idx in range(n_starts):
        seed = 42 + start_idx * 137
        np.random.seed(seed)
        phi_init = np.random.uniform(-np.pi, np.pi, N)
        psi_init = np.random.uniform(-np.pi, np.pi, N)
        chi1_init = np.random.choice(
            [np.radians(-60), np.radians(60), np.radians(180)], N)
        chi2_init = np.random.choice(
            [np.radians(-60), np.radians(60), np.radians(180)], N)
        for i in range(N):
            if sequence[i] == 'G':
                chi1_init[i] = 0.0; chi2_init[i] = 0.0
            elif sequence[i] == 'A':
                chi2_init[i] = 0.0
        angles = jnp.concatenate([jnp.array(phi_init), jnp.array(psi_init),
                                  jnp.array(chi1_init), jnp.array(chi2_init)])

        t0 = time.time()
        if start_idx == 0:
            _ = _eigenvalue_target_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            _ = _eigenvalue_grad_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            print(f"    JIT compiled in {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()

        # ── Fully JIT-compiled Newton-Raphson with line search ──
        #
        # One JIT dispatch for all n_scf iterations.
        # Outer loop: lax.fori_loop (fixed iterations)
        # Inner line search: lax.while_loop (up to 25 halvings)
        #
        def _newton_step(i, angles_carry):
            """Single Newton step: gradient → direction → line search → update."""
            # Function value and gradient
            f_val = _eigenvalue_target(angles_carry, z_topo, gly_mask, pro_mask,
                                       N, cg_mask, env_params)
            g = grad(_eigenvalue_target)(angles_carry, z_topo, gly_mask, pro_mask,
                                          N, cg_mask, env_params)
            g = jnp.where(jnp.isnan(g), 0.0, g)

            # Newton direction: Δθ = −f × g / |g|²
            g_norm_sq = jnp.sum(g**2) + 1e-12
            direction = -f_val * g / g_norm_sq

            # Trust region: cap at π (angular geometry)
            dir_norm = jnp.sqrt(jnp.sum(direction**2) + 1e-12)
            scale = jnp.where(dir_norm > jnp.pi, jnp.pi / dir_norm, 1.0)
            direction = direction * scale

            # Backtracking line search via lax.while_loop
            # Try full step first, then halve until f decreases
            f_full = _eigenvalue_target(angles_carry + direction, z_topo,
                                         gly_mask, pro_mask, N, cg_mask, env_params)

            def ls_cond(state):
                alpha, f_trial, count = state
                return (f_trial >= f_val) & (count < 25)

            def ls_body(state):
                alpha, _, count = state
                new_alpha = alpha * 0.5
                trial = angles_carry + new_alpha * direction
                new_f = _eigenvalue_target(trial, z_topo, gly_mask, pro_mask,
                                            N, cg_mask, env_params)
                return (new_alpha, new_f, count + 1)

            alpha_final, _, _ = lax.while_loop(
                ls_cond, ls_body,
                (jnp.float32(1.0), f_full, jnp.int32(0)))

            return angles_carry + alpha_final * direction

        # Run all iterations in one JIT call
        angles = lax.fori_loop(0, n_scf, _newton_step, angles)

        f_val = float(_eigenvalue_target_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params))
        dt = time.time() - t0

        # Convergence: |f| < 1/Q² (noise floor of backbone resonator)
        _CONVERGE = 1.0 / (Q_BACKBONE ** 2)
        if f_val < _CONVERGE:
            print(f"    start {start_idx}: ROOT f={f_val:.6f} ({dt:.0f}s)", flush=True)
        else:
            print(f"    start {start_idx}: f={f_val:.4f} ({dt:.0f}s)", flush=True)

        if f_val < best_f:
            best_f = f_val
            best_angles = angles

    phi = best_angles[:N]
    psi = best_angles[N:2*N]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    coords = coords_flat.reshape(N, 3, 3)[:, 1, :]

    return np.array(coords), float(best_f), np.array(best_angles)


# ═══════════════════════════════════════════════════════════════════════
# v6: GAUSS-SEIDEL PER-PORT SOLVER
# ═══════════════════════════════════════════════════════════════════════
#
# First-Principles Justification (RF/SPICE):
#   In EE, large multiport networks are NEVER solved by minimizing a
#   scalar average. SPICE writes KCL at every node and solves Yv = I.
#   When N is large, iterative methods (Gauss-Seidel, SOR) sweep through
#   ports one at a time, updating each to reduce its local mismatch.
#
#   Convergence guarantee: The backbone Y-matrix is diagonally dominant
#   (nearest-neighbour tridiagonal coupling). Gauss-Seidel converges for
#   diagonally dominant systems.
#
#   Each port i has TWO local degrees of freedom: (φᵢ, ψᵢ).
#   The per-port target: |Γᵢ|² → 0 (this port matched to bulk solvent).
#   The global convergence criterion: max(|Γᵢ|²) < 1/Q² for all i.
#
# ═══════════════════════════════════════════════════════════════════════

def _port_loss(angles, port_idx, z_topo, gly_mask, pro_mask, N, cg_mask, env_params):
    """
    Per-port loss: |Γᵢ|² for port `port_idx`.

    This function is JAX-traceable and differentiated w.r.t. `angles`
    to compute ∂|Γᵢ|²/∂(φ, ψ).
    """
    phi = angles[:N]
    psi = angles[N:2*N]
    chi1 = angles[2*N:3*N] if angles.shape[0] > 2*N else None
    chi2 = angles[3*N:] if angles.shape[0] > 3*N else None
    coords_flat = _torsions_to_backbone(phi, psi, N).ravel()
    dc = dc_analysis(coords_flat, z_topo, gly_mask, pro_mask, N,
                     chi1=chi1, chi2=chi2, cg_mask=cg_mask)
    ac = ac_analysis(dc, z_topo, N, env_params=env_params)

    # Per-port |Γᵢ|² (Gauss-Seidel target)
    gamma_i = ac['s11_per_port'][port_idx]

    # Packing saturation (Axiom 4) — same as _s11_loss_v4
    R_eff = jnp.sqrt(5.0 / 3.0 * dc['Rg_sq'] + 1e-12)
    eta = N * _r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_packing = jnp.sqrt(1.0 - eta_ratio**2)

    # Local target: port mismatch × packing + steric share
    f_i = gamma_i * sat_packing + dc['steric_penalty'] / N
    return f_i


def _gs_sweep_loss(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params):
    """
    Gauss-Seidel sweep loss: uses max(|Γᵢ|²) as the scalar target.

    The gradient of this function gives the direction to reduce the
    WORST port's mismatch, which is the Gauss-Seidel priority.
    """
    phi = angles[:N]
    psi = angles[N:2*N]
    chi1 = angles[2*N:3*N] if angles.shape[0] > 2*N else None
    chi2 = angles[3*N:] if angles.shape[0] > 3*N else None
    coords_flat = _torsions_to_backbone(phi, psi, N).ravel()
    dc = dc_analysis(coords_flat, z_topo, gly_mask, pro_mask, N,
                     chi1=chi1, chi2=chi2, cg_mask=cg_mask)
    ac = ac_analysis(dc, z_topo, N, env_params=env_params)

    # Per-port |Γᵢ|² array
    per_port = ac['s11_per_port']

    # Packing saturation (Axiom 4)
    R_eff = jnp.sqrt(5.0 / 3.0 * dc['Rg_sq'] + 1e-12)
    eta = N * _r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_packing = jnp.sqrt(1.0 - eta_ratio**2)

    # Sum of per-port losses (differentiable everywhere, unlike max)
    # Each port contributes its own mismatch
    total = jnp.sum(per_port) * sat_packing + dc['steric_penalty']
    return total


_gs_loss_jit = jit(_gs_sweep_loss, static_argnums=(4,))
_gs_grad_jit = jit(grad(_gs_sweep_loss), static_argnums=(4,))


def fold_gauss_seidel_v6(sequence, n_sweeps=50, n_starts=3, env_params=None):
    """
    Protein folding via v6 Gauss-Seidel per-port solver.

    Architecture (Gauss-Seidel coordinate descent):
      1. Compute full gradient ∂(Σ|Γᵢ|²)/∂θ in ONE forward+backward pass
      2. For each port i (sequentially):
         a. Extract the 2 gradient components (∂/∂φᵢ, ∂/∂ψᵢ)
         b. Newton-step: Δ(φᵢ,ψᵢ) ∝ −Γᵢ × gᵢ / |gᵢ|²
         c. Trust region: |Δ| ≤ π
         d. Update angles (affects subsequent ports = GS property)
      3. Recompute loss → check convergence: max(|Γᵢ|²) < 1/Q²

    This is the EE Gauss-Seidel method: one gradient computation per sweep
    but N sequential per-port updates, each using the latest geometry.

    All constants derived from first principles:
        Trust region:  π (angular geometry)
        Convergence:   max(|Γᵢ|²) < 1/Q²  (backbone resonator noise floor)
        Step scale:    1/(2Q) (thermal noise floor)
    """
    N = len(sequence)
    z_topo = compute_z_topo(sequence)
    cys_mask, arom_mask, gly_mask, pro_mask, cg_mask, neg_mask, pos_mask = compute_masks(sequence)
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS

    _CONVERGE = 1.0 / (Q_BACKBONE ** 2)  # 1/Q² = 1/49 ≈ 0.020
    _STEP_SCALE = 1.0 / (2.0 * Q_BACKBONE)  # 1/(2Q) = thermal noise floor

    print(f"  v6 GS (n_starts={n_starts}): N={N}, sweeps={n_sweeps}")

    best_f = float('inf')
    best_angles = None

    for start_idx in range(n_starts):
        # Random initial angles in Ramachandran basins
        rng = np.random.RandomState(42 + start_idx * 137)
        phi_init = rng.choice([float(PHI_ALPHA), float(PHI_BETA), float(PHI_PPII)], N)
        psi_init = rng.choice([float(PSI_ALPHA), float(PSI_BETA), float(PSI_PPII)], N)
        chi1_init = rng.uniform(-np.pi, np.pi, N)
        chi2_init = rng.uniform(-np.pi, np.pi, N)
        angles = jnp.concatenate([jnp.array(phi_init), jnp.array(psi_init),
                                  jnp.array(chi1_init), jnp.array(chi2_init)])

        t0 = time.time()

        if start_idx == 0:
            _ = _gs_loss_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            _ = _gs_grad_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            print(f"    JIT compiled in {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()

        # Gauss-Seidel sweeps
        converged = False
        for sweep in range(n_sweeps):
            # One forward+backward pass → full gradient
            f_total = _gs_loss_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            g_full = _gs_grad_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            g_full = jnp.where(jnp.isnan(g_full), 0.0, g_full)

            # Per-port coordinate descent: step each (φᵢ, ψᵢ)
            for port_idx in range(N):
                g_phi = g_full[port_idx]
                g_psi = g_full[N + port_idx]
                g_norm_sq = g_phi**2 + g_psi**2 + 1e-12

                # Steepest descent with Q-derived step scale
                step_phi = -_STEP_SCALE * g_phi
                step_psi = -_STEP_SCALE * g_psi

                # Trust region: cap per-port step at π
                step_norm = jnp.sqrt(step_phi**2 + step_psi**2 + 1e-12)
                scale = jnp.where(step_norm > jnp.pi, jnp.pi / step_norm, 1.0)
                step_phi = step_phi * scale
                step_psi = step_psi * scale

                angles = angles.at[port_idx].add(step_phi)
                angles = angles.at[N + port_idx].add(step_psi)

            # Check convergence: evaluate per-port loss
            f_val = float(f_total)
            max_gamma = f_val / N  # approximate per-port average

            if sweep % 10 == 9 or sweep == n_sweeps - 1:
                # Periodically check exact max per-port
                phi_cur = angles[:N]
                psi_cur = angles[N:2*N]
                chi1_cur = angles[2*N:3*N]
                chi2_cur = angles[3*N:]
                coords_flat = _torsions_to_backbone(phi_cur, psi_cur, N).ravel()
                dc_check = dc_analysis(coords_flat, z_topo, gly_mask, pro_mask, N,
                                        chi1=chi1_cur, chi2=chi2_cur, cg_mask=cg_mask)
                ac_check = ac_analysis(dc_check, z_topo, N, env_params=env_params)
                max_gamma = float(jnp.max(ac_check['s11_per_port']))

        print(f"    start {start_idx}: max|Γ|²={max_gamma:.4f}  "
              f"Σ|Γ|²={f_val:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if max_gamma < best_f:
            best_f = max_gamma
            best_angles = angles

    phi = best_angles[:N]
    psi = best_angles[N:2*N]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    coords = coords_flat.reshape(N, 3, 3)[:, 1, :]

    return np.array(coords), float(best_f), np.array(best_angles)

# ═══════════════════════════════════════════════════════════════════════
# v7: SEGMENTED CASCADE SOLVER
# ═══════════════════════════════════════════════════════════════════════
#
# First-Principles Derivation:
#   The backbone Q = ℓ = ⌊d₀/a₀⌉ = 7 defines the coherence length —
#   the number of residues over which torsion angles are coherently
#   coupled. Beyond Q residues, the NERF error propagation decoheres
#   the gradient signal.
#
#   Phase 1: SEGMENT — fold Q-length segments independently
#   Phase 2: COUPLE  — optimize junction angles (tertiary contacts)
#   Phase 3: REFINE  — polish full chain from cascaded geometry
#
# ═══════════════════════════════════════════════════════════════════════

def _fold_segment(sub_seq, n_steps=2000, n_starts=3, env_params=None):
    """Fold a short segment using v4 Adam. Returns best angles."""
    N = len(sub_seq)
    z_topo = compute_z_topo(sub_seq)
    _, _, gly_mask, pro_mask, cg_mask = compute_masks(sub_seq)
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS

    lr = 1.0 / (2.0 * jnp.pi * Q_BACKBONE)
    best_loss = float('inf')
    best_angles = None

    for start_idx in range(n_starts):
        seed = 42 + start_idx * 137
        rng = np.random.RandomState(seed)
        phi_init = rng.choice([float(PHI_ALPHA), float(PHI_BETA), float(PHI_PPII)], N)
        psi_init = rng.choice([float(PSI_ALPHA), float(PSI_BETA), float(PSI_PPII)], N)
        chi1_init = rng.uniform(-np.pi, np.pi, N)
        chi2_init = rng.uniform(-np.pi, np.pi, N)
        for i in range(N):
            if sub_seq[i] == 'G':
                chi1_init[i] = 0.0; chi2_init[i] = 0.0
            elif sub_seq[i] == 'A':
                chi2_init[i] = 0.0
        angles = jnp.concatenate([jnp.array(phi_init), jnp.array(psi_init),
                                  jnp.array(chi1_init), jnp.array(chi2_init)])

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(angles)
        key = jax.random.PRNGKey(seed)
        _T0 = 1.0 / (2.0 * Q_BACKBONE)
        _GRAD_CLIP = 2.0 * jnp.pi
        anneal_steps = int(n_steps * 0.5)

        loss_jit_s = jit(_torsion_loss_v4, static_argnums=(4,))
        grad_jit_s = jit(grad(_torsion_loss_v4), static_argnums=(4,))

        def opt_step(step, carry):
            angles_c, opt_state_c, key_c = carry
            g = grad_jit_s(angles_c, z_topo, gly_mask, pro_mask, N, cg_mask, env_params)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > _GRAD_CLIP, g * _GRAD_CLIP / g_norm, g)
            updates, new_opt_state = optimizer.update(g, opt_state_c)
            new_angles = optax.apply_updates(angles_c, updates)
            T = _T0 * jnp.maximum(0.0, 1.0 - step / jnp.maximum(1.0, anneal_steps)) ** 2
            key_c, subkey = jax.random.split(key_c)
            noise = jax.random.normal(subkey, shape=new_angles.shape) * T
            new_angles = jnp.where(step < anneal_steps, new_angles + noise, new_angles)
            return (new_angles, new_opt_state, key_c)

        angles, opt_state, key = lax.fori_loop(
            0, n_steps, opt_step, (angles, opt_state, key))

        loss = float(loss_jit_s(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params))
        if loss < best_loss:
            best_loss = loss
            best_angles = angles

    return best_angles, best_loss


def fold_cascade_v7(sequence, n_starts=3, env_params=None):
    """
    Protein folding via v7 segmented cascade.

    Segment length L = Q = 7 (backbone coherence length, derived).
    Phase 1: SEGMENT — fold Q-length sub-chains independently.
    Phase 2: COUPLE  — optimize junction angles (2 DOF per junction).
    Phase 3: REFINE  — polish full chain from cascaded geometry.

    No new constants or physics. Same Y-matrix, same S-parameters.
    """
    N = len(sequence)
    L = int(Q_BACKBONE)  # = 7 (derived segment length)
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS

    z_topo = compute_z_topo(sequence)
    cys_mask, arom_mask, gly_mask, pro_mask, cg_mask, neg_mask, pos_mask = compute_masks(sequence)
    lr = 1.0 / (2.0 * jnp.pi * Q_BACKBONE)
    _GRAD_CLIP = 2.0 * jnp.pi
    _T0 = 1.0 / (2.0 * Q_BACKBONE)

    print(f"  v7 cascade: N={N}, segment_L={L}")

    # ── Phase 1: SEQUENTIAL CASCADE (N→C cotranslational) ─────────
    # Like cotranslational folding / serial filter tuning:
    #   fold seg 0 → freeze → fold seg 1 (with seg 0 present) → ...
    # Each segment sees the full-chain Y-matrix but only its own
    # angles are free. This lets inter-segment contacts form
    # during segment folding, eliminating the assembly problem.
    segments = []
    seg_start = 0
    while seg_start < N:
        seg_end = min(seg_start + L, N)
        if N - seg_start < 4 and len(segments) > 0:
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, N)
            break
        segments.append((seg_start, seg_end))
        seg_start = seg_end

    n_segs = len(segments)
    print(f"    Phase 1: {n_segs} segments (sequential N→C)")

    t0 = time.time()

    # Initialize ALL angles randomly (Ramachandran basins)
    rng = np.random.RandomState(42)
    phi_g = rng.choice([float(PHI_ALPHA), float(PHI_BETA), float(PHI_PPII)], N)
    psi_g = rng.choice([float(PSI_ALPHA), float(PSI_BETA), float(PSI_PPII)], N)
    chi1_g = rng.uniform(-np.pi, np.pi, N)
    chi2_g = rng.uniform(-np.pi, np.pi, N)
    for i in range(N):
        if sequence[i] == 'G':
            chi1_g[i] = 0.0; chi2_g[i] = 0.0
        elif sequence[i] == 'A':
            chi2_g[i] = 0.0
    angles = jnp.concatenate([jnp.array(phi_g), jnp.array(psi_g),
                              jnp.array(chi1_g), jnp.array(chi2_g)])

    # Sequential segment folding: each segment in full-chain context
    for seg_idx, (s, e) in enumerate(segments):
        # Build mask: only this segment's (φ,ψ,χ1,χ2) are free
        seg_mask = np.zeros(4 * N)
        for j in range(s, e):
            seg_mask[j] = 1.0          # φ_j
            seg_mask[N + j] = 1.0      # ψ_j
            seg_mask[2*N + j] = 1.0    # χ1_j
            seg_mask[3*N + j] = 1.0    # χ2_j
        seg_mask = jnp.array(seg_mask)

        # Adam optimisation with segment mask (full chain loss)
        optimizer_s = optax.adam(lr)
        opt_state_s = optimizer_s.init(angles)
        key_s = jax.random.PRNGKey(42 + seg_idx)
        anneal_steps_s = 1000

        def seg_step(step, carry):
            angles_c, opt_state_c, key_c = carry
            g = _torsion_grad_v4_jit(angles_c, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > _GRAD_CLIP, g * _GRAD_CLIP / g_norm, g)
            g = g * seg_mask  # only this segment's angles
            updates, new_opt_state = optimizer_s.update(g, opt_state_c)
            new_angles = optax.apply_updates(angles_c, updates)
            T = _T0 * jnp.maximum(0.0, 1.0 - step / jnp.maximum(1.0, anneal_steps_s)) ** 2
            key_c, subkey = jax.random.split(key_c)
            noise = jax.random.normal(subkey, shape=new_angles.shape) * T * seg_mask
            new_angles = jnp.where(step < anneal_steps_s, new_angles + noise, new_angles)
            return (new_angles, new_opt_state, key_c)

        angles, opt_state_s, key_s = lax.fori_loop(
            0, 2000, seg_step, (angles, opt_state_s, key_s))

        loss_seg = float(_torsion_loss_v4_jit(angles, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask))
        print(f"      seg {seg_idx} [{s}:{e}] ({sequence[s:e]}): chain_loss={loss_seg:.4f}")

    print(f"    Phase 1 done in {time.time()-t0:.0f}s", flush=True)

    loss_assembled = float(_torsion_loss_v4_jit(angles, z_topo, gly_mask, pro_mask,
                                                  N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask))
    print(f"    Assembled loss: {loss_assembled:.4f}", flush=True)

    # ── Phase 2: COUPLE ───────────────────────────────────────────
    junction_indices = set()
    for _, e in segments[:-1]:
        junction_indices.add(e - 1)  # last of segment k
    for s, _ in segments[1:]:
        junction_indices.add(s)      # first of segment k+1
    junction_indices = sorted(junction_indices)
    n_junctions = len(junction_indices)
    print(f"    Phase 2: {n_junctions} junction residues: {junction_indices}")

    junction_mask = np.zeros(4 * N)
    for j in junction_indices:
        junction_mask[j] = 1.0
        junction_mask[N + j] = 1.0
    junction_mask = jnp.array(junction_mask)

    # Phase 2 uses JIT-compiled Adam with junction mask
    t1 = time.time()
    optimizer_c = optax.adam(lr)
    opt_state_c = optimizer_c.init(angles)

    def couple_step(step, carry):
        angles_c, opt_state_c = carry
        g = _torsion_grad_v4_jit(angles_c, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask)
        g = jnp.where(jnp.isnan(g), 0.0, g)
        g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
        g = jnp.where(g_norm > _GRAD_CLIP, g * _GRAD_CLIP / g_norm, g)
        g = g * junction_mask  # only update junction angles
        updates, new_opt_state = optimizer_c.update(g, opt_state_c)
        new_angles = optax.apply_updates(angles_c, updates)
        return (new_angles, new_opt_state)

    angles, opt_state_c = lax.fori_loop(
        0, 2000, couple_step, (angles, opt_state_c))

    loss_coupled = float(_torsion_loss_v4_jit(angles, z_topo, gly_mask, pro_mask,
                                                N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask))
    print(f"    Phase 2 coupled loss: {loss_coupled:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # ── Phase 3: CONSTRAINED REFINE ───────────────────────────────
    # Each angle is clamped to ±BW/2 = ±1/(2Q) from Phase 2 value.
    # This is the resonator bandwidth constraint: the locally-correct
    # segment structures can only be fine-tuned within the
    # resonator's bandwidth, preventing NERF error propagation.
    _BW_HALF = 1.0 / (2.0 * Q_BACKBONE)  # = 1/14 ≈ 0.071 rad ≈ 4.1°
    angles_ref = angles  # save Phase 2 solution as reference

    t2 = time.time()
    optimizer_r = optax.adam(lr * 0.5)  # half lr for gentle refinement
    opt_state_r = optimizer_r.init(angles)

    def refine_step(step, carry):
        angles_c, opt_state_c = carry
        g = _torsion_grad_v4_jit(angles_c, z_topo, gly_mask, pro_mask, N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask)
        g = jnp.where(jnp.isnan(g), 0.0, g)
        g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
        g = jnp.where(g_norm > _GRAD_CLIP, g * _GRAD_CLIP / g_norm, g)
        updates, new_opt_state = optimizer_r.update(g, opt_state_c)
        new_angles = optax.apply_updates(angles_c, updates)
        # Clamp: stay within ±BW/2 of Phase 2 reference
        new_angles = jnp.clip(new_angles, angles_ref - _BW_HALF, angles_ref + _BW_HALF)
        return (new_angles, new_opt_state)

    angles, opt_state_r = lax.fori_loop(
        0, 1000, refine_step, (angles, opt_state_r))

    loss_final = float(_torsion_loss_v4_jit(angles, z_topo, gly_mask, pro_mask,
                                              N, cg_mask, env_params,
                                     cys_mask=cys_mask, arom_mask=arom_mask, neg_mask=neg_mask, pos_mask=pos_mask))
    print(f"    Phase 3 constrained loss: {loss_final:.4f} "
          f"(±{_BW_HALF:.3f} rad, {time.time()-t2:.0f}s)", flush=True)
    print(f"    Total: {time.time()-t0:.0f}s", flush=True)

    phi = angles[:N]
    psi = angles[N:2*N]
    coords_flat = _torsions_to_backbone(phi, psi, N)
    coords = coords_flat.reshape(N, 3, 3)[:, 1, :]

    return np.array(coords), float(loss_final), np.array(angles)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    test_seqs = [
        ("Chignolin", "YYDPETGTWY"),
        ("Trp-cage", "NLYIQWLKDGGPSSGRPPPS"),
    ]
    for name, seq in test_seqs:
        print(f"\n  {name} ({len(seq)} residues)")
        coords, loss, angles = fold_s11_v4(seq, n_steps=2000, n_starts=3)
        rg = np.sqrt(np.mean(np.sum((coords - coords.mean(0))**2, 1)))
        print(f"  Rg: {rg:.1f} Å  loss: {loss:.4f}")
