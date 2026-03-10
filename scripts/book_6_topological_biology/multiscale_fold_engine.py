"""
Multi-Scale Protein Fold Engine — Impedance-Stratified Solver
==============================================================

Architecture (classical EE → protein):
    Bond (1.5 Å)      → Lumped L,C      → protein_bond_constants.py
    Residue (3.8 Å)    → Shunt stub      → Z_TOPO table
    SS Element (10-60Å)→ Filter section   → ABCD cascade per element
    Tertiary (>20 Å)   → PCB network     → K×K Y-matrix

Biology-driven segmentation:
    Pro = helix breaker → open circuit (rigid, no H-bond donor)
    Gly = flexibility   → near-short (minimal sidechain)
    Large ΔZ           → impedance step (connector/via)

Each SS element between discontinuities is a quasi-periodic TL
filter section with its own Thevenin equivalent and structural energy.
"""

import numpy as np
import os

# Import lumped element values (Step 2 of architecture)
from protein_bond_constants import Z_TOPO, Q_BACKBONE, KAPPA_HB, D_HB_DETECT
from ave.core.constants import P_C, ETA_EQ
from ave.axioms.scale_invariant import reflection_coefficient, saturation_factor


# ═══════════════════════════════════════════════════════════════
# STEP 1: SEGMENTATION AT IMPEDANCE DISCONTINUITIES
# ═══════════════════════════════════════════════════════════════
#
# Biology: Pro breaks helices, Gly breaks strands, turns reverse
#          chain direction. These are natural lumped-sum boundaries.
#
# EE:      Segment a TL at impedance discontinuities (connectors,
#          vias, bends). Between them: quasi-periodic filter sections.
#
# Physics: |ΔZ/Z| ≡ |Γ| = reflection coefficient at the boundary.
#          When |Γ| > 1/√(2Q), the boundary is a significant
#          discontinuity (same threshold as turn detection in v3).

# Minimum element size: 3 residues (minimum for a single helical turn
# or 1 H-bonded pair + connector). Below this, it's a junction, not
# a filter section.
MIN_ELEMENT_SIZE = 3

# Reflection threshold for boundary detection.
# From cavity physics: |Γ|² ≥ 1/(2Q) → |Γ| ≥ 1/√(2Q) ≈ 0.267
# This is the SAME threshold used for turn detection in v3.
GAMMA_BOUNDARY = 1.0 / np.sqrt(2.0 * Q_BACKBONE)  # ≈ 0.267


def detect_ss_elements(seq):
    """
    Segment amino acid sequence at impedance discontinuities.
    
    Scans the Z_TOPO profile for:
        1. Proline (Pro/P) — rigid ring, helix breaker (always a boundary)
        2. Glycine (Gly/G) — minimal stub, flexibility point (boundary if isolated)
        3. Large ΔZ — |Γ| = |Z_{i+1} - Z_i| / |Z_{i+1} + Z_i| > threshold
    
    Returns a list of (start, end) index pairs defining each SS element.
    Elements are guaranteed to have ≥ MIN_ELEMENT_SIZE residues.
    
    Args:
        seq: amino acid sequence string (e.g., "NLYIQWLKDGGPSSGRPPPS")
    
    Returns:
        elements: list of (start_idx, end_idx) tuples (end is exclusive)
        boundaries: list of boundary indices (for diagnostics)
        reasons: list of strings explaining each boundary
    """
    N = len(seq)
    if N < MIN_ELEMENT_SIZE:
        return [(0, N)], [], []
    
    z_profile = np.array([abs(Z_TOPO.get(aa, Z_TOPO['A'])) for aa in seq])
    
    # Detect boundaries
    boundary_candidates = []
    reasons = []
    
    for i in range(N):
        # Rule 1: Proline is ALWAYS a boundary (helix breaker)
        if seq[i] == 'P':
            boundary_candidates.append(i)
            reasons.append(f"Pro at {i}")
            continue
        
        # Rule 2: Glycine is a boundary if isolated (not in a Gly-rich region)
        if seq[i] == 'G':
            # Check if there are other Gly within ±2 positions
            neighbors = seq[max(0,i-2):min(N,i+3)]
            n_gly = sum(1 for c in neighbors if c == 'G')
            if n_gly <= 2:  # Isolated or pair → boundary
                boundary_candidates.append(i)
                reasons.append(f"Gly at {i}")
            continue
        
        # Rule 3: Large impedance step between adjacent residues
        if i < N - 1:
            Z_i = z_profile[i]
            Z_next = z_profile[i + 1]
            gamma = abs(reflection_coefficient(Z_i, Z_next))
            if gamma > GAMMA_BOUNDARY:
                boundary_candidates.append(i)
                reasons.append(f"ΔZ at {i}: |Γ|={gamma:.3f}")
    
    # Convert boundaries to elements, enforcing minimum size
    boundaries = []
    filtered_reasons = []
    
    # Sort and deduplicate
    boundary_candidates = sorted(set(boundary_candidates))
    
    # Filter: enforce minimum element size between boundaries
    if not boundary_candidates:
        return [(0, N)], [], []
    
    last_boundary = -MIN_ELEMENT_SIZE  # Virtual boundary before chain start
    for i, b in enumerate(boundary_candidates):
        if b - last_boundary >= MIN_ELEMENT_SIZE:
            boundaries.append(b)
            filtered_reasons.append(reasons[i])
            last_boundary = b
    
    # Build elements from boundaries
    elements = []
    start = 0
    for b in boundaries:
        if b - start >= MIN_ELEMENT_SIZE:
            elements.append((start, b))
            start = b
    # Last element: from last boundary to end
    if N - start >= MIN_ELEMENT_SIZE:
        elements.append((start, N))
    elif elements:
        # Merge short tail into last element
        elements[-1] = (elements[-1][0], N)
    else:
        elements = [(0, N)]
    
    return elements, boundaries, filtered_reasons


# ═══════════════════════════════════════════════════════════════
# STEP 2: PER-ELEMENT STRUCTURAL ENERGY
# ═══════════════════════════════════════════════════════════════
#
# Bond solver: E_struct = Σ √(μ_r · ε_r) × dx/ℓ_node
# Protein:     E_k = mean(seg_Zc_k) within element k
#
# seg_Zc encodes:
#   × √(1+R²)         — sidechain mass (μ enhancement)
#   / √(1+ε_boost)    — H-bond coupling (ε-channel, attractive)
#
# EE analog: input impedance of a filter section.
# High Z_in = mismatched (sterically strained, high energy)
# Low Z_in = well-matched (H-bonded, low energy)

# Backbone bond impedances (from protein_bond_constants.py)
# Z = √(mass_Da / n_electrons) — bond solver pattern
from protein_bond_constants import BACKBONE_BONDS
M_N_CA = BACKBONE_BONDS['N-Ca']['mass_Da']   # 26 Da
N_E_N_CA = BACKBONE_BONDS['N-Ca']['n_electrons']  # 2
M_CA_C = BACKBONE_BONDS['Ca-C']['mass_Da']   # 24 Da
N_E_CA_C = BACKBONE_BONDS['Ca-C']['n_electrons']  # 2
M_C_N = BACKBONE_BONDS['C-N']['mass_Da']     # 26 Da
N_E_C_N = BACKBONE_BONDS['C-N']['n_electrons']  # 3

Z_NCa = np.sqrt(M_N_CA / N_E_N_CA)   # √(26/2) = 3.61
Z_CaC = np.sqrt(M_CA_C / N_E_CA_C)   # √(24/2) = 3.46
Z_CN  = np.sqrt(M_C_N  / N_E_C_N)    # √(26/3) = 2.94
Z_BB_AVG = (Z_NCa + Z_CaC + Z_CN) / 3.0  # ≈ 3.34


def ss_element_energy(seq_element, ca_coords=None):
    """
    Compute structural energy for one SS element.
    
    Bond solver pattern: E = mean(seg_Zc) where seg_Zc encodes
    backbone bond Z + sidechain μ-enhancement + H-bond ε-enhancement.
    
    Args:
        seq_element: amino acid subsequence (e.g., "NLYIQWLK")
        ca_coords: (n, 3) Cα coordinates for inter-residue distances.
                   If None, structural energy is from sequence only
                   (no H-bond or steric terms — pure impedance profile).
    
    Returns:
        E_struct: structural energy (mean seg_Zc)
        seg_Zc: (3n-1,) impedance profile for the element
        Z_th: Thevenin equivalent impedance (geometric mean of seg_Zc)
    """
    n = len(seq_element)
    
    # --- Build seg_Zc_base: 3 bonds per residue (same as v3 L577-598) ---
    z_triplet = np.array([Z_NCa, Z_CaC, Z_CN])
    z_last = np.array([Z_NCa, Z_CaC])
    if n > 1:
        seg_Zc_base = np.concatenate([np.tile(z_triplet, n-1), z_last])
    else:
        seg_Zc_base = z_last.copy()
    
    # --- Sidechain μ-enhancement: Z_eff = Z_bb × √(1+R²) ---
    R_sc = np.array([abs(Z_TOPO.get(aa, Z_TOPO['A'])) for aa in seq_element])
    
    if n > 1:
        R_at_NCa = R_sc[:-1]
        R_at_CaC = R_sc[:-1]
        R_at_CN  = (R_sc[:-1] + R_sc[1:]) / 2.0
        R_triplets = np.stack([R_at_NCa, R_at_CaC, R_at_CN], axis=1).reshape(-1)
        R_last_pair = np.array([R_sc[-1], R_sc[-1]])
        R_all = np.concatenate([R_triplets, R_last_pair])
    else:
        R_all = np.array([R_sc[0], R_sc[0]])
    
    seg_Zc = seg_Zc_base * np.sqrt(1.0 + R_all**2)
    
    # --- H-bond ε-enhancement (if coordinates provided) ---
    if ca_coords is not None and n > 3:
        d0 = 3.8  # Cα-Cα bond length
        dists = np.sqrt(np.sum((ca_coords[:, None, :] - ca_coords[None, :, :])**2, axis=-1) + 1e-12)
        idx = np.arange(n)
        
        # H-bond detection: d < d_hb_detect + d0, |i-j| >= 3
        hb_threshold = D_HB_DETECT + d0  # ≈ 6.0 Å
        hb_mask = (dists < hb_threshold) & (np.abs(idx[:, None] - idx[None, :]) >= 3)
        hb_mask = hb_mask & hb_mask.T  # symmetrise
        hb_upper = np.triu(hb_mask, k=3)
        
        # ε per residue from K/r coupling
        kr_per_pair = hb_upper * KAPPA_HB / (dists + 1e-12)
        eps_per_residue = np.sum(kr_per_pair, axis=1) + np.sum(kr_per_pair, axis=0)
        
        # Map to segments
        if n > 1:
            eps_seg = np.repeat(eps_per_residue[:-1], 3)
            eps_last = np.array([eps_per_residue[-1], eps_per_residue[-1]])
            eps_all = np.concatenate([eps_seg, eps_last])
        else:
            eps_all = np.array([eps_per_residue[0], eps_per_residue[0]])
        
        seg_Zc = seg_Zc / np.sqrt(1.0 + eps_all)
        
        # --- Steric μ-boost ---
        steric_mask = np.abs(idx[:, None] - idx[None, :]) >= 3
        steric_overlap = np.maximum(0.0, d0 - dists)**2 / (d0**2)
        steric_overlap = np.where(steric_mask, steric_overlap, 0.0)
        mu_steric_per_res = np.sum(steric_overlap, axis=1)
        
        if n > 1:
            mu_steric_seg = np.repeat(mu_steric_per_res[:-1], 3)
            mu_steric_last = np.array([mu_steric_per_res[-1], mu_steric_per_res[-1]])
            mu_steric_all = np.concatenate([mu_steric_seg, mu_steric_last])
        else:
            mu_steric_all = np.array([mu_steric_per_res[0], mu_steric_per_res[0]])
        
        KAPPA = 0.5  # critical coupling
        seg_Zc = seg_Zc * np.sqrt(1.0 + KAPPA * mu_steric_all)
    
    # Structural energy (bond solver analog)
    E_struct = np.mean(seg_Zc)
    
    # Thevenin equivalent: geometric mean of profile impedance
    Z_th = np.exp(np.mean(np.log(seg_Zc)))
    
    return E_struct, seg_Zc, Z_th


# ═══════════════════════════════════════════════════════════════
# STEP 3: NETWORK ASSEMBLY (K×K Y-MATRIX)
# ═══════════════════════════════════════════════════════════════
#
# EE analog: PCB network — discrete components (SS elements)
# connected by traces (backbone, H-bonds, hydrophobic contacts).
#
# Each SS element k becomes a node with:
#   Z_th_k = Thevenin equivalent impedance
#   E_k    = structural energy within element
#
# Edge types:
#   1. Sequential backbone: adjacent elements share a peptide bond
#   2. Inter-element H-bonds: H-bonds between element boundaries
#   3. Hydrophobic Z-matching: conjugate impedance matching
#
# From scale_invariant.py (imported at top):
#   Γ = (Z_2 - Z_1) / (Z_2 + Z_1)  — reflection at the junction
#   T = 1 + Γ = 2Z_1 / (Z_1 + Z_2) — transmission through


def build_tertiary_Y(elements, seq, ca_coords):
    """
    Build K×K admittance matrix for inter-element coupling.
    
    Args:
        elements: list of (start, end) tuples from detect_ss_elements()
        seq: full amino acid sequence
        ca_coords: (N, 3) full Cα coordinates
    
    Returns:
        Y: (K, K) admittance matrix
        Z_th: (K,) Thevenin impedance per element
        E_struct: (K,) structural energy per element
        element_info: list of dicts with per-element diagnostics
    """
    K = len(elements)
    N = len(seq)
    d0 = 3.8  # Cα-Cα bond length
    
    # --- Per-element energy and Thevenin equivalent ---
    Z_th = np.zeros(K)
    E_struct = np.zeros(K)
    centroids = np.zeros((K, 3))
    element_info = []
    
    for k, (s, e) in enumerate(elements):
        seg = seq[s:e]
        ca_seg = ca_coords[s:e]
        E_k, seg_Zc_k, Z_th_k = ss_element_energy(seg, ca_seg)
        E_struct[k] = E_k
        Z_th[k] = Z_th_k
        centroids[k] = np.mean(ca_seg, axis=0)
        
        # Hydrophobic character: mean of hydrophobic R values
        R_sc = np.array([abs(Z_TOPO.get(aa, Z_TOPO['A'])) for aa in seg])
        is_hydrophobic = np.array([aa in 'AILMFVWP' for aa in seg])
        f_hydro = np.mean(is_hydrophobic)
        
        element_info.append({
            'seq': seg, 'start': s, 'end': e,
            'E': E_k, 'Z_th': Z_th_k,
            'R_mean': np.mean(R_sc),
            'f_hydro': f_hydro,
        })
    
    # --- Build K×K Y-matrix ---
    Y = np.zeros((K, K), dtype=complex)
    
    # Edge 1: Sequential backbone (adjacent elements share a peptide bond)
    # Y_backbone = 1/Z_avg at the junction (lumped connection)
    for k in range(K - 1):
        Z_avg = 0.5 * (Z_th[k] + Z_th[k+1])
        Y_bb = 1.0 / Z_avg
        Y[k, k] += Y_bb
        Y[k, k+1] -= Y_bb
        Y[k+1, k] -= Y_bb
        Y[k+1, k+1] += Y_bb
    
    # Edge 2: Inter-element H-bonds
    # Look for Cα proximity between elements (not within an element)
    hb_threshold = D_HB_DETECT + d0  # ≈ 6.0 Å
    for i in range(K):
        for j in range(i + 2, K):  # skip adjacent (already connected by backbone)
            # Find closest Cα pair between elements i and j
            s_i, e_i = elements[i]
            s_j, e_j = elements[j]
            ca_i = ca_coords[s_i:e_i]
            ca_j = ca_coords[s_j:e_j]
            
            # Pairwise distances between all Cα in element i and j
            dists_ij = np.sqrt(np.sum(
                (ca_i[:, None, :] - ca_j[None, :, :])**2, axis=-1) + 1e-12)
            d_min = np.min(dists_ij)
            n_contacts = np.sum(dists_ij < hb_threshold)
            
            if n_contacts > 0:
                # Y_hb = KAPPA_HB × n_contacts / Z_match
                Z_match = 0.5 * (Z_th[i] + Z_th[j])
                Y_hb = KAPPA_HB * n_contacts / Z_match
                Y[i, i] += Y_hb
                Y[i, j] -= Y_hb
                Y[j, i] -= Y_hb
                Y[j, j] += Y_hb
    
    # Edge 3: Hydrophobic impedance matching
    # When two elements have similar hydrophobic character,
    # their impedances match (Γ → 0), creating attractive coupling.
    # This is the protein-scale analog of conjugate Z-matching.
    for i in range(K):
        for j in range(i + 1, K):
            # Centroid distance
            d_ij = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
            if d_ij > 20.0:  # Too far for tertiary contact
                continue
            
            # Γ from Thevenin impedances
            Gamma_ij = abs(reflection_coefficient(Z_th[i], Z_th[j]))
            
            # Hydrophobic coupling: stronger when both elements are hydrophobic
            f_h = element_info[i]['f_hydro'] * element_info[j]['f_hydro']
            
            # Y_hydro ∝ (1 - |Γ|²) × f_hydro / d  — attractive when matched
            Y_hydro = f_h * (1.0 - Gamma_ij**2) * (d0 / d_ij)**2
            Y[i, i] += Y_hydro
            Y[i, j] -= Y_hydro
            Y[j, i] -= Y_hydro
            Y[j, j] += Y_hydro
    
    return Y, Z_th, E_struct, element_info


def total_energy(elements, seq, ca_coords):
    """
    Compute total multi-scale energy.
    
    E_total = Σ E_struct_k + E_network
    
    E_struct_k: per-element structural energy (bond solver pattern)
    E_network: inter-element coupling from Y-matrix
    """
    Y, Z_th, E_struct, info = build_tertiary_Y(elements, seq, ca_coords)
    K = len(elements)
    
    # Per-element structural energy (sum)
    E_intra = np.sum(E_struct)
    
    # Network coupling energy from Y-matrix
    # E_network = Σ_{i≠j} |Y_ij| × Z_th_i × Z_th_j / K²
    # This rewards strong coupling (large |Y_ij|) between matched elements
    E_network = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            E_network += abs(Y[i, j]) * Z_th[i] * Z_th[j]
    E_network /= (K * K)
    
    # Total: intra + inter (inter is negative because coupling stabilises)
    E_total = E_intra / K - E_network
    
    return E_total, E_intra, E_network, info


# ═══════════════════════════════════════════════════════════════
# STEP 4: JAX MULTI-SCALE FOLD FUNCTION
# ═══════════════════════════════════════════════════════════════
#
# Reuses from v3:
#   _torsions_to_backbone — NERF backbone placement (JAX-compatible)
#   Adam optimizer + annealing — same optimization loop
#
# New loss:
#   E_total = mean(E_struct_k) + rama_penalty - E_network
#
# Segmentation is PRE-COMPUTED from sequence (fixed during optimization).
# Only torsion angles (φ, ψ) are optimised.

import jax
import jax.numpy as jnp
import optax
import time

# Import v3 backbone placement
from s11_fold_engine_v3_jax import (
    _torsions_to_backbone, _compute_cb_positions,
    compute_gly_mask, compute_pro_mask,
    D_N_CA, D_CA_C, D_C_N,
)

# Backbone atom steric weight (derived: LAMBDA_BOND × d0/r_Ca)
# LAMBDA_BOND = 2.0, d0 = 3.8 Å, r_Ca = 1.7 Å → LAMBDA_STERIC = 4.47
_LAMBDA_BOND = 2.0
_d0 = 3.8
_r_Ca = 1.7
LAMBDA_BB_STERIC = _LAMBDA_BOND * _d0 / _r_Ca  # ≈ 4.47

# JAX versions of backbone bond impedances
_Z_NCa_j = jnp.sqrt(jnp.float64(M_N_CA / N_E_N_CA))
_Z_CaC_j = jnp.sqrt(jnp.float64(M_CA_C / N_E_CA_C))
_Z_CN_j  = jnp.sqrt(jnp.float64(M_C_N / N_E_C_N))


def _multiscale_loss(angles, z_topo_mag, z_topo_complex, gly_mask, pro_mask, N):
    """
    JAX-differentiable multi-scale energy.
    
    Four physical terms (each with EE motivation):
      Term 1: seg_Zc × √(1+R²) — sidechain mass = μ-enhancement (heavier stub → higher Z)
      Term 2: seg_Zc × √(1+κ_steric) — Pauli exclusion = μ-boost (crowding → higher Z)
      Term 3: seg_Zc × √(1-κ_HB) — H-bond mutual inductance = transformer (coupling → lower Z)
      Term 4: mean(exposure × |Γ|²) × S(η) — hydrophobic solvation (complex Γ at water boundary)
    
    + Cα steric penalty (DRC: minimum trace spacing)
    
    CRITICAL: Term 4 uses COMPLEX Z_TOPO and complex Z_water (Debye model).
    The hydrophilic/hydrophobic distinction emerges from the REACTIVE component:
      Hydrophobic (X=0): can’t match water’s reactance → |Γ|² HIGH → bury
      Charged (X=±R/Q): conjugate-matches water → |Γ|² LOW → expose
      Polar (X=±R/2Q): partial match → |Γ|² MEDIUM
    """
    phi = angles[:N]
    psi = angles[N:]
    
    # Generate backbone coordinates via NERF
    coords_flat = _torsions_to_backbone(phi, psi, N)
    bb = coords_flat.reshape(N, 3, 3)
    atom_N  = bb[:, 0, :]
    atom_Ca = bb[:, 1, :]
    atom_C  = bb[:, 2, :]
    coords = atom_Ca  # Cα positions (N, 3)
    
    # Pairwise Cα distances
    dists = jnp.sqrt(jnp.sum(
        (coords[:, None, :] - coords[None, :, :])**2, axis=-1) + 1e-12)
    idx = jnp.arange(N)
    
    # === Build full seg_Zc (3N-1 segments) ===
    # PHYSICS: each backbone bond is a lumped L,C element.
    #   Z = √(L/C) = √(mass_Da / n_electrons)
    #   This is the same √(μ/ε) from the bond solver.
    z_triplet = jnp.array([_Z_NCa_j, _Z_CaC_j, _Z_CN_j])
    z_last = jnp.array([_Z_NCa_j, _Z_CaC_j])
    seg_Zc_base = jnp.concatenate([jnp.tile(z_triplet, N-1), z_last])
    
    # --- Term 1: Sidechain μ-enhancement ---
    # PHYSICS: sidechain mass adds to backbone inductance.
    #   L_total = L_bb + L_sc → μ_total = μ_bb(1 + R²)
    #   Z_eff = Z_bb × √(1 + R²)
    # EE: heavier stub on a TL raises Z (more mass = more inductance).
    # Gly (R=0.30): √(1+0.09) = 1.04 (minimal stub)
    # Trp (R=0.89): √(1+0.80) = 1.34 (massive indole ring)
    R_sc = z_topo_mag
    R_at_NCa = R_sc[:-1]
    R_at_CaC = R_sc[:-1]
    R_at_CN  = (R_sc[:-1] + R_sc[1:]) / 2.0
    R_triplets = jnp.stack([R_at_NCa, R_at_CaC, R_at_CN], axis=1).reshape(-1)
    R_last_pair = jnp.array([R_sc[-1], R_sc[-1]])
    R_all = jnp.concatenate([R_triplets, R_last_pair])
    seg_Zc = seg_Zc_base * jnp.sqrt(1.0 + R_all**2)
    
    # --- Term 2: Steric μ-boost ---
    # PHYSICS: Pauli exclusion = additional inductance from overlapping
    #   electron clouds. Two atoms closer than 2×r_Ca create a
    #   repulsive M-field (same as nuclear defect μ-boost in bond solver).
    # EE: crosstalk between nearby TL segments raises effective Z.
    # EXCEPTION: H-bonding pairs are EXCLUDED from steric — transformer
    #   windings are intentionally close (DRC doesn't apply to coils).
    d0 = 3.8
    seq_sep = jnp.abs(idx[:, None] - idx[None, :])
    steric_mask = seq_sep >= 3
    
    # H-bond contact matrix (for exclusion from steric)
    hb_threshold = D_HB_DETECT + d0  # ≈ 6.0 Å
    hb_contacts = jnp.where(
        (dists < hb_threshold) & (seq_sep >= 3), 1.0, 0.0)
    
    # Steric overlap: only where NOT H-bonding
    non_hb_steric = steric_mask & (hb_contacts < 0.5)
    steric_overlap = jnp.maximum(0.0, d0 - dists)**2 / (d0**2)
    steric_overlap = jnp.where(non_hb_steric, steric_overlap, 0.0)
    mu_steric = jnp.sum(steric_overlap, axis=1)
    
    KAPPA_STERIC = 0.5  # critical coupling (κ = ½)
    mu_seg = jnp.repeat(mu_steric[:-1], 3, total_repeat_length=3*(N-1))
    mu_last = jnp.array([mu_steric[-1], mu_steric[-1]])
    mu_all = jnp.concatenate([mu_seg, mu_last])
    seg_Zc = seg_Zc * jnp.sqrt(1.0 + KAPPA_STERIC * mu_all)
    
    # === Term 3: Transformer coupling (H-bond mutual inductance) ===
    # PHYSICS: H-bonds create MUTUAL INDUCTANCE between backbone turns.
    #   Extended chain: each segment is an isolated inductor (L, I²).
    #   Helix: H-bonds (i→i+4) couple turns like a transformer.
    #     L_eff = L_self - M  where M = κ × L
    #     E_coupled = E_uncoupled × (1 - κ_total)
    #     κ_total = n_HB × κ_per_HB
    #
    # EE: a solenoid with shorted turns has LOWER stored energy
    #   than one with open turns, because mutual inductance provides
    #   a return path that cancels part of the flux.
    #
    # DERIVATION: KAPPA_HB = 1/(2Q) = 1/14 ≈ 0.071
    #   For 5 H-bonds at a residue: κ = 5 × 0.071 = 0.355
    #   √(1 - 0.355) = 0.803 → 20% energy reduction.
    #
    # Vectorized per-residue (no loops, JIT-friendly).
    n_hb_per_res = jnp.sum(hb_contacts, axis=1)
    
    kappa_per_res = jnp.clip(n_hb_per_res * KAPPA_HB, 0.0, 0.95)
    transformer_per_res = jnp.sqrt(1.0 - kappa_per_res)
    
    tf_at_NCa = transformer_per_res[:-1]
    tf_at_CaC = transformer_per_res[:-1]
    tf_at_CN  = (transformer_per_res[:-1] + transformer_per_res[1:]) / 2.0
    tf_triplets = jnp.stack([tf_at_NCa, tf_at_CaC, tf_at_CN], axis=1).reshape(-1)
    tf_last = jnp.array([transformer_per_res[-1], transformer_per_res[-1]])
    tf_all = jnp.concatenate([tf_triplets, tf_last])
    
    seg_Zc = seg_Zc * tf_all
    
    # === Term 4: Water terminal load (standing wave model) ===
    # PHYSICS: Each exposed segment is terminated by water (Z_water).
    #   The backbone-water mismatch creates standing waves that increase
    #   stored energy: seg_Zc_loaded = seg_Zc × (1 + exposure × |Γ|²)
    #
    # This is the SAME mechanism as the S-param engine's solvent ground load
    #   (Y_solvent = exposure / Z_water), but expressed as impedance modulation
    #   rather than admittance matrix entry.
    #
    # KEY: Solvation is MULTIPLICATIVE on structural energy, not additive.
    #   ~14% modulation for hydrophobic vs ~7% for charged = double the
    #   gradient to bury hydrophobics.
    #
    # COMPLEX Γ: Hydrophilic/hydrophobic from REACTANCE (see docstring).
    #   Z_TOPO_complex carries the reactive component:
    #     Hydrophobic (X=0): can't match water → |Γ|² = 0.14 → 14% Z increase
    #     Charged (X=±R/Q): conjugate-matches water → |Γ|² = 0.07 → 7% Z increase
    #     Polar (X=±R/2Q): partial match → |Γ|² = 0.06 → 6% Z increase
    
    # Complex Z_water from Debye relaxation (S-param engine)
    EPS_S = 80.0; EPS_INF = 1.77; TAU = 8.3e-12
    omega = 2.0 * jnp.pi * 23e12
    eps_water = EPS_INF + (EPS_S - EPS_INF) / (1.0 + 1j * omega * TAU)
    Z_water_complex = jnp.sqrt(eps_water)
    
    # Complex Γ per residue (Axiom 1)
    Gamma = (z_topo_complex - Z_water_complex) / (z_topo_complex + Z_water_complex)
    Gamma_sq = jnp.real(Gamma * jnp.conj(Gamma))  # |Γ|²
    
    # Burial detection (same model as S-param engine)
    R_burial = 2.0 * d0  # 7.6 Å
    burial_sigmoid = jax.nn.sigmoid(4.0 * (R_burial - dists))
    burial_contacts = jnp.sum(
        jnp.where(seq_sep >= 3, burial_sigmoid, 0.0), axis=1)
    n_max = jnp.minimum((R_burial / d0)**3, N / 3.0)
    n_max = jnp.maximum(n_max, 4.0)
    exposure_raw = jnp.clip(1.0 - burial_contacts / n_max, 0.0, 1.0)
    
    # Packing saturation floor (Axiom 4, same as S-param engine)
    # At high density, surface residues can't be fully buried.
    r_Ca = 1.7
    com = jnp.mean(coords, axis=0)
    Rg_sq = jnp.mean(jnp.sum((coords - com)**2, axis=1))
    R_eff = jnp.sqrt(5.0/3.0 * Rg_sq + 1e-12)
    eta = N * r_Ca**3 / (R_eff**3 + 1e-12)
    eta_ratio = jnp.clip(eta / P_C, 0.0, 0.999)
    sat_global = jnp.sqrt(1.0 - eta_ratio**2)
    exposure_floor = 1.0 - sat_global
    exposure = jnp.maximum(exposure_raw, exposure_floor)
    
    # Standing wave factor: map per-residue exposure×|Γ|² to segments
    sw_per_res = exposure * Gamma_sq  # per-residue standing wave
    sw_at_NCa = sw_per_res[:-1]
    sw_at_CaC = sw_per_res[:-1]
    sw_at_CN  = (sw_per_res[:-1] + sw_per_res[1:]) / 2.0
    sw_triplets = jnp.stack([sw_at_NCa, sw_at_CaC, sw_at_CN], axis=1).reshape(-1)
    sw_last = jnp.array([sw_per_res[-1], sw_per_res[-1]])
    sw_all = jnp.concatenate([sw_triplets, sw_last])
    
    # Apply standing wave load: exposed segments store more energy
    seg_Zc = seg_Zc * (1.0 + sw_all)
    
    # === ABCD CASCADE S₁₁ (wave propagation through modulated seg_Zc) ===
    # PHYSICS: The S₁₁ cascade propagates a wave through the impedance
    #   landscape. PERIODIC Z patterns (helices) create passbands → low S₁₁.
    #   RANDOM Z patterns (coil) create stopbands → high S₁₁.
    #   The optimizer minimizes S₁₁ → drives helix/sheet formation.
    #
    # EE: A cascade of TL segments with smoothly varying Z is a bandpass
    #   filter. The helix (3.6 res/turn) creates a periodic Z lattice
    #   at the resonance pitch. Constructive interference at this pitch
    #   amplifies small Z differences into large S₁₁ changes.
    #
    # All 4 multiplicative terms (μ, steric, transformer, water load)
    #   are already applied to seg_Zc. The cascade amplifies their effects
    #   through wave interference.
    #
    # Propagation constant γ = α + jβ per segment:
    #   α = |d - d₀| / d₀ : bond strain → evanescent loss
    #   β = ω × d / d₀    : phase accumulation
    # At correct bond length (d=d₀): α=0, pure propagation → min S₁₁
    
    # Bond distances for propagation constant
    seg_d0 = jnp.array([D_N_CA, D_CA_C, D_C_N])  # equilibrium bond lengths
    d_NCa = jnp.sqrt(jnp.sum((atom_N - atom_Ca)**2, axis=-1) + 1e-12)
    d_CaC = jnp.sqrt(jnp.sum((atom_Ca - atom_C)**2, axis=-1) + 1e-12)
    d_CN = jnp.sqrt(jnp.sum((atom_C[:-1] - atom_N[1:])**2, axis=-1) + 1e-12)
    
    # Build per-segment distance and equilibrium arrays
    d_triplets = jnp.stack([d_NCa[:-1], d_CaC[:-1], d_CN], axis=1).reshape(-1)
    d_last = jnp.array([d_NCa[-1], d_CaC[-1]])
    seg_d = jnp.concatenate([d_triplets, d_last])
    
    d0_triplets = jnp.tile(seg_d0, N-1)
    d0_last = jnp.array([D_N_CA, D_CA_C])
    seg_d0_all = jnp.concatenate([d0_triplets, d0_last])
    
    # Complex propagation: γ = α + jβ
    omega_cascade = 2.0 * jnp.pi * 1.0  # ω=2π (unit frequency, f=1)
    alpha_arr = jnp.abs(seg_d - seg_d0_all) / seg_d0_all
    beta_arr = omega_cascade * seg_d / seg_d0_all
    gamma_arr = alpha_arr + 1j * beta_arr
    
    cosh_arr = jnp.cosh(gamma_arr)
    sinh_arr = jnp.sinh(gamma_arr)
    
    n_segs = 3 * N - 1
    n_junctions = 3 * N - 2
    
    # Junction shunt admittances (same architecture as S-param engine)
    # Sidechain stubs at Cα junctions: Y_stub = 1/Z_TOPO (each sidechain
    # is a parasitic shunt to ground). This makes S₁₁ sequence-dependent:
    #   Large sidechain (Trp, Z=0.89) → Y = 1.12 (small leakage)
    #   Small sidechain (Gly, Z=0.30) → Y = 3.30 (large leakage)
    # Plus solvent load at exposed Cα: Y_solvent = exposure/|Z_water|
    seg_Y = jnp.zeros(n_junctions)
    ca_junctions = jnp.arange(N) * 3  # Cα at junctions 0, 3, 6, ...
    ca_junctions = jnp.clip(ca_junctions, 0, n_junctions - 1)
    
    # Sidechain admittance: Y = 1/|Z_TOPO|
    Y_sidechain = 1.0 / (z_topo_mag + 1e-12)
    seg_Y = seg_Y.at[ca_junctions].add(Y_sidechain)
    
    # Solvent admittance: Y = exposure/|Z_water| (same as S-param engine)
    Z_water_mag = jnp.abs(Z_water_complex)
    Y_solvent = exposure / (Z_water_mag + 1e-12)
    seg_Y = seg_Y.at[ca_junctions].add(Y_solvent)
    
    # ABCD cascade via lax.fori_loop (with junction shunt admittances)
    init_state = jnp.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
    
    def cascade_step(i, state):
        A, B, C, D = state[0], state[1], state[2], state[3]
        ch = cosh_arr[i]
        sh = sinh_arr[i]
        Zc = seg_Zc[i] + 1e-12
        # TL segment ABCD multiplication
        A_n = A * ch + B * (sh / Zc)
        B_n = A * (Zc * sh) + B * ch
        C_n = C * ch + D * (sh / Zc)
        D_n = C * (Zc * sh) + D * ch
        # Junction shunt admittance (at junction AFTER segment i)
        Y = jnp.where(i < n_junctions,
                      seg_Y[jnp.clip(i, 0, n_junctions - 1)], 0.0)
        C_n = C_n + Y * A_n
        D_n = D_n + Y * B_n
        return jnp.array([A_n, B_n, C_n, D_n])
    
    abcd = jax.lax.fori_loop(0, n_segs, cascade_step, init_state)
    A_total, B_total, C_total, D_total = abcd[0], abcd[1], abcd[2], abcd[3]
    
    # S₁₁ extraction: Γ = (B/Z₀ - C×Z₀) / (A + B/Z₀ + C×Z₀ + D)
    Z0 = jnp.mean(seg_Zc) + 1e-12  # reference impedance = mean backbone Z
    gamma_s11 = (B_total/Z0 - C_total*Z0) / (A_total + B_total/Z0 + C_total*Z0 + D_total + 1e-12)
    E_cascade = jnp.real(gamma_s11 * jnp.conj(gamma_s11))  # |S₁₁|²
    
    # Total structural energy: S₁₁ from cascade (captures resonance)
    E_struct = E_cascade
    
    # === Cα steric penalty ===
    # PHYSICS: hard-sphere exclusion for NON-H-bonded contacts.
    # EE: DRC — minimum trace spacing (but not for transformer windings).
    ca_steric = jnp.where(non_hb_steric, jnp.maximum(0.0, 2*1.7 - dists)**2, 0.0)
    steric_penalty = LAMBDA_BB_STERIC * jnp.sum(ca_steric) / (N * N)
    
    return E_struct + steric_penalty


def fold_multiscale(seq, n_steps=None, lr=2e-3, n_starts=None, anneal=True):
    """
    Multi-scale protein fold solver.
    
    Architecture:
        1. Pre-segment sequence at impedance discontinuities (Step 1)
        2. Optimize (φ,ψ,χ₁) to minimise multi-scale energy (Steps 2-3)
        3. Per-element structural energy + backbone steric
    
    Computational effort scaling (derived, not tuned):
    
        The backbone is a damped resonant chain with quality factor Q = 7
        (amide-V resonance, f₀/Δf = 23/3.3 THz, Axiom 1).
        
        n_steps = D × Q × N × k_adam
            D = DOF per residue (3: φ, ψ, χ₁)
            Q = 7 (backbone quality factor)
            N = chain length
            k_adam ≈ 20 (Adam convergence cycles per mode)
            Physical meaning: the optimizer settles D×N coupled oscillators,
            each requiring Q damping cycles, with k_adam gradient steps per cycle.
        
        n_starts = max(2, ⌈3N / (2πQ)⌉)
            The backbone has correlation length L_Q = 2πQ ≈ 44 residues.
            Independent correlation volumes: N_vol = N / L_Q.
            Each volume has ~3 topological basins (α, β, coil).
            n_starts samples the independent basins.
        
        Cross-check (N=20, D=4):
            n_steps = 4 × 7 × 20 × 20 = 11,200
            n_starts = max(2, ⌈240/44⌉) = 2 ✓
        
        Cross-check (N=100, D=4):
            n_steps = 4 × 7 × 100 × 20 = 56,000
            n_starts = max(2, ⌈300/44⌉) = 7
    
    Args:
        seq: amino acid sequence string
        n_steps: optimization steps per start (None = auto-derive from N)
        lr: learning rate for Adam
        n_starts: number of random restarts (None = auto-derive from N)
        anneal: whether to add noise annealing in first 50% of steps
    
    Returns:
        ca: (N, 3) Cα coordinates
        elements: list of (start, end) element tuples
        trace: [best_loss]
        bb: (N, 3, 3) full backbone [N, Cα, C]
    """
    N = len(seq)
    
    # ── Derived computational effort (no magic numbers) ──
    from s11_fold_engine_v3_jax import Q_BACKBONE
    D_DOF = 4          # DOF per residue: φ, ψ, χ₁, χ₂
    K_ADAM = 20         # Adam convergence cycles per mode
    Q = Q_BACKBONE      # = 7.0
    
    if n_steps is None:
        n_steps = int(D_DOF * Q * N * K_ADAM)
    if n_starts is None:
        import math
        L_Q = 2.0 * math.pi * Q   # correlation length ≈ 44 residues
        n_starts = max(2, math.ceil(3.0 * N / L_Q))
    
    # Step 1: pre-segment (multi-scale initialization strategy)
    elements, bounds, reasons = detect_ss_elements(seq)
    K = len(elements)
    element_starts = [s for s, e in elements]
    element_ends = [e for s, e in elements]
    
    # Pre-compute sequence-dependent arrays for v3 S₁₁ loss
    # Import v3 engine functions (the correct EE nodal Y-matrix approach)
    from s11_fold_engine_v3_jax import (
        _torsion_loss as _s11_torsion_loss,
        compute_z_topo as compute_z_topo_v3,
        compute_cys_mask as compute_cys_mask_v3,
        compute_aromatic_mask as compute_aromatic_mask_v3,
        compute_cg_mask as compute_cg_mask_v3,
    )
    z_topo = compute_z_topo_v3(seq)
    cys_mask = compute_cys_mask_v3(seq)
    arom_mask = compute_aromatic_mask_v3(seq)
    gly_mask = compute_gly_mask(seq)
    pro_mask = compute_pro_mask(seq)
    cg_mask = compute_cg_mask_v3(seq)
    
    print(f"  Multi-scale fold: N={N}, K={K} elements, {n_starts}-start, "
          f"{n_steps} steps (D={D_DOF}×Q={Q:.0f}×N={N}×k={K_ADAM})", flush=True)
    print(f"  Elements: {elements}", flush=True)
    
    # Loss = v3 S₁₁ from nodal Y-matrix (full conformation-dependent network)
    # This captures ALL the physics: backbone cascade, sidechain stubs, H-bonds,
    # hydrophobic coupling, chirality, solvent loading, packing saturation.
    # The multi-scale engine provides segmentation and initialization;
    # the v3 engine provides the loss function (the correct EE approach).
    loss_fn = lambda a: _s11_torsion_loss(
        a, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N, cg_mask)
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    
    best_loss = float('inf')
    best_angles = None
    
    for start_idx in range(n_starts):
        seed = 42 + start_idx * 137
        np.random.seed(seed)
        phi_init = np.random.uniform(-np.pi, np.pi, N)
        psi_init = np.random.uniform(-np.pi, np.pi, N)
        # χ₁ sidechain rotamer DOF (Axiom 2: tetrahedral sp³ minima)
        chi1_init = np.random.choice(
            [np.radians(-60), np.radians(60), np.radians(180)], N)
        # χ₂ Cγ branching point DOF (same sp³ minima)
        chi2_init = np.random.choice(
            [np.radians(-60), np.radians(60), np.radians(180)], N)
        for i in range(N):
            if seq[i] == 'G':
                chi1_init[i] = 0.0
                chi2_init[i] = 0.0
            elif seq[i] == 'A':
                chi2_init[i] = 0.0
        angles = jnp.concatenate([jnp.array(phi_init), jnp.array(psi_init),
                                  jnp.array(chi1_init), jnp.array(chi2_init)])
        
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(angles)
        key = jax.random.PRNGKey(seed)
        
        t0 = time.time()
        if start_idx == 0:
            _ = loss_jit(angles)
            _ = grad_jit(angles)
            print(f"    JIT compiled in {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()
        
        # ── Compiled optimization loop (jax.lax.fori_loop) ──
        # Entire optimization compiled into single XLA program.
        # Eliminates ~1ms Python dispatch overhead per step.
        anneal_steps = int(n_steps * 0.5) if anneal else 0
        
        def opt_step(step, carry):
            angles_c, opt_state_c, key_c = carry
            g = grad_jit(angles_c)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
            g = jnp.where(g_norm > 10.0, g * 10.0 / g_norm, g)
            updates, new_opt_state = optimizer.update(g, opt_state_c)
            new_angles = optax.apply_updates(angles_c, updates)
            # Annealing: branchless (jnp.where avoids Python if)
            T = 0.05 * jnp.maximum(0.0, 1.0 - step / jnp.maximum(1.0, anneal_steps)) ** 2
            key_c, subkey = jax.random.split(key_c)
            noise = jax.random.normal(subkey, shape=new_angles.shape) * T
            new_angles = jnp.where(step < anneal_steps, new_angles + noise, new_angles)
            return (new_angles, new_opt_state, key_c)
        
        angles, opt_state, key = jax.lax.fori_loop(
            0, n_steps, opt_step, (angles, opt_state, key))
        
        loss = float(loss_jit(angles))
        dt = time.time() - t0
        print(f"    start {start_idx}: loss={loss:.4f} ({dt:.0f}s)", flush=True)
        
        if loss < best_loss:
            best_loss = loss
            best_angles = angles
    
    print(f"    best loss = {best_loss:.6f}", flush=True)
    
    # Build final coordinates
    phi_final = best_angles[:N]
    psi_final = best_angles[N:2*N]
    coords_flat = _torsions_to_backbone(phi_final, psi_final, N)
    bb_final = np.array(coords_flat.reshape(N, 3, 3))
    ca_final = bb_final[:, 1, :]
    
    return ca_final, elements, [best_loss], bb_final

if __name__ == "__main__":
    # -------------------------------------------------------
    # STEP 1 TEST: Segmentation
    # -------------------------------------------------------
    print("=" * 60)
    print("STEP 1 TEST: Segmentation at Impedance Discontinuities")
    print("=" * 60)
    
    trp_seq = "NLYIQWLKDGGPSSGRPPPS"
    elements, bounds, reasons = detect_ss_elements(trp_seq)
    
    print(f"\nTrp-cage: {trp_seq}")
    z_vals = [f'{abs(Z_TOPO.get(aa, Z_TOPO["A"])):.2f}' for aa in trp_seq]
    print(f"  Z profile: {z_vals}")
    print(f"  Boundaries: {bounds}")
    print(f"  Reasons: {reasons}")
    print(f"  Elements ({len(elements)}):")
    for i, (s, e) in enumerate(elements):
        seg = trp_seq[s:e]
        z_seg = [abs(Z_TOPO.get(aa, Z_TOPO['A'])) for aa in seg]
        z_mean = np.mean(z_seg)
        z_std = np.std(z_seg)
        print(f"    [{s:2d}:{e:2d}] {seg:20s}  <|Z|>={z_mean:.3f} ± {z_std:.3f}")
    
    print(f"\n  Known SS: helix [2:9], 310-helix [11:14], polyPro [17:19]")
    
    # -------------------------------------------------------
    # STEP 2 TEST: Per-Element Structural Energy
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 TEST: Per-Element Structural Energy")
    print("=" * 60)
    
    # Test A: Sequence-only energy (no coordinates)
    print("\n--- A) Sequence-only energy (no coordinates) ---")
    for i, (s, e) in enumerate(elements):
        seg = trp_seq[s:e]
        E, seg_Zc, Z_th = ss_element_energy(seg)
        print(f"  [{s:2d}:{e:2d}] {seg:12s}  E={E:.4f}  Z_th={Z_th:.4f}  n_segs={len(seg_Zc)}")
    
    # Test B: With native PDB coordinates
    aa_map = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
              'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
              'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    
    pdb_path = '/tmp/1L2Y.pdb'
    if os.path.exists(pdb_path):
        print(f"\n--- B) With native Cα coordinates ({pdb_path}) ---")
        ca = []
        seq_from_pdb = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in (' ', 'A'):
                    ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    seq_from_pdb.append(aa_map.get(line[17:20].strip(), 'A'))
                    if len(ca) >= 20:
                        break
        ca = np.array(ca)
        seq_pdb = ''.join(seq_from_pdb)
        
        print(f"  Sequence: {seq_pdb}")
        elements_pdb, _, _ = detect_ss_elements(seq_pdb)
        
        print(f"\n  {'Element':12s} {'Seq':15s} {'E_struct':>8s} {'Z_th':>8s} {'n_HB':>5s} {'Physics':20s}")
        print(f"  {'-'*12} {'-'*15} {'-'*8} {'-'*8} {'-'*5} {'-'*20}")
        
        for i, (s, e) in enumerate(elements_pdb):
            seg = seq_pdb[s:e]
            ca_seg = ca[s:e]
            E, seg_Zc, Z_th = ss_element_energy(seg, ca_seg)
            
            # Count H-bonds in this element
            n = len(seg)
            n_hb = 0
            if n > 3:
                d0 = 3.8
                dists = np.sqrt(np.sum((ca_seg[:, None, :] - ca_seg[None, :, :])**2, axis=-1) + 1e-12)
                idx_arr = np.arange(n)
                hb_threshold = D_HB_DETECT + d0
                hb_mask = (dists < hb_threshold) & (np.abs(idx_arr[:, None] - idx_arr[None, :]) >= 3)
                n_hb = np.sum(np.triu(hb_mask, k=3))
            
            # Classify
            if n_hb > 0:
                physics = f"ε↑ (H-bonded, {n_hb} HB)"
            else:
                physics = "Z_bb only"
            
            print(f"  [{s:2d}:{e:2d}]     {seg:15s} {E:8.4f} {Z_th:8.4f} {n_hb:5d} {physics}")
        
        # Compare: helix element vs coil
        print(f"\n  Key comparison:")
        E_helix, _, Z_helix = ss_element_energy(seq_pdb[0:8], ca[0:8])
        E_coil, _, Z_coil = ss_element_energy(seq_pdb[8:11], ca[8:11])
        print(f"    Helix [0:8]:  E={E_helix:.4f}, Z_th={Z_helix:.4f}")
        print(f"    Coil  [8:11]: E={E_coil:.4f}, Z_th={Z_coil:.4f}")
        dE = E_helix - E_coil
        print(f"    ΔE = {dE:+.4f}  {'✅ Helix LOWER (H-bonds stabilise)' if dE < 0 else '⚠️ Helix HIGHER'}")
    else:
        print(f"\n  PDB file not found: {pdb_path}")
        print(f"  Download: wget https://files.rcsb.org/download/1L2Y.pdb -O /tmp/1L2Y.pdb")

