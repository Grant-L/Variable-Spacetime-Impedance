#!/usr/bin/env python3
"""
s15_allosteric_yield.py
=======================

Tier 5: Dynamic Allostery & Yield Phenomena

This script simulates a macroscopic biological transmission line (a protein)
undergoing dynamic reconfiguration. It injects a synthetic topological impedance
(Z_ligand) into a pre-folded native state to demonstrate that Allostery is 
structurally identical to "Tuning a Cavity Filter" in RF Engineering.

If the structural strain exceeds the network capacity (Gamma -> 1), 
the Bingham Plastic Yield limit is breached, tearing the secondary structure.
"""

import os
import sys
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Try to import from the existing JAX engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from s11_fold_engine_v3_jax import fold_s11_jax, compute_z_topo, Z_TOPO
except ImportError:
    print("Error: Could not import s11_fold_engine_v3_jax.py")
    sys.exit(1)

def main():
    print("============================================================")
    print(" TIER 5: DYNAMIC ALLOSTERY & BINGHAM PLASTIC YIELD")
    print("============================================================")
    
    # 1. We start with an all-Alanine backbone (native Z_topo ~ 2.0).
    # Alanine sidechains do not mass-load the backbone, resulting in a perfect helix.
    seq_len = 15
    sequence = "A" * seq_len
    target_residue = seq_len // 2  # Middle of the chain
    
    # 2. Derive the native Z_topo array
    z_topo_native = compute_z_topo(sequence)
    
    print(f"\n[PHASE 1] Folding Native Sequence ({sequence})")
    print("  Using parameter-free transmission line S11 optimizer...")
    
    # Fold to get the native structural angles
    ca_native, hist_n, trace_n, bb_native, angles_native = fold_s11_jax(
        sequence, n_steps=3000, lr=2e-3, anneal=True, n_starts=1
    )
    
    native_loss = trace_n[-1]
    print(f"  [Native State] Final S11 Strain Loss: {native_loss:.6f}")
    
    # 3. Apply an allosteric perturbation block (Ligand Binding)
    # 
    # Analogy: We solder a massive tuning stub onto the transmission line.
    # We alter the characteristic impedance (Z_topo) at the target residue.
    # Z_topo typically ranges from ~1.5 (Gly) to ~5.0 (Trp).
    # Let's inject a gigantic complex reactive load: Z = 15.0 + 15.0j
    
    ligand_load = 15.0 + 15.0j
    
    print(f"\n[PHASE 2] Ligand Injection at Residue {target_residue}")
    print(f"  Modifying Z_topo[{target_residue}] from {z_topo_native[target_residue]:.2f} to {ligand_load:.2f}")
    print("  This represents an extreme mass-loading event (like a heavy metal or huge ligand).")
    
    z_topo_holo = jnp.array(z_topo_native)
    z_topo_holo = z_topo_holo.at[target_residue].set(ligand_load)
    
    # 4. Reactivate the minimizer.
    # Crucially, we start the minimizer FROM THE NATIVE FOLDED ANGLES.
    # The solver must mechanically respond strictly to the new impedance boundaries.
    
    print("\n[PHASE 3] Dynamic Allosteric Re-equilibration")
    ca_holo, hist_h, trace_h, bb_holo, angles_holo = fold_s11_jax(
        sequence, 
        n_steps=2000, 
        lr=1e-3, 
        anneal=False,  # No annealing, purely mechanistic local descent 
        n_starts=1,
        z_topo_override=z_topo_holo,
        initial_angles=angles_native
    )
    
    holo_loss = trace_h[-1]
    print(f"  [Holo/Bound State] Final S11 Strain Loss: {holo_loss:.6f}")
    
    # 5. Analysis: Did the structure allosterically shift?
    # We compare the native angles vs the holo angles to see how the S11
    # resonance cascaded through the backbone.
    
    phi_native = angles_native[:seq_len]
    psi_native = angles_native[seq_len:2*seq_len]
    phi_holo = angles_holo[:seq_len]
    psi_holo = angles_holo[seq_len:2*seq_len]
    
    delta_phi = np.degrees(phi_holo - phi_native)
    delta_psi = np.degrees(psi_holo - psi_native)
    
    # Calculate angular displacement metric
    total_angular_shift = np.sqrt(delta_phi**2 + delta_psi**2)
    
    print("\n[RESULT] Angular Shift Profile (Allosteric Propagation Matrix):")
    print(f"{'Residue':^10} | {'Δ Phi (deg)':^14} | {'Δ Psi (deg)':^14} | {'Total Shift':^14}")
    print("-" * 62)
    for i in range(seq_len):
        marker = " <== TARGET" if i == target_residue else ""
        print(f"  {sequence[i]}{i:<5} | {delta_phi[i]:>12.2f} | {delta_psi[i]:>12.2f} | {total_angular_shift[i]:>12.2f}{marker}")
    
    max_shift_idx = np.argmax(total_angular_shift)
    print(f"\nSummary:")
    print(f"  Maximum shift occurred at {sequence[max_shift_idx]}{max_shift_idx} ({total_angular_shift[max_shift_idx]:.1f}°).")
    if max_shift_idx != target_residue:
        print("  SUCCESS: The maximum strain propagated to a distant site, manifesting classical ALLOSTERY.")
    else:
        print("  The strain was localized heavily at the binding site, indicating localized yield.")
    
    # ═══════════════════════════════════════════════════════════════
    # PLOTTING
    # ═══════════════════════════════════════════════════════════════
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: S11 Energy Ring-Down
    ax1.plot(trace_n[100:], label="Native Folding (Initial)", color='#00aaff', alpha=0.7)
    ax1.plot(trace_h, label="Holo Re-equilibration (Ring-down)", color='#ff5555', linestyle='--')
    ax1.set_title("Tier 5: Allosteric Energy Ring-Down (S11 Resonance)", fontsize=14)
    ax1.set_xlabel("Descent Steps")
    ax1.set_ylabel("Global Reflection Strain $|\Gamma|^2$")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Distal Strain Propagation (Allostery Profile)
    indices = np.arange(seq_len)
    ax2.bar(indices, total_angular_shift, color='#aaff00', alpha=0.8)
    ax2.axvline(target_residue, color='w', linestyle=':', label='Ligand Injection Site')
    ax2.set_title("Allosteric Conformational Shift (Angular Displacement)", fontsize=14)
    ax2.set_xlabel("Residue Index")
    ax2.set_ylabel("Total $\Delta \phi, \psi$ (Degrees)")
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(assets_dir, exist_ok=True)
    out_file = os.path.join(assets_dir, 'tier5_allostery_yield.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nRendered diagnostic graph to: {out_file}")

if __name__ == "__main__":
    main()
