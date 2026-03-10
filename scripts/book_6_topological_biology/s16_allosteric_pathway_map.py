#!/usr/bin/env python3
"""
s16_allosteric_pathway_map.py
=============================

Tier 5b: Full Allosteric Pathway Mapping

Sweeps a synthetic impedance perturbation across every residue position
in a folded backbone, building the full N×N allosteric coupling matrix:

    M[i,j] = angular displacement at residue j when ligand is injected at i

This matrix reveals the standing-wave node/antinode topology of the
transmission line.  Off-diagonal peaks indicate allosteric pathways;
diagonal dominance indicates local yield.

Physics:  Z → Γ → S₁₁ minimisation (zero empirical parameters)
Analogy:  Injecting a tuning stub at every port of an RF filter
          and measuring the S-parameter shift at all other ports.
"""

import os
import sys
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from s11_fold_engine_v3_jax import fold_s11_jax, compute_z_topo

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SEQ_LEN = 12          # Keep short for tractable sweep (N² folds)
SEQUENCE = "A" * SEQ_LEN
LIGAND_Z = 15.0 + 15.0j   # Same reactive load as Tier 5
NATIVE_STEPS = 3000
HOLO_STEPS = 1500


def compute_angular_displacement(angles_a, angles_b, N):
    """Compute per-residue angular displacement between two angle vectors."""
    phi_a, psi_a = angles_a[:N], angles_a[N:2*N]
    phi_b, psi_b = angles_b[:N], angles_b[N:2*N]
    dphi = np.degrees(np.array(phi_b - phi_a))
    dpsi = np.degrees(np.array(psi_b - psi_a))
    return np.sqrt(dphi**2 + dpsi**2)


def main():
    N = SEQ_LEN
    print("=" * 66)
    print(f" ALLOSTERIC PATHWAY MAP: {SEQUENCE} (N={N})")
    print("=" * 66)

    # ── Phase 1: Fold the native state once ──
    print(f"\n[1/3] Folding native state ({NATIVE_STEPS} steps)...", flush=True)
    z_native = compute_z_topo(SEQUENCE)
    ca_nat, _, trace_nat, _, angles_native = fold_s11_jax(
        SEQUENCE, n_steps=NATIVE_STEPS, lr=2e-3, anneal=True, n_starts=1
    )
    print(f"  Native loss = {trace_nat[-1]:.6f}")

    # ── Phase 2: Sweep perturbation across all residue positions ──
    coupling_matrix = np.zeros((N, N))   # M[injection_site, response_site]

    print(f"\n[2/3] Sweeping ligand injection across {N} sites...", flush=True)
    t0_sweep = time.time()

    for inj in range(N):
        # Build perturbed Z_topo
        z_holo = jnp.array(z_native)
        z_holo = z_holo.at[inj].set(LIGAND_Z)

        # Re-equilibrate from native angles
        _, _, trace_h, _, angles_holo = fold_s11_jax(
            SEQUENCE, n_steps=HOLO_STEPS, lr=1e-3, anneal=False,
            n_starts=1, z_topo_override=z_holo,
            initial_angles=angles_native
        )

        disp = compute_angular_displacement(angles_native, angles_holo, N)
        coupling_matrix[inj, :] = disp

        peak_j = np.argmax(disp)
        is_distal = "DISTAL" if peak_j != inj else "LOCAL"
        print(f"  inj={inj:2d}  loss={trace_h[-1]:.4f}  "
              f"peak={peak_j:2d} ({disp[peak_j]:.1f}°)  [{is_distal}]",
              flush=True)

    dt = time.time() - t0_sweep
    print(f"\n  Sweep completed in {dt:.0f}s")

    # ── Phase 3: Analysis ──
    print(f"\n[3/3] Analysing coupling matrix...", flush=True)

    # Diagonal vs off-diagonal energy
    diag_mean = np.mean(np.diag(coupling_matrix))
    off_diag = coupling_matrix[~np.eye(N, dtype=bool)]
    off_mean = np.mean(off_diag)
    off_max = np.max(off_diag)

    print(f"  Diagonal mean (local strain):      {diag_mean:.1f}°")
    print(f"  Off-diagonal mean (distal strain):  {off_mean:.1f}°")
    print(f"  Off-diagonal max (strongest path):  {off_max:.1f}°")
    print(f"  Distal/Local ratio:                 {off_mean/diag_mean:.2f}")

    # Count how many injection sites produce distal maxima
    n_distal = 0
    for inj in range(N):
        if np.argmax(coupling_matrix[inj, :]) != inj:
            n_distal += 1
    print(f"  Distal-dominant injections:          {n_distal}/{N}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTTING
    # ═══════════════════════════════════════════════════════════════
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), width_ratios=[1.2, 1])

    # --- Left: Full coupling heatmap ---
    ax1 = axes[0]
    im = ax1.imshow(coupling_matrix, cmap='inferno', aspect='equal',
                    interpolation='nearest', origin='lower')
    ax1.set_xlabel("Response Residue $j$", fontsize=13)
    ax1.set_ylabel("Injection Residue $i$", fontsize=13)
    ax1.set_title("Allosteric Coupling Matrix $M_{ij}$\n"
                  r"(Angular displacement at $j$ when ligand at $i$)",
                  fontsize=13, pad=12)
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label("$\\Delta\\theta$ (degrees)", fontsize=12)

    # Mark diagonal
    for k in range(N):
        ax1.plot(k, k, 's', color='cyan', markersize=5, alpha=0.6)

    # --- Right: Row-mean profile (average strain received) ---
    ax2 = axes[1]
    col_mean = coupling_matrix.mean(axis=0)  # avg strain received by j
    row_mean = coupling_matrix.mean(axis=1)  # avg strain emitted by i

    ax2.barh(range(N), col_mean, height=0.4, color='#ff5555',
             alpha=0.8, label="Strain received (mean over i)")
    ax2.barh(np.arange(N) + 0.4, row_mean, height=0.4, color='#00aaff',
             alpha=0.8, label="Strain emitted (mean over j)")
    ax2.set_yticks(range(N))
    ax2.set_yticklabels([f"{SEQUENCE[k]}{k}" for k in range(N)])
    ax2.set_xlabel("Mean Angular Shift (°)", fontsize=13)
    ax2.set_title("Per-Residue Allosteric\nSusceptibility Profile", fontsize=13, pad=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.15, axis='x')

    plt.tight_layout()

    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', '..', 'assets', 'sim_outputs')
    os.makedirs(assets_dir, exist_ok=True)
    out_file = os.path.join(assets_dir, 'allosteric_pathway_matrix.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nRendered pathway map to: {out_file}")

    # Save raw matrix for further analysis
    np.save(os.path.join(assets_dir, 'allosteric_coupling_matrix.npy'),
            coupling_matrix)
    print(f"Saved raw matrix to:     allosteric_coupling_matrix.npy")


if __name__ == "__main__":
    main()
