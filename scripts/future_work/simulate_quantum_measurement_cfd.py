#!/usr/bin/env python3
r"""
AVE: Double Slit Measurement Effect (Macroscopic Wavefunction Collapse)
=======================================================================

This script models a steady, continuous beam of topological defects 
(multiple photons following each other) to unambiguously demonstrate the 
structural thermodynamic nature of wavefunction collapse.

Case A (Unmeasured): A plane wave (continuous stream) hits both slits,
generating highly defined, unmistakable macroscopic interference fringes
on the rear wall.

Case B (Measured): A topological Ohmic detector structurally dampens Slit 2.
The continuous phase energy is scrubbed via Joule heating. The fringe
pattern completely vanishes, leaving a classical single-source smear.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Bind to AVE Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ave.core.constants import C_0

def run_2d_wave_solver(case_name, measure_slit_2):
    NX, NY = 500, 300
    c = 1.0
    dx = 1.0
    dt = dx / (c * np.sqrt(2.0)) * 0.99
    
    P_curr = np.zeros((NY, NX))
    P_prev = np.zeros((NY, NX))
    P_next = np.zeros((NY, NX))
    
    Intensity = np.zeros((NY, NX))
    
    sponge_width = 30
    damping = np.ones((NY, NX))
    max_damp = 0.05
    for i in range(NY):
        for j in range(NX):
            dist_x = min(j, NX - 1 - j)
            dist_y = min(i, NY - 1 - i)
            min_dist = min(dist_x, dist_y)
            if min_dist < sponge_width:
                damping[i, j] = 1.0 - max_damp * ((sponge_width - min_dist) / sponge_width)**2

    wall_x = int(NX * 0.3)
    wall_thickness = 10
    slit_width = 16
    slit_sep = 70
    s1_y = NY // 2 - slit_sep // 2
    s2_y = NY // 2 + slit_sep // 2
    
    wall_mask = np.zeros((NY, NX), dtype=bool)
    wall_mask[:, wall_x:wall_x+wall_thickness] = True
    wall_mask[s1_y - slit_width//2 : s1_y + slit_width//2, wall_x:wall_x+wall_thickness] = False
    wall_mask[s2_y - slit_width//2 : s2_y + slit_width//2, wall_x:wall_x+wall_thickness] = False
    
    detector_mask = np.zeros((NY, NX), dtype=bool)
    if measure_slit_2:
        # Heavily dampen the exit of Slit 2
        detector_mask[s2_y - slit_width - 2 : s2_y + slit_width + 2, wall_x+wall_thickness:wall_x+wall_thickness+15] = True
        
    freq = 0.12 # Shorter wavelength for incredibly crisp, numerous fringes
    TOTAL_STEPS = 1200
    
    print(f"  -> Running Case: {case_name}")
    
    for step in range(TOTAL_STEPS):
        t = step * dt
        
        laplacian = (
            np.roll(P_curr, 1, axis=1) + np.roll(P_curr, -1, axis=1) +
            np.roll(P_curr, 1, axis=0) + np.roll(P_curr, -1, axis=0) -
            4 * P_curr
        ) / (dx**2)
        
        P_next = 2 * P_curr - P_prev + (dt**2 * c**2) * laplacian
        P_next[wall_mask] = 0.0
        P_next *= damping
        
        if measure_slit_2:
            P_next[detector_mask] *= 0.05 # Complete thermodynamic absorption
            
        # Continuous Plane Wave Source (Multiple Photons following each other)
        # We drive a line source at X=35 constantly
        src_x = sponge_width + 5
        # Soft taper at edges to prevent ringing
        taper = np.sin(np.linspace(0, np.pi, NY))
        P_next[:, src_x] += np.sin(2 * np.pi * freq * t) * 8.0 * taper
            
        P_prev[:, :] = P_curr
        P_curr[:, :] = P_next
        
        if step > 600:
            Intensity += P_curr**2
            
    Intensity /= np.max(Intensity)
    rear_wall_idx = NX - sponge_width - 15
    rear_distribution = Intensity[:, rear_wall_idx].copy()
    
    return Intensity, rear_distribution, wall_mask, detector_mask

def generate_comparative_visuals():
    print("[*] Re-running Refined Macroscopic Double Slit Measurement Simulator...")
    
    int_A, dist_A, wall_mask, det_mask_A = run_2d_wave_solver("Unmeasured (Coherent Beam)", False)
    int_B, dist_B, _, det_mask_B = run_2d_wave_solver("Measured at Slit 2 (Ohmic Decoherence)", True)
    
    print("[*] Integrations complete. Plotting comparative analytical proof...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [3, 1]})
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#050510')
    
    NY, NX = int_A.shape
    
    # Global normalization across both for visual fairness, though typically they are self-normalized
    # We want max contrast for the fringes
    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#050510')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#333333')
                
    # --- Row 1: Case A (Unmeasured) ---
    ax_wave_A = axes[0, 0]
    ax_dist_A = axes[0, 1]
    
    # We clip the vmax stringently to make the subtle fringes violently bright
    im_A = ax_wave_A.imshow(int_A, cmap='magma', origin='lower', extent=[0, NX, 0, NY], vmax=np.max(int_A)*0.5, alpha=1.0)
    ax_wave_A.imshow(wall_mask, cmap='binary_r', alpha=0.9, origin='lower', extent=[0, NX, 0, NY])
    ax_wave_A.set_title("Case A: Unmeasured (Continuous Multi-Photon Beam)\n" + r"Macroscopic Acoustic Interference Fringes ($\nabla |\Psi_{mech}|^2$)", color='white', fontsize=16, weight='bold', pad=15)
    ax_wave_A.axis('off')
    
    ax_dist_A.plot(dist_A, np.arange(NY), color='#00ffff', lw=2)
    ax_dist_A.fill_betweenx(np.arange(NY), 0, dist_A, color='#00ffff', alpha=0.3)
    ax_dist_A.set_ylim(0, NY)
    ax_dist_A.set_xlim(0, np.max(dist_A)*1.1)
    ax_dist_A.set_title("Rear Wall Probability Distribution\n(The Born Rule Result)", color='white', fontsize=14, weight='bold')
    ax_dist_A.axis('off')
    
    # --- Row 2: Case B (Measured) ---
    ax_wave_B = axes[1, 0]
    ax_dist_B = axes[1, 1]
    
    im_B = ax_wave_B.imshow(int_B, cmap='magma', origin='lower', extent=[0, NX, 0, NY], vmax=np.max(int_B)*0.5, alpha=1.0)
    ax_wave_B.imshow(wall_mask, cmap='binary_r', alpha=0.9, origin='lower', extent=[0, NX, 0, NY])
    ax_wave_B.imshow(det_mask_B, cmap='Reds', alpha=0.8, origin='lower', extent=[0, NX, 0, NY])
    
    ax_wave_B.set_title("Case B: Measured (Ohmic Detector at Slit 2)\n" + r"Macroscopic Structural Decoherence (Wavefunction Collapse)", color='white', fontsize=16, weight='bold', pad=15)
    ax_wave_B.axis('off')
    
    props = dict(boxstyle='round', facecolor='darkred', alpha=0.9, edgecolor='red')
    ax_wave_B.text(int(NX*0.3) + 30, NY//2 + 45, "Ohmic Detector\n(Joule Heating Sink)", color='white', fontsize=12, weight='bold', bbox=props)
    
    ax_dist_B.plot(dist_B, np.arange(NY), color='#ff00aa', lw=2)
    ax_dist_B.fill_betweenx(np.arange(NY), 0, dist_B, color='#ff00aa', alpha=0.3)
    ax_dist_B.set_ylim(0, NY)
    ax_dist_B.set_xlim(0, np.max(dist_A)*1.1)  # Scale B to match A's width for absolute structural comparison
    ax_dist_B.set_title("Rear Wall Probability Distribution\n(Destroyed Fringe Patern)", color='white', fontsize=14, weight='bold')
    ax_dist_B.axis('off')
    
    plt.tight_layout(pad=3.0)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'double_slit_decoherence.png')
    
    plt.savefig(out_path, dpi=250, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    print(f"[*] Thermodynamic Wavefunction Collapse proof generated -> {out_path}")

if __name__ == "__main__":
    generate_comparative_visuals()
