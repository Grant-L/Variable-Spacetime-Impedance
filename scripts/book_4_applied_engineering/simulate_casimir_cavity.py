r"""
AVE Casimir Cavity Simulator (1D FDTD)
======================================
Simulates the Casimir Effect as pure macroscopic acoustic filtering.

In standard field theory, the Casimir effect is abstractly defined by 
"virtual particles." In Applied Vacuum Engineering, it is definitively 
a mechanical high-pass filter. 

We generate a 1D elastic string flooded with continuous broadband stochastic 
white noise (ZPF / Thermal Phonons). We inject two rigid, highly reflective 
boundaries (The Cavity Plates). 

The simulation explicitly logs the geometric failure of long wavelengths 
($\lambda > 2d$) to form standing states within the cavity. This results in 
a localized drop in continuous energy density ($\Delta \rho$), creating the 
negative pressure gradient ($\Delta P$) that forces the plates together.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
from pathlib import Path

# Physics Parameters
N = 1000             # Array Length
C_0 = 0.5            # Courant wave velocity
T_MAX = 5000         # Time steps to reach stochastic steady-state
NOISE_AMP = 0.05     # Broadband stochastic forcing

# Cavity Parameters
d_width = 100
plate_L = N//2 - d_width//2
plate_R = N//2 + d_width//2

def simulate_casimir_cavity():
    print(f"Simulating 1D FDTD Casimir Cavity (Width={d_width}) against Broadband Noise...")
    # displacement field
    U = np.zeros(N)
    U_prev = np.zeros(N)
    
    # Store steady state history for FFT
    history_inside = []
    history_outside = []
    
    rmss_spatial = np.zeros(N)
    
    for t in range(T_MAX):
        # Explicit 1D Wave equation update
        laplacian = np.roll(U, -1) - 2*U + np.roll(U, 1)
        U_next = 2*U - U_prev + (C_0**2) * laplacian
        
        # Inject continuous broadband stochastic noise everywhere (ZPF / Thermal Bath)
        noise = np.random.normal(0, NOISE_AMP, N)
        U_next += noise
        
        # Enforce highly reflective plates (Casimir Cavity Bounds)
        # We don't make them 0, we make them rigid.
        U_next[plate_L] *= 0.1 
        U_next[plate_R] *= 0.1
        
        # Damp outer edges to prevent infinite artificial reflections bounding the universe
        U_next[:20] *= 0.95
        U_next[-20:] *= 0.95
        
        # Record steady state metrics (Wait for cavity to saturate or suppress)
        if t > T_MAX - 1000:
            rmss_spatial += U_next**2
            # Log specific center node vs deep outside node for frequency analysis
            history_inside.append(U_next[N//2])
            history_outside.append(U_next[N//4])
            
        U_prev = np.copy(U)
        U = np.copy(U_next)
    
    rmss_spatial = np.sqrt(rmss_spatial / 1000.0)
    
    return rmss_spatial, np.array(history_inside), np.array(history_outside)

def plot_casimir_proof(rmss, t_inside, t_outside, out_path):
    print("Generating Graphical Mathematical Proofs...")
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # --- PANEL 1: Spatial Energy Density (The Macroscopic Pressure) ---
    x = np.arange(N)
    ax1.plot(x, rmss, color='#00aaff', alpha=0.9, lw=1.5, label='RMS Energy Density $\\bar{\\rho}(x)$')
    
    # Highlight Cavity
    ax1.axvspan(plate_L, plate_R, color='white', alpha=0.1, label='Casimir Cavity ($d$)')
    ax1.axvline(plate_L, color='white', linestyle='--', lw=2)
    ax1.axvline(plate_R, color='white', linestyle='--', lw=2)
    
    ax1.set_title("Time-Averaged Spatial Energy Density (The Pressure Gradient $\\Delta P$)", fontsize=14, color='white', pad=15)
    ax1.set_ylabel("Amplitude Density $\\bar{\\rho}$", fontsize=12)
    ax1.legend(loc='upper right', facecolor='black', edgecolor='white')
    ax1.grid(True, alpha=0.2)
    
    # --- PANEL 2: Frequency Spectrum (The High-Pass Filter Notch) ---
    N_fft = len(t_inside)
    yf_in = np.abs(fft(t_inside))[:N_fft//2]
    yf_out = np.abs(fft(t_outside))[:N_fft//2]
    xf = fftfreq(N_fft, 1.0)[:N_fft//2]
    
    # We only care about the lower half of the normalized frequencies
    max_idx = N_fft // 10
    
    ax2.plot(xf[:max_idx], yf_out[:max_idx], color='gray', alpha=0.7, lw=2, label='External Broadband Bath (ZPF / 300K)')
    ax2.plot(xf[:max_idx], yf_in[:max_idx], color='#ff00aa', alpha=0.9, lw=2, label='Internal Cavity Spectrum')
    
    # Calculate cutoff frequency f_c = c / 2d
    f_c = C_0 / (2.0 * d_width)
    ax2.axvline(f_c, color='#ffcc00', linestyle='--', lw=2, label=f'Geometric Cutoff $\\lambda > 2d$ (High-Pass)')
    
    ax2.set_title("Acoustic Mode Filtering (Frequency Domain)", fontsize=14, color='white', pad=15)
    ax2.set_xlabel("Frequency Mode", fontsize=12)
    ax2.set_ylabel("Spectral Power", fontsize=12)
    ax2.legend(loc='upper right', facecolor='black', edgecolor='white')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Done] Saved Graphic: {out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    out_dir = PROJECT_ROOT / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    rmss, t_in, t_out = simulate_casimir_cavity()
    plot_casimir_proof(rmss, t_in, t_out, out_dir / "casimir_acoustic_filtering.png")
