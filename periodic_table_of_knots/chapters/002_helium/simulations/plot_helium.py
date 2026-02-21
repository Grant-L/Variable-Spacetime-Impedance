"""
AVE Visualizer: Topological Helium
Generates a publication-quality render of the Helium-4 strain field.
Demonstrates the 180-degree phase-locked pairing of the 1s shell.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ave.matter.atoms import HeliumAtom
from ave.core import constants as k

def render_helium():
    print("[-] Instantiating Topological Helium...")
    atom = HeliumAtom()
    a0 = atom.a_0
    
    # Grid scaled to perfectly frame the tightly bound 1s orbital
    bounds = a0 * 1.5
    N = 800
    x = np.linspace(-bounds, bounds, N)
    y = np.linspace(-bounds, bounds, N)
    X, Y = np.meshgrid(x, y)
    
    print("[-] Calculating Continuous Metric Tensor (Z=2)...")
    
    # FIXED: Ensure the clip boundary matches the numerator to prevent massive visual washout
    R_eff = atom.Z * 3.0 * k.L_NODE
    r_nuc = np.sqrt(X**2 + Y**2)
    r_nuc = np.clip(r_nuc, R_eff, None)
    V_nuc = R_eff / r_nuc

    # Electron 1 Strain (Phase 0)
    r_e1 = np.sqrt((X - atom.e1.pos[0])**2 + (Y - atom.e1.pos[1])**2)
    r_e1 = np.clip(r_e1, atom.e1.r_core, None)
    V_e1 = atom.e1.r_core / r_e1
    
    # Electron 2 Strain (Phase 180 - Antipodal)
    r_e2 = np.sqrt((X - atom.e2.pos[0])**2 + (Y - atom.e2.pos[1])**2)
    r_e2 = np.clip(r_e2, atom.e2.r_core, None)
    V_e2 = atom.e2.r_core / r_e2
    
    # Total Axiom 4 Superposition (Saturated 1s Shell)
    V_tot = np.clip(V_nuc + V_e1 + V_e2, 1e-4, 0.99999)

    print("[-] Rendering Publication Graphic...")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Plot the strain field
    c = ax.pcolormesh(X/a0, Y/a0, V_tot, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=1.0), shading='auto')
    
    # Plot the squeezed orbital track
    orbit = plt.Circle((0, 0), atom.a_He/a0, color='white', fill=False, linestyle='--', alpha=0.4, linewidth=1.5)
    ax.add_patch(orbit)
    
    # Explicitly mark the phase-locked 1s^2 electrons
    ax.scatter([0], [0], color='white', s=80, edgecolor='yellow', zorder=5, label='Alpha Particle')
    ax.scatter([atom.e1.pos[0]/a0, atom.e2.pos[0]/a0], 
               [atom.e1.pos[1]/a0, atom.e2.pos[1]/a0], 
               color='cyan', s=40, edgecolor='white', zorder=5, label='Phase-Locked Trefoils')
    ax.legend(loc='upper right', framealpha=0.3, labelcolor='white')
    
    ax.set_aspect('equal')
    ax.set_title(r"Topological Helium ($^4$He) - Phase-Locked $1s^2$ Saturation", fontsize=15, pad=15)
    ax.set_xlabel(r"Distance ($a_0$)", fontsize=12)
    ax.set_ylabel(r"Distance ($a_0$)", fontsize=12)
    
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Dielectric Strain ($V / V_{snap}$)', rotation=270, labelpad=20, fontsize=12)
    
    out_dir = root_dir / "periodic_table_of_knots" / "chapters" / "002_helium" / "simulations" / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "helium_topological_strain.png"
    
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    print(f"[+] Graphic saved successfully to: {out_file}")

if __name__ == "__main__":
    render_helium()
