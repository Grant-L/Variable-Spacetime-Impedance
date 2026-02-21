"""
AVE Visualizer: Topological Lithium
Generates a publication-quality render of the Lithium-7 strain field.
Demonstrates Pauli Exclusion pushing the 3rd electron to the n=2 harmonic.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ave.matter.atoms import LithiumAtom
from ave.core import constants as k

def render_lithium():
    print("[-] Instantiating Topological Lithium...")
    atom = LithiumAtom()
    a0 = atom.a_0
    
    # Grid expanded to capture the distant 2s harmonic
    bounds = a0 * 4.5
    N = 900
    x = np.linspace(-bounds, bounds, N)
    y = np.linspace(-bounds, bounds, N)
    X, Y = np.meshgrid(x, y)
    
    print("[-] Calculating Continuous Metric Tensor (Z=3)...")
    # Nuclear Strain
   # FIXED: Ensure the clip boundary matches the numerator
    R_eff = atom.Z * 3.0 * k.L_NODE
    r_nuc = np.sqrt(X**2 + Y**2)
    r_nuc = np.clip(r_nuc, R_eff, None)
    V_nuc = R_eff / r_nuc
    # 1s Inner Shell (e1 and e2)
    r_e1 = np.sqrt((X - atom.e1.pos[0])**2 + (Y - atom.e1.pos[1])**2)
    r_e1 = np.clip(r_e1, atom.e1.r_core, None)
    V_e1 = atom.e1.r_core / r_e1
    
    r_e2 = np.sqrt((X - atom.e2.pos[0])**2 + (Y - atom.e2.pos[1])**2)
    r_e2 = np.clip(r_e2, atom.e2.r_core, None)
    V_e2 = atom.e2.r_core / r_e2
    
    # 2s Outer Shell (e3)
    r_e3 = np.sqrt((X - atom.e3.pos[0])**2 + (Y - atom.e3.pos[1])**2)
    r_e3 = np.clip(r_e3, atom.e3.r_core, None)
    V_e3 = atom.e3.r_core / r_e3
    
    V_tot = np.clip(V_nuc + V_e1 + V_e2 + V_e3, 1e-4, 0.99999)

    print("[-] Rendering Publication Graphic...")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    c = ax.pcolormesh(X/a0, Y/a0, V_tot, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=1.0), shading='auto')
    
    ax.add_patch(plt.Circle((0, 0), atom.a_1s/a0, color='white', fill=False, linestyle='-', alpha=0.6, linewidth=1.5))
    ax.add_patch(plt.Circle((0, 0), atom.a_2s/a0, color='cyan', fill=False, linestyle='--', alpha=0.5, linewidth=1.2))
    
    # NEW: Explicitly mark the 1s and 2s electrons
    ax.scatter([0], [0], color='white', s=100, edgecolor='yellow', zorder=5, label='Lithium Nucleus')
    ax.scatter([atom.e1.pos[0]/a0, atom.e2.pos[0]/a0], 
               [atom.e1.pos[1]/a0, atom.e2.pos[1]/a0], 
               color='cyan', s=40, edgecolor='white', zorder=5, label='1s Saturated Pair')
    ax.scatter([atom.e3.pos[0]/a0], [atom.e3.pos[1]/a0], 
               color='magenta', s=60, edgecolor='white', zorder=5, label='2s Expelled Soliton')
    ax.legend(loc='upper right', framealpha=0.3, labelcolor='white')
    
    ax.set_aspect('equal')
    ax.set_title(r"Topological Lithium ($^7$Li) - Metric Expulsion to the $n=2$ Shell", fontsize=15, pad=15)
    ax.set_xlabel(r"Distance ($a_0$)", fontsize=12)
    ax.set_ylabel(r"Distance ($a_0$)", fontsize=12)
    
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Dielectric Strain ($V / V_{snap}$)', rotation=270, labelpad=20, fontsize=12)
    
    out_dir = root_dir / "periodic_table_of_knots" / "chapters" / "003_lithium" / "simulations" / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "lithium_topological_strain.png"
    
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    print(f"[+] Graphic saved successfully to: {out_file}")

if __name__ == "__main__":
    render_lithium()