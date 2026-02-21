"""
AVE Visualizer: Topological Periodic Table
Generates a publication-quality 1x3 render of H, He, and Li.
Demonstrates the mechanical necessity of orbital shells via metric saturation.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.matter.atoms import HydrogenAtom, HeliumAtom, LithiumAtom
from ave.core import constants as k

def get_composite_strain(atom, X, Y, Z):
    # Nuclear Strain
    r_nuc = np.sqrt((X - atom.nucleus.pos[0])**2 + (Y - atom.nucleus.pos[1])**2)
    R_base = 3.0 * k.L_NODE
    r_nuc = np.clip(r_nuc, R_base, None)
    
    # Nuclear metric strain scales with atomic number
    V_tot = (Z * R_base) / r_nuc
    
    # Extract electrons dynamically
    electrons = []
    if hasattr(atom, 'electron'): electrons.append(atom.electron)
    if hasattr(atom, 'e1'): electrons.append(atom.e1)
    if hasattr(atom, 'e2'): electrons.append(atom.e2)
    if hasattr(atom, 'e3'): electrons.append(atom.e3)
    
    for e in electrons:
        r_e = np.sqrt((X - e.pos[0])**2 + (Y - e.pos[1])**2)
        r_e = np.clip(r_e, e.r_core, None)
        V_tot += e.r_core / r_e
        
    # Apply Axiom 4 Saturation Limit
    return np.clip(V_tot, 1e-4, 0.99999)

def render_periodic_evolution():
    print("[-] Instantiating Topological Atomic Hierarchies...")
    atoms = [(HydrogenAtom(), 1), (HeliumAtom(), 2), (LithiumAtom(), 3)]
    
    titles = [
        "Hydrogen ($^1$H) \n $1s$ Resonance", 
        "Helium ($^4$He) \n Phase-Locked $1s^2$ Saturation", 
        "Lithium ($^7$Li) \n Expulsion to $n=2$ Shell ($2s^1$)"
    ]
    
    a0 = k.L_NODE / k.ALPHA_GEOM
    bounds = a0 * 4.5 # Wide enough to see the Lithium 2s orbital
    N = 800
    x = np.linspace(-bounds, bounds, N)
    y = np.linspace(-bounds, bounds, N)
    X, Y = np.meshgrid(x, y)

    print("[-] Calculating Continuous Metric Tensors...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    for i, ax in enumerate(axes):
        atom, Z = atoms[i]
        V_tot = get_composite_strain(atom, X, Y, Z)
        
        c = ax.pcolormesh(X/a0, Y/a0, V_tot, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=1.0), shading='auto')
        
        # Draw theoretical orbital tracks
        if i == 0:
            ax.add_patch(plt.Circle((0, 0), 1.0, color='white', fill=False, linestyle='--', alpha=0.3))
        elif i == 1:
            ax.add_patch(plt.Circle((0, 0), atom.a_He/a0, color='white', fill=False, linestyle='--', alpha=0.3))
        elif i == 2:
            ax.add_patch(plt.Circle((0, 0), atom.a_1s/a0, color='white', fill=False, linestyle='--', alpha=0.3))
            ax.add_patch(plt.Circle((0, 0), atom.a_2s/a0, color='white', fill=False, linestyle='--', alpha=0.5))

        ax.set_aspect('equal')
        ax.set_title(titles[i], fontsize=16, pad=15)
        ax.set_xlabel(r"Distance ($a_0$)", fontsize=12)
        if i == 0:
            ax.set_ylabel(r"Distance ($a_0$)", fontsize=12)
        
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)

    # Global Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(c, cax=cbar_ax)
    cbar.set_label(r'Dielectric Strain ($V / V_{snap}$)', rotation=270, labelpad=20, fontsize=14)
    
    # Save the figure
    out_dir = Path(__file__).parent.parent.parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "atomic_hierarchy_evolution.png"
    
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(out_file, bbox_inches='tight')
    print(f"[+] Graphic saved successfully to: {out_file}")

if __name__ == "__main__":
    render_periodic_evolution()