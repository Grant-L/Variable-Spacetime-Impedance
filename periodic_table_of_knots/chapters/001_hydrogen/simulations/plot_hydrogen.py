"""
AVE Visualizer: Topological Hydrogen
Generates a publication-quality render of the Protium strain field.
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add src directory to path if running as script
# Script is at: periodic_table_of_knots/chapters/001_hydrogen/simulations/plot_hydrogen.py
# Need to go up 5 levels to reach root, then add src/
root_dir = Path(__file__).parent.parent.parent.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.matter.atoms import HydrogenAtom
from ave.core import constants as k

def render_hydrogen_strain_field():
    print("[-] Instantiating Topological Hydrogen...")
    h_atom = HydrogenAtom()
    a0 = h_atom.a_0
    
    # Create high-res grid (1.5x Bohr radius)
    bounds = a0 * 1.5
    N = 800
    x = np.linspace(-bounds, bounds, N)
    y = np.linspace(-bounds, bounds, N)
    X, Y = np.meshgrid(x, y)
    
    print("[-] Calculating Continuous Metric Tensor (Axiom 4)...")
    # Proton Strain
    r_p = np.sqrt(X**2 + Y**2)
    r_p = np.clip(r_p, h_atom.nucleus.R_core, None)
    V_p = h_atom.nucleus.R_core / r_p
    
    # Electron Strain (Orbiting at +a_0 on the X-axis)
    r_e = np.sqrt((X - a0)**2 + Y**2)
    r_e = np.clip(r_e, h_atom.electron.r_core, None)
    V_e = h_atom.electron.r_core / r_e
    
    # Total Axiom 4 Superposition
    V_tot = np.clip(V_p + V_e, 1e-6, 0.99999)

    print("[-] Rendering Publication Graphic...")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Plot the Non-Linear Varactor Strain (using LogNorm due to 1/r steepness)
    c = ax.pcolormesh(X/a0, Y/a0, V_tot, cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1.0), shading='auto')
    
    # Plot the classical orbital track
    orbit = plt.Circle((0, 0), 1.0, color='white', fill=False, linestyle='--', alpha=0.3, linewidth=1)
    ax.add_patch(orbit)
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_title(r"Topological Hydrogen (Protium) Metric Strain ($\mathcal{M}_A$ Lattice)", fontsize=14, pad=15)
    ax.set_xlabel(r"Distance ($a_0$)", fontsize=12)
    ax.set_ylabel(r"Distance ($a_0$)", fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Dielectric Strain ($V / V_{snap}$)', rotation=270, labelpad=20, fontsize=12)
    
    # Save the figure
    out_dir = root_dir / "periodic_table_of_knots" / "chapters" / "001_hydrogen" / "simulations" / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "hydrogen_topological_strain.png"
    
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    print(f"[+] Graphic saved successfully to: {out_file}")

if __name__ == "__main__":
    render_hydrogen_strain_field()