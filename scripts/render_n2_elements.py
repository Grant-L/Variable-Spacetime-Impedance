import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Ensure correct pathing for your existing src architecture
# Script is at: scripts/render_n2_elements.py
# Need to go up one level to reach root, then add src/
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path: 
    sys.path.insert(0, str(src_dir))

from ave.matter.atoms import NitrogenAtom, OxygenAtom, FluorineAtom, NeonAtom
from ave.core import constants as k

def render_element(atom_cls, elem_name, symbol, mass_num, subtitle, chap_num):
    atom = atom_cls()
    a0 = atom.a_0
    bounds = a0 * 3.5
    N_grid = 800
    X, Y = np.meshgrid(np.linspace(-bounds, bounds, N_grid), np.linspace(-bounds, bounds, N_grid))
    
    # Base Nuclear Refractive Strain Gradient
    R_eff = atom.Z * 3.0 * k.L_NODE
    V_tot = R_eff / np.clip(np.sqrt(X**2 + Y**2), R_eff, None)
    
    # Local Dielectric Displacements from Solitons
    electrons = [atom.e1, atom.e2] + atom.outer_electrons
    for e in electrons:
        V_tot += e.r_core / np.clip(np.sqrt((X - e.pos[0])**2 + (Y - e.pos[1])**2), e.r_core, None)
        
    # Enforce Axiom 4 Strict Dielectric Breakdown Limit
    V_tot = np.clip(V_tot, 1e-4, 0.99999)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    c = ax.pcolormesh(X/a0, Y/a0, V_tot, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=1.0), shading='auto')
    
    # Hardware Standing Wave Orbital Boundaries
    ax.add_patch(plt.Circle((0, 0), atom.a_1s/a0, color='white', fill=False, linestyle='-', alpha=0.6))
    ax.add_patch(plt.Circle((0, 0), atom.a_2s/a0, color='cyan', fill=False, linestyle='--', alpha=0.5))
    
    # Render Knots
    ax.scatter([0], [0], color='white', s=100, edgecolor='yellow', zorder=5) # Nucleus
    ax.scatter([atom.e1.pos[0]/a0, atom.e2.pos[0]/a0], [atom.e1.pos[1]/a0, atom.e2.pos[1]/a0], color='cyan', s=40, zorder=5)
    
    outer_x = [e.pos[0]/a0 for e in atom.outer_electrons]
    outer_y = [e.pos[1]/a0 for e in atom.outer_electrons]
    ax.scatter(outer_x, outer_y, color='magenta', s=60, zorder=5)

    ax.set_aspect('equal')
    ax.set_title(rf"Topological {elem_name} ($^{{{mass_num}}}{symbol}$) - {subtitle}", fontsize=15, pad=15)
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04).set_label(r'Dielectric Strain ($V/V_{snap}$)', rotation=270, labelpad=20)
    
    out_dir = root_dir / "periodic_table_of_knots" / "chapters" / f"{chap_num:03d}_{elem_name.lower()}" / "simulations" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = out_dir / f"{elem_name.lower()}_strain.png"
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved {elem_name} topological strain field to {file_path}")

if __name__ == "__main__":
    elements = [
        (NitrogenAtom, "Nitrogen", "N", 14, "Pentagonal Resonance", 7),
        (OxygenAtom, "Oxygen", "O", 16, "Hexagonal Resonance", 8),
        (FluorineAtom, "Fluorine", "F", 19, "Heptagonal Resonance", 9),
        (NeonAtom, "Neon", "Ne", 20, "Octagonal Full Shell Saturation", 10)
    ]
    
    for cls, name, sym, mass, sub, chap in elements:
        print(f"Rendering {name}...")
        render_element(cls, name, sym, mass, sub, chap)