"""
Test Script: Visualizing the Nucleon Combiner.
Uses the `src/ave/topological/combiner.py` engine to assemble 
four 6^3_2 Borromean Links (2 protons, 2 neutrons) into a Helium-4 Nucleus.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Ensure the core AVE packages are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ave.topological.borromean import FundamentalTopologies
from ave.topological.combiner import NucleonCombiner

def construct_helium_4(shift_distance: float):
    """
    Helium-4 is a perfectly symmetrical, closed topological shell.
    We position 4 nucleons (2p, 2n) in a tetrahedral interlocking formation.
    The exact rotations interleave the boundary curves without violating
    the hard-sphere repulsion of the underlying discrete lattice.
    """
    
    # Tetrahedral lattice coordinates scaled by the optimal overlap distance
    placements = [
        # Proton 1 (Red)
        {'shift': [ shift_distance,  shift_distance,  shift_distance], 
         'rot':   [0, 0, 0], 
         'color': '#ff3366', 'label': 'Proton (Up)'},
         
        # Proton 2 (Red)
        {'shift': [-shift_distance, -shift_distance,  shift_distance], 
         'rot':   [0, 0, np.pi/2], 
         'color': '#ff3366', 'label': 'Proton (Down)'},
         
        # Neutron 1 (Cyan)
        {'shift': [-shift_distance,  shift_distance, -shift_distance], 
         'rot':   [np.pi/2, 0, 0], 
         'color': '#00ffcc', 'label': 'Neutron (Up)'},
         
        # Neutron 2 (Cyan)
        {'shift': [ shift_distance, -shift_distance, -shift_distance], 
         'rot':   [0, np.pi/2, 0], 
         'color': '#00ffcc', 'label': 'Neutron (Down)'}
    ]
    
    # Run the physics combiner engine
    assembled_helium = NucleonCombiner.assemble_cluster(
        FundamentalTopologies.generate_borromean_6_3_2, placements
    )
    
    return assembled_helium

def plot_alpha_particle(helium_cluster, out_file, title_prefix="Topological"):
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('#0f0f0f')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f0f')
    
    # Render all interlocking knots
    for nucleon in helium_cluster:
        rings = nucleon['mesh']
        color = nucleon['color']
        
        # Calculate the centroid of this nucleon to place the bounding sphere
        all_points = np.vstack(rings)
        centroid = np.mean(all_points, axis=0)
        
        # Each nucleon is composed of 3 mutually interlocking rings
        for i, ring_coords in enumerate(rings):
            # Only label the first ring of a nucleon to prevent legend clutter
            label = nucleon['label'] if i == 0 else ""
            ax.plot(ring_coords[:, 0], ring_coords[:, 1], ring_coords[:, 2], 
                    color=color, linewidth=2.5, alpha=0.9, label=label,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
                
    # Force a perfectly symmetrical viewing angle (looking down a 2-fold symmetry axis)
    # This projects the tetrahedral lattice into a perfect square, making 
    # all 4 nucleons equally visible ("facing out of the page").
    ax.view_init(elev=0, azim=0)
    
    _clean_axes(ax)
    
    os.makedirs('tests/outputs', exist_ok=True)
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Validation Render: {out_file}")

def _clean_axes(ax):
    """Helper to enforce dark-mode aesthetic boundaries"""
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)
    ax.axis('off')
    
    # Enforce cubic bounds to prevent aspect stretching
    bound = 2.5
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_zlim([-bound, bound])

if __name__ == "__main__":
    # 1. Bound State (Actual physical topology)
    helium_bound = construct_helium_4(shift_distance=0.85)
    plot_alpha_particle(helium_bound, 'tests/outputs/helium_4_bound.png', title_prefix="Bound")
    
    # 2. Exploded View (For easier human viewing of the interlocking mechanism)
    helium_exploded = construct_helium_4(shift_distance=2.5)
    plot_alpha_particle(helium_exploded, 'tests/outputs/helium_4_exploded.png', title_prefix="Exploded View:")
