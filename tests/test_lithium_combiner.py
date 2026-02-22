"""
Test Script: Visualizing Lithium-7 Topology.
Uses the `src/ave/topological/combiner.py` engine to assemble 
7 Borromean Links (3 protons, 4 neutrons) into a Lithium-7 Nucleus.
This demonstrates the secondary-shell growth mechanics of the AVE framework.
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
from test_combiner import plot_alpha_particle  # Reuse the rendering system

def construct_lithium_7(shift_distance: float):
    """
    Lithium-7 consists of a closed-shell Alpha particle core (2p, 2n)
    tightly bound, surrounded by a secondary asymmetric shell (1p, 2n).
    This script geometrically maps that dual-shell stacking behavior.
    """
    # 1. First Shell: The Alpha Core (Tetrahedral Lattice)
    core_shift = shift_distance
    placements = [
        # Core Protons
        {'shift': [ core_shift,  core_shift,  core_shift], 'rot': [0, 0, 0], 'color': '#ff3366', 'label': 'Core Proton'},
        {'shift': [-core_shift, -core_shift,  core_shift], 'rot': [0, 0, np.pi/2], 'color': '#ff3366'},
        # Core Neutrons
        {'shift': [-core_shift,  core_shift, -core_shift], 'rot': [np.pi/2, 0, 0], 'color': '#00ffcc', 'label': 'Core Neutron'},
        {'shift': [ core_shift, -core_shift, -core_shift], 'rot': [0, np.pi/2, 0], 'color': '#00ffcc'}
    ]
    
    # 2. Second Shell: 1 Proton + 2 Neutrons binding to the exterior facies
    # They snap to the outer geometric voids (interstitial sites) of the alpha core.
    # The secondary shell binds at roughly twice the topological offset due to screening.
    outer_shift = shift_distance * 2.2 
    
    # Outer Proton (Attaches to the upper face void)
    placements.append(
        {'shift': [outer_shift, -outer_shift, outer_shift], 'rot': [0, np.pi/4, 0], 'color': '#ff99aa', 'label': 'Outer Shell Proton'}
    )
    # Outer Neutron 1 (Attaches to side void)
    placements.append(
        {'shift': [-outer_shift, -outer_shift, -outer_shift], 'rot': [np.pi/4, 0, 0], 'color': '#99ffee', 'label': 'Outer Shell Neutron'}
    )
    # Outer Neutron 2 (Attaches to opposing side void)
    placements.append(
        {'shift': [outer_shift, outer_shift, -outer_shift], 'rot': [0, 0, np.pi/4], 'color': '#99ffee'}
    )
    
    # Run the physics combiner engine
    assembled_lithium = NucleonCombiner.assemble_cluster(
        FundamentalTopologies.generate_borromean_6_3_2, placements
    )
    
    return assembled_lithium

def plot_lithium_nucleus(lithium_cluster, out_file, title_prefix="Topological"):
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('#0f0f0f')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f0f')
    
    # Render all interlocking knots
    for nucleon in lithium_cluster:
        rings = nucleon['mesh']
        color = nucleon['color']
        
        # Calculate the centroid of this nucleon to place the bounding sphere
        all_points = np.vstack(rings)
        centroid = np.mean(all_points, axis=0)
        
        for i, ring_coords in enumerate(rings):
            label = nucleon['label'] if i == 0 and 'label' in nucleon else ""
            ax.plot(ring_coords[:, 0], ring_coords[:, 1], ring_coords[:, 2], 
                    color=color, linewidth=2.5, alpha=0.9, label=label,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
                
    # Force a perfectly symmetrical viewing angle to see the tetrahedral core
    # while the outer shell orbits it.
    ax.view_init(elev=15, azim=45)
    
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
    
    # Expanded cubic bounds for the secondary shell
    bound = 4.5
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_zlim([-bound, bound])

if __name__ == "__main__":
    # 1. Bound State (Actual physical topology of Li-7)
    lithium_bound = construct_lithium_7(shift_distance=0.85)
    plot_lithium_nucleus(lithium_bound, 'tests/outputs/lithium_7_bound.png', title_prefix="Bound")
    
    # 2. Exploded View (Core remains intact, outer shell blasts away)
    lithium_exploded = construct_lithium_7(shift_distance=2.5)
    plot_lithium_nucleus(lithium_exploded, 'tests/outputs/lithium_7_exploded.png', title_prefix="Exploded View:")
