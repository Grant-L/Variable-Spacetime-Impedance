import matplotlib.pyplot as plt
import numpy as np

def draw_component(ax, start, end, type='L', color='black', label=''):
    """Helper to draw circuit components (Inductor/Capacitor) between points."""
    x = np.linspace(start[0], end[0], 100)
    y = np.linspace(start[1], end[1], 100)
    
    # Draw line
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2, zorder=1)
    
    # Add symbol in middle
    mid = (start + end) / 2
    if type == 'L': # Coil symbol approximation
        circle = plt.Circle(mid, 0.05, color='white', ec=color, lw=2, zorder=2)
        ax.add_patch(circle)
        # Squiggle
        ax.text(mid[0], mid[1], 'UUU', color=color, ha='center', va='center', 
                fontsize=8, rotation=0, weight='bold', zorder=3)
    elif type == 'C': # Capacitor plates
        ax.plot([mid[0]-0.05, mid[0]+0.05], [mid[1], mid[1]], color='white', lw=10, zorder=2) # Gap
        ax.plot([mid[0]-0.02, mid[0]-0.02], [mid[1]-0.05, mid[1]+0.05], color=color, lw=2, zorder=3)
        ax.plot([mid[0]+0.02, mid[0]+0.02], [mid[1]-0.05, mid[1]+0.05], color=color, lw=2, zorder=3)
        
    if label:
        ax.text(mid[0], mid[1]+0.1, label, color=color, fontsize=10, ha='center', fontweight='bold')

def visualize_helium_circuit():
    print("Generating Helium-4 Equivalent Circuit Diagram...")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Nodes (Flattened Tetrahedron layout)
    # P1 (Top), P2 (Center), N1 (Left), N2 (Right)
    nodes = {
        'P1': np.array([0.5, 0.9]),
        'N1': np.array([0.1, 0.1]),
        'N2': np.array([0.9, 0.1]),
        'P2': np.array([0.5, 0.4])
    }
    
    colors = {'P1': 'red', 'P2': 'red', 'N1': 'blue', 'N2': 'blue'}
    
    # Draw Coupling Lines (The Strong Force Gluon Bridges)
    # In a tetrahedron, everyone connects to everyone.
    from itertools import combinations
    for n1_name, n2_name in combinations(nodes.keys(), 2):
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        draw_component(ax, p1, p2, type='L', color='gold', label='$M_{ij}$')

    # Draw Nodes (The Nucleons)
    for name, pos in nodes.items():
        # Draw the LC Tank for the nucleon itself
        # Inductor to Ground
        ground = pos - np.array([0, 0.15])
        draw_component(ax, pos, ground, type='L', color=colors[name])
        
        # Capacitor to Ground (offset slightly)
        cap_start = pos + np.array([0.03, 0])
        cap_end = ground + np.array([0.03, 0])
        draw_component(ax, cap_start, cap_end, type='C', color=colors[name])
        
        # Node Circle
        circle = plt.Circle(pos, 0.04, color=colors[name], zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], name, color='white', ha='center', va='center', weight='bold', zorder=11)
        
        # Ground symbol
        ax.plot([ground[0]-0.05, ground[0]+0.05], [ground[1], ground[1]], 'k-', lw=2)
        ax.plot([ground[0]-0.03, ground[0]+0.03], [ground[1]-0.02, ground[1]-0.02], 'k-', lw=2)
        ax.plot([ground[0]-0.01, ground[0]+0.01], [ground[1]-0.04, ground[1]-0.04], 'k-', lw=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')
    ax.set_title("Helium-4: Equivalent Lumped-Element Circuit\n(4 Coupled LC Oscillators)", fontsize=16)
    
    # Legend/Explanation
    plt.figtext(0.5, 0.05, 
                "Nodes (P/N): Resonant Tanks ($L_{mass} || C_{vac}$)\n"
                "Gold Links: Mutual Inductance ($M_{ij}$) representing Strong Force Tension.\n"
                "The tetrahedral mesh topology creates a perfect lossless quadrupole resonator.",
                ha="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig("assets/figures/helium_circuit_diagram.png", dpi=300)
    print("Circuit diagram saved to assets/figures/helium_circuit_diagram.png")
    plt.close()

if __name__ == "__main__":
    visualize_helium_circuit()