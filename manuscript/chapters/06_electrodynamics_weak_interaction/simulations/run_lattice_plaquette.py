import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_lattice_plaquette():
    print("Simulating U(1) Lattice Gauge Plaquette...")
    
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    nodes = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], color='white', s=150, zorder=5)
    
    colors = ['#00ffcc', '#00ffcc', '#00ffcc', '#00ffcc']
    labels = [r'$U_{ij}$', r'$U_{jk}$', r'$U_{kl}$', r'$U_{li}$']
    for i in range(4):
        start = nodes[i]
        end = nodes[(i+1)%4]
        ax.quiver(start[0], start[1], start[2], end[0]-start[0], end[1]-start[1], end[2]-start[2], 
                  color=colors[i], arrow_length_ratio=0.15, linewidth=3)
        mid = (start + end) / 2
        offset_x = 0.1 if i in [0, 2] else -0.15
        offset_y = 0.1 if i in [1, 3] else -0.15
        ax.text(mid[0] + offset_x, mid[1] + offset_y, mid[2], labels[i], color='white', fontsize=14, weight='bold')

    ax.quiver(0.5, 0.5, 0, 0, 0, 1.2, color='#ff3366', linewidth=5, arrow_length_ratio=0.2)
    ax.text(0.5, 0.5, 1.3, r'$F_{\mu\nu} = \nabla \times A$', color='#ff3366', fontsize=16, weight='bold', ha='center')

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(0, 1.5)
    ax.axis('off')
    
    ax.text2D(0.5, 0.95, r"U(1) Lattice Plaquette ($U_P$)", transform=ax.transAxes, color='white', fontsize=16, weight='bold', ha='center')
    textstr = r"$U_P = U_{ij}U_{jk}U_{kl}U_{li} \approx \exp(i a^2 F_{\mu\nu})$"
    ax.text2D(0.5, 0.88, textstr, transform=ax.transAxes, color='#00ffcc', fontsize=14, ha='center')
    
    filepath = os.path.join(OUTPUT_DIR, "lattice_plaquette.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_lattice_plaquette()