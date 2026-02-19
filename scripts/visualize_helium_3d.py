import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualize_helium_forces():
    print("==========================================================")
    print("   AVE HELIUM-4 TOPOLOGY: TETRAHEDRAL FORCE MODEL")
    print("==========================================================")
    
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # --- 1. GEOMETRY: The Tetrahedral Nucleus ---
    # Nucleons are at vertices of a tetrahedron
    s = 1.0 # Scale factor
    vertices = np.array([
        [ s,  s,  s], # Proton 1
        [ s, -s, -s], # Proton 2
        [-s,  s, -s], # Neutron 1
        [-s, -s,  s]  # Neutron 2
    ])
    
    colors = ['red', 'red', 'cyan', 'cyan'] # P, P, N, N
    labels = ['p+', 'p+', 'n0', 'n0']
    
    # --- 2. RENDER NUCLEONS (The Borromean Cores) ---
    # We render them as dense wireframe spheres to represent the 
    # saturated Borromean knot core (The Mass)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    print("Rendering Nucleon Cores...")
    for i, pos in enumerate(vertices):
        x = 0.4 * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = 0.4 * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z = 0.4 * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
        
        # Plot wireframe to show it's a "knot" not a solid
        ax.plot_wireframe(x, y, z, color=colors[i], alpha=0.4, rstride=2, cstride=2)
        
        # Core Glow
        ax.scatter(pos[0], pos[1], pos[2], color=colors[i], s=200, alpha=0.8)
        
        # Label
        ax.text(pos[0], pos[1], pos[2]+0.6, labels[i], color='white', fontsize=14, weight='bold', ha='center')

    # --- 3. RENDER THE STRONG FORCE (Flux Tube Tension) ---
    # The bonds are not sticks; they are tensioned flux tubes (Pion Exchange).
    # We render tubes connecting every nucleon pair.
    
    from itertools import combinations
    indices = range(len(vertices))
    
    print("Rendering Vacuum Flux Tension (Gluon Field)...")
    for i, j in combinations(indices, 2):
        p1 = vertices[i]
        p2 = vertices[j]
        
        # Vector along bond
        vector = p2 - p1
        length = np.linalg.norm(vector)
        direction = vector / length
        
        # Parametric Tube Generation
        t = np.linspace(0, length, 30)
        theta = np.linspace(0, 2*np.pi, 12)
        T, Theta = np.meshgrid(t, theta)
        
        # Coordinate frame for tube (Orthogonal vectors)
        # Arbitrary vector not parallel to direction
        not_v = np.array([1, 0, 0])
        if np.abs(np.dot(direction, not_v)) > 0.9:
            not_v = np.array([0, 1, 0])
            
        n1 = np.cross(direction, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(direction, n1)
        
        # Tube Radius varies: Thicker at nodes (low stress), Thinner in middle (High Tension)
        # This visualizes the "Flux Constriction" of the Strong Force
        r_tube = 0.15 * (1 - 0.3 * np.sin(np.pi * T / length)) 
        
        # Construct Tube Surface
        X = p1[0] + direction[0]*T + r_tube * (n1[0]*np.cos(Theta) + n2[0]*np.sin(Theta))
        Y = p1[1] + direction[1]*T + r_tube * (n1[1]*np.cos(Theta) + n2[1]*np.sin(Theta))
        Z = p1[2] + direction[2]*T + r_tube * (n1[2]*np.cos(Theta) + n2[2]*np.sin(Theta))
        
        # Color based on "Tension" (Gold = High Energy Density)
        ax.plot_surface(X, Y, Z, color='gold', alpha=0.3, shade=True)

    # --- 4. FORMATTING ---
    ax.set_axis_off()
    
    # Set equal limits to prevent distortion
    limit = 1.5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    plt.title("Helium-4 Topology: The Tetrahedral Borromean Braid\n(Binding Energy = Flux Tube Tension)", 
              color='white', fontsize=16)
    
    # Add caption in plot
    plt.figtext(0.5, 0.05, 
                "The 4 nucleons pack into a tetrahedron to minimize surface tension.\n"
                "Gold tubes represent the high-tension vacuum flux (Strong Force) holding them together.\n"
                "This geometry explains the extreme stability (28 MeV binding energy) of the Alpha particle.",
                ha="center", color="gray", fontsize=10)

    output_file = "assets/figures/helium_force_model.png"
    plt.savefig(output_file, dpi=300, facecolor='black')
    print(f"3D Force Model saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    visualize_helium_forces()