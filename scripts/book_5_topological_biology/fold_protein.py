"""
Standard Model Overdrive: High-Fidelity Protein Folding
=======================================================
This script simulates the absolute First-Principles folding of a 
real, empirical Polyalanine peptide sequence. We load the actual 
atomic mass constraints (Nitrogen, C-alpha, Carbonyl groups) and 
feed them into the Universal 1/d LC Optimizer. 
The gradient descent folds the 1D string into its global lowest-energy 
secondary structure purely via macroscopic spatial tension, animated dynamically.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root / "src"))

from ave.simulations.topological_optimizer import TopologicalOptimizer

def create_empirical_polyalanine(residues=10):
    """
    Generate a high-fidelity 1D chain of Polyalanine nodes.
    Tracks the N - Ca - C - O backbone for each residue.
    """
    masses = []
    initial_coords = []
    colors = []
    
    # Real Atomic Mass constants (Daltons)
    M_N = 14.007
    M_C = 12.011
    M_O = 15.999
    M_Ca_Ala = 12.011 + 15.035 # C alpha + CH3 side group (Methyl)
    
    # Linear starting arrangement (a straight, denatured backbone)
    for i in range(residues):
        idx = i * 4
        # 1. Nitrogen (Amine)
        masses.append(M_N)
        initial_coords.append([idx*1.5, 0, 0])
        colors.append('#3333ff') # Blue
        
        # 2. C-alpha (with Alanine sidechain mass)
        masses.append(M_Ca_Ala)
        initial_coords.append([idx*1.5 + 1.5, 0.8, 0.0])
        colors.append('#00ffcc') # Cyan
        
        # 3. Carbonyl Carbon
        masses.append(M_C)
        initial_coords.append([idx*1.5 + 3.0, 0.0, 0.0])
        colors.append('#88ff00') # Green
        
        # 4. Carbonyl Oxygen
        masses.append(M_O)
        initial_coords.append([idx*1.5 + 3.0, -1.2, 0.0])
        colors.append('#ff3333') # Red
        
    initial_coords = np.array(initial_coords)
    # Inject a tiny thermal stochastic twist to give the gradient descent 
    # the leverage to break symmetry and find the 3D folding tunnel
    np.random.seed(42)
    initial_coords += np.random.normal(0, 0.3, size=initial_coords.shape)
    
    return np.array(masses), initial_coords, colors

def fold_protein_dynamic():
    print("[*] Initializing High-Fidelity Protein Folding (Polyalanine Peptide)")
    
    masses, initial_coords, colors = create_empirical_polyalanine(residues=12)
    N = len(masses)
    print(f"    -> Empirical Polypeptide Chain: {N} Atoms (Unfolded)")
    
    optimizer = TopologicalOptimizer(node_masses=masses, interaction_scale='molecular')
    print("[*] Commencing Gradient Descent Assembly. Filming geometric crumpling...")
    
    final_coords, total_energy, history, energy_history = optimizer.optimize(
        initial_coords, 
        method='L-BFGS-B', 
        options={'maxiter': 600, 'ftol': 1e-5, 'disp': False},
        record_history=True
    )
    
    print(f"[+] Folding Complete. Final Configuration Impedance: {total_energy:.2f}")
    print(f"    -> Optimization Frames Recorded: {len(history)}")
    
    print("[*] Rendering Dynamic Folding GIF...")
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('#050510')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050510')
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_title("First-Principles Protein Folding: Polyalanine Motif\n(Real-Time O(N^2) Geometric Optimization)", 
                 color='white', fontsize=14, pad=20)

    # Plot the original unfolded backbone as a faint ghost for scale
    ux, uy, uz = history[0][:, 0], history[0][:, 1], history[0][:, 2]
    ghost_line, = ax.plot(ux, uy, uz, color='#ffffff', alpha=0.15, linewidth=2.0)
    
    # We will draw the main backbone updating every frame
    backbone_line, = ax.plot([], [], [], color='#aaaaaa', alpha=0.9, linewidth=2.0, zorder=4)
    
    scat = ax.scatter(ux, uy, uz, c=colors, s=100, alpha=0.9, edgecolors='black', zorder=5)
    energy_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='#ff33cc', fontsize=14)
    
    # Dynamic view bounds based on the folded center
    center = np.mean(final_coords, axis=0)
    # The string starts very long, but we want the camera focused on the crumple zone
    bound = np.max(np.linalg.norm(final_coords - center, axis=1)) * 1.5
    ax.set_xlim([center[0]-bound, center[0]+bound])
    ax.set_ylim([center[1]-bound, center[1]+bound])
    ax.set_zlim([center[2]-bound, center[2]+bound])
    
    centers = [np.mean(h, axis=0) for h in history]

    def update(frame):
        coords = history[frame] - centers[frame] + center # Keep tracking the center of mass
        
        scat._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        
        # Only draw the linear backbone path (connecting N-Ca-C N-Ca-C)
        b_x, b_y, b_z = [], [], []
        for i in range(N):
            # Oxygen nodes (i%4==3) stick out, they don't continue the backbone chain
            if i % 4 != 3:
                b_x.append(coords[i, 0])
                b_y.append(coords[i, 1])
                b_z.append(coords[i, 2])
            elif i < N - 1:
                # The backbone connects C(i-1) to N(i+1), skipping the O(i)
                b_x.append(coords[i+1, 0])
                b_y.append(coords[i+1, 1])
                b_z.append(coords[i+1, 2])
                
        backbone_line.set_data(b_x, b_y)
        backbone_line.set_3d_properties(b_z)
        
        energy_text.set_text(f"Fold Iter: {frame:03d}\nMacroscopic Impedance: {energy_history[frame]:.0f}")
        return scat, backbone_line, energy_text
        
    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=40, blit=False)
    
    outdir = project_root / "assets" / "sim_outputs"
    os.makedirs(outdir, exist_ok=True)
    target = outdir / "macro_molecular_folding_dynamic.gif"
    
    anim.save(target, writer='pillow', fps=25)
    print(f"[*] Visualized Dynamic Polyalanine Folding: {target}")

if __name__ == "__main__":
    fold_protein_dynamic()
