"""
AVE 3D Topological Protein Folding Engine
=========================================
Demonstrates how 1D local SPICE impedance (Z_topo) translates directly into 
3D spatial driving potentials, collapsing a random continuous backbone coil 
into deterministic secondary structures via gradient descent, without the NP-hard 
rotational search space of classic molecular dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Aesthetic params
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['grid.alpha'] = 0.2

class AVE3DFolder:
    def __init__(self, sequence: str, lr=0.001, steps=2000):
        self.sequence = list(sequence)
        self.N = len(self.sequence)
        self.lr = lr
        self.steps = steps
        
        # 3.8 A is the standard distance between sequential alpha-carbons
        self.bond_length_target = 3.8 
        self.k_bond = 50.0  # High stiffness Hooke constant for backbone integrity
        
        # Determine the target vectors based on the sequence's local AC impedance
        self.Z_map = {
            'A': 0.8,  # Ala: Low Drag, Helix 
            'L': 0.9,  # Leu: Low Drag
            'K': 0.95, # Lys: Flexible
            'E': 0.95, # Glu: Flexible
            'G': 4.5,  # Gly: Highly mismatched
            'V': 3.8,  # Val: Branched, Sheet
            'P': 5.0,  # Pro: Rigid Kink
            'S': 2.5,  # Ser: Polar disruption
            'C': 1.1   # Cys: Moderate
        }
        
        # Initialize sequence as a straight line with some random noise (an unfolded extended coil)
        self.coords = np.zeros((self.N, 3))
        self.coords[:, 2] = np.arange(self.N) * self.bond_length_target
        # Add a little jitter so derivatives aren't exactly zero
        self.coords += np.random.normal(scale=0.5, size=(self.N, 3))
        
        # To record the collapse trace
        self.history = [self.coords.copy()]

    def _get_z(self, i):
        aa = self.sequence[i]
        return self.Z_map.get(aa, 1.5)

    def _step(self):
        forces = np.zeros_like(self.coords)
        
        # 1. Backbone Integrity (Hooke Springs)
        for i in range(self.N - 1):
            vec = self.coords[i+1] - self.coords[i]
            r = np.linalg.norm(vec)
            if r > 0.001:
                f_mag = self.k_bond * (r - self.bond_length_target)
                f_dir = vec / r
                forces[i] += f_mag * f_dir
                forces[i+1] -= f_mag * f_dir

        # 2. Topological Strain Potentials (The AVE driver)
        # We apply torques/forces based on the local Z_topo.
        # - Low Z (< 1.0) means the local environment wants to curl into the 
        #   smooth, continuous helical basis state (low AC wave reflection).
        # - High Z (> 1.0) creates a massive local mismatch penalty if it tries to 
        #   curl (steric clash = high capacitance), so it violently flattens out (Beta strand).
        
        for i in range(1, self.N - 1):
            # We look at the triplet (i-1, i, i+1) to define the local bend angle
            v1 = self.coords[i] - self.coords[i-1]
            v2 = self.coords[i+1] - self.coords[i]
            
            # Normalize
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            
            if n1 < 0.1 or n2 < 0.1: continue
            
            u1 = v1 / n1
            u2 = v2 / n2
            
            # Dot product (cosine of the angle between sequential vectors)
            cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
            
            # Ideal Alpha-Helix angle for alpha-carbons is ~100 degrees = ~1.74 rad
            # (which means the vectors dot to about cos(pi - 1.74) = cos(1.4) = 0.17)
            target_cos_helix = 0.5 
            
            # Ideal Beta-Sheet is extended zig-zag, vectors are nearly parallel 
            # (in the backbone progression sense), cos ~ 0.8
            target_cos_sheet = 0.8 
            
            Z = self._get_z(i)
            
            if Z <= 1.0:
                # Helix Driver: Pulls the dot product toward target_cos_helix
                # The torque is proportional to the deviation.
                diff = cos_theta - target_cos_helix
                k_bend = 5.0 * (1.0 / Z)  # Stronger drive the better the match
            else:
                # Sheet Driver: Pulls the dot product toward target_cos_sheet to flatten it
                diff = cos_theta - target_cos_sheet
                k_bend = 10.0 * Z  # High impedance actively punishes bending
                
            # Gradient of the dot product potential: U = 0.5 * k * (cos_theta - target)^2
            # dU/du2 = k * diff * u1. We apply force to coords[i+1] along u1.
            
            bend_force_1 = -k_bend * diff * u2 
            bend_force_2 = -k_bend * diff * u1 
            
            # Normalize to prevent explosion, apply strictly as directional vectors
            f1_norm = np.linalg.norm(bend_force_1)
            f2_norm = np.linalg.norm(bend_force_2)
            if f1_norm > 0: bend_force_1 = bend_force_1 / f1_norm * min(f1_norm, 10.0)
            if f2_norm > 0: bend_force_2 = bend_force_2 / f2_norm * min(f2_norm, 10.0)
            
            forces[i-1] += bend_force_1
            forces[i+1] += bend_force_2
            # Balance on center node
            forces[i] -= (bend_force_1 + bend_force_2)
            
            # 3. Handedness (Chirality) - Alpha-helices need a specific right-handed twist
            # We apply a small cross-product torque if Z is helical
            if Z <= 1.0 and i < self.N - 2:
                v3 = self.coords[i+2] - self.coords[i+1]
                u3 = v3 / max(np.linalg.norm(v3), 0.1)
                
                # Desired binormal vector for right-handed helix
                cross_prod = np.cross(u1, u2)
                twist_force = 2.0 * np.cross(cross_prod, u3)
                
                forces[i+2] -= twist_force
                forces[i+1] += twist_force

        # Force clipping to prevent NaN explosions
        forces = np.clip(forces, -20.0, 20.0)
        
        # Update positions
        self.coords += self.lr * forces
        
        # Center the molecule
        center_of_mass = np.mean(self.coords, axis=0)
        self.coords -= center_of_mass
        
        self.history.append(self.coords.copy())

    def simulate(self):
        print(f"Simulating collapse for sequence: {str().join(self.sequence)}")
        for step in range(self.steps):
            self._step()
            if step % 100 == 0:
                print(f"  Step {step}/{self.steps}")
        print("Collapse complete.")

def create_animation(engine, title, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-calculate limits to keep camera steady
    all_coords = np.concatenate(engine.history)
    max_range = np.max(np.abs(all_coords)) * 1.1
    
    # Determine color based on final state dominance
    avg_Z = np.mean([engine._get_z(i) for i in range(engine.N)])
    color = '#00ffff' if avg_Z < 1.0 else '#ff00ff'
    
    line, = ax.plot([], [], [], color=color, lw=3, marker='o', markersize=4)
    
    def init():
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_title(title, color='white')
        
        # Hide axes for clean look
        ax.set_axis_off()
        return line,

    def update(frame):
        coords = engine.history[frame]
        line.set_data(coords[:, 0], coords[:, 1])
        line.set_3d_properties(coords[:, 2])
        # Auto-rotate camera slightly
        ax.view_init(elev=20, azim=frame * 0.5)
        return line,

    # Render video
    print(f"Rendering animation to {filename}...")
    # Record every 50th frame to keep GIF size manageable over 10000 steps
    frames_to_render = list(np.arange(0, len(engine.history), 50))
    
    # Hold the final converged frame for 3 seconds (90 frames at 30 fps)
    final_frame = len(engine.history) - 1
    frames_to_render.extend([final_frame] * 90)
    
    ani = animation.FuncAnimation(fig, update, frames=frames_to_render, init_func=init, blit=False)
    
    writer = animation.PillowWriter(fps=30)
    ani.save(filename, writer=writer)
    plt.close()
    print(f"Saved {filename}")

def create_progression_figure(engine, title, filename):
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle(title, fontsize=16, color='white')
    
    # Select 4 frames representing the collapse
    total_frames = len(engine.history)
    frame_indices = [0, int(total_frames * 0.1), int(total_frames * 0.3), total_frames - 1]
    labels = ["t = 0 (Random Coil)", "t = 1000", "t = 3000", "t = Final (Converged)"]
    
    all_coords = np.concatenate(engine.history)
    max_range = np.max(np.abs(all_coords)) * 1.1
    
    avg_Z = np.mean([engine._get_z(i) for i in range(engine.N)])
    color = '#00ffff' if avg_Z < 1.0 else '#ff00ff'
    
    for i, (f_idx, label) in enumerate(zip(frame_indices, labels)):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        coords = engine.history[f_idx]
        
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, lw=3, marker='o', markersize=4)
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_title(label, color='white', fontsize=12)
        ax.set_axis_off()
        
        # Standardize the viewing angle to see the structure develop
        ax.view_init(elev=20, azim=45)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"Saved static progression to {filename}")

if __name__ == '__main__':
    # Define output directory
    def _find_repo_root():
        d = os.path.dirname(os.path.abspath(__file__))
        while d != os.path.dirname(d):
            if os.path.exists(os.path.join(d, "pyproject.toml")):
                return d
            d = os.path.dirname(d)
        return os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(_find_repo_root(), "assets", "sim_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Simulation 1: The Helix Collapse
    seq_alpha = "EAAAKAAAAAAKAAAAAAAK"
    engine_a = AVE3DFolder(seq_alpha, steps=10000, lr=0.01)
    engine_a.simulate()
    
    create_progression_figure(engine_a, "Topological Gradient Descent: Sequence A (Alpha-Helix Collapse)", 
                              os.path.join(output_dir, "ave_helix_progression.png"))
    create_animation(engine_a, "Deterministic Helix Collapse (AVE)", 
                     os.path.join(output_dir, "ave_helix_collapse.gif"))

    # Simulation 2: The Beta-Strand Flattening
    seq_beta = "VGVGVGVGVGVGVGVGVGVG"
    engine_b = AVE3DFolder(seq_beta, steps=10000, lr=0.01)
    engine_b.simulate()
    
    create_progression_figure(engine_b, "Topological Gradient Descent: Sequence B (Beta-Sheet Flare)", 
                              os.path.join(output_dir, "ave_sheet_progression.png"))
    create_animation(engine_b, "Deterministic Beta-Strand Flare (AVE)", 
                     os.path.join(output_dir, "ave_sheet_collapse.gif"))
