import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/01_discrete_amorphous_manifold/simulations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_action_scale():
    print("Simulating True Hardware Action Scale...")
    scales = [1.6e-35, 3.74e-19, 1e-18, 3.86e-13]
    labels = ['Planck "Illusion"\n(Macro Gravity Artifact)', 'AVE Lattice Pitch ($l_{node}$)\n(Hardware Rupture Limit)', 'Weak Force Cutoff ($l_c$)\n(Cosserat Evanescence)', 'Electron ($3_1$ Soliton)\n(Macroscopic Defect)']
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    y_pos = [1, 2, 2.5, 3.5]
    colors = ['#ff3366', '#00ffcc', '#ffff00', '#0099ff']
    
    for i in range(4):
        ax.plot(scales[i], y_pos[i], marker='o', markersize=15, color=colors[i])
        ax.axhline(y_pos[i], color=colors[i], alpha=0.3, linestyle='--')
        ax.text(scales[i] * 1.5, y_pos[i] + 0.15, labels[i], color=colors[i], fontsize=11, weight='bold')

    ax.set_xscale('log'); ax.set_xlim(1e-36, 1e-11); ax.set_ylim(0.5, 4.5); ax.set_yticks([])
    ax.set_xlabel('Spatial Scale (Meters)', color='white', fontsize=12)
    ax.set_title('The True Granularity of the Spacetime Substrate', color='white', fontsize=14, weight='bold', pad=15)
    ax.grid(True, which="major", ls="--", color='#333333', alpha=0.7); ax.tick_params(axis='x', colors='white')
    
    filepath = os.path.join(OUTPUT_DIR, "hardware_action_scale.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_action_scale()