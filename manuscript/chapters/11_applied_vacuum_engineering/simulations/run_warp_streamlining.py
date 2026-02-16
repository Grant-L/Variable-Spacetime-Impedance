import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_warp_streamlining():
    print("Simulating Relativistic Bow Shock and Metric Streamlining...")
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-3, 3, 200))
    hull = (X**2 / 1.5**2 + Y**2 / 0.8**2) <= 1.0  
    v_c = 0.9
    mach_angle = np.arcsin(1.0 / (v_c / 0.5)) 
    shock_std = np.exp(-10 * (X - -1.5 - np.abs(Y) / np.tan(mach_angle))**2) * (X < -1.5)
    beam = np.exp(-5 * (X + 3)**2 - 20 * Y**2) * (X < -1.5)
    shock_stream = shock_std * (1.0 - 0.8 * np.exp(-2 * np.abs(Y)))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150); fig.patch.set_facecolor('#050508')
    for ax in (ax1, ax2): ax.set_facecolor('#050508'); ax.set_xticks([]); ax.set_yticks([]); ax.contourf(X, Y, hull, levels=[0.5, 1.5], colors=['#888888'])
    
    ax1.contourf(X, Y, shock_std, 50, cmap='plasma', alpha=0.8); ax1.set_title('Standard Flight (v = 0.9c)\nMassive Vacuum Bow Shock (High Inertia)', color='white')
    ax2.contourf(X, Y, shock_stream, 50, cmap='plasma', alpha=0.8); ax2.contour(X, Y, beam, levels=[0.1, 0.5, 0.9], colors='#00ffcc', linestyles='dashed', linewidths=2)
    ax2.plot([-5, -1.5], [0, 0], color='#00ffcc', linestyle='--', lw=2, label='Metric Actuator Beam')
    ax2.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white'); ax2.set_title('Metric Streamlining (Active Flow Control)\nViscosity Reduced by Shear Beam (Low Inertia)', color='white')
    
    plt.tight_layout(); filepath = os.path.join(OUTPUT_DIR, "vacuum_aerodynamics.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close(); print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_warp_streamlining()