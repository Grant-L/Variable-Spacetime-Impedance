import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/07_gravitation_metric_refraction/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_tensor_lensing():
    print("Simulating Tensor Refraction vs Scalar Deflection...")
    
    b = 1.0
    z = np.linspace(-5, 5, 1000)
    
    grad_chi = b / (z**2 + b**2)**(1.5)
    
    # Scalar Theory Deflection (n = 1 + chi) -> Newtonian Bending
    delta_scalar = np.cumsum(grad_chi) * (z[1]-z[0])
    
    # Tensor Theory Deflection (n = 1 + 2*chi) -> Trace-Reversed (Einstein) Bending
    delta_tensor = np.cumsum(2 * grad_chi) * (z[1]-z[0])
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    y_scalar = b - delta_scalar * 0.2
    y_tensor = b - delta_tensor * 0.2
    
    ax.plot(z, y_scalar, color='#ff3366', lw=2, linestyle='--', label=r'Scalar Strain ($n = 1 + \chi$): Newtonian Deflection')
    ax.plot(z, y_tensor, color='#00ffcc', lw=3, label=r'Trace-Reversed Tensor Strain ($n = 1 + 2\chi$): Einstein Deflection')
    
    ax.scatter([0], [0], color='#ffff00', s=300, zorder=5, label='Topological Defect (Mass)')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Propagation Distance ($z$)', fontsize=12, color='white')
    ax.set_ylabel('Impact Parameter / Deflection ($y$)', fontsize=12, color='white')
    ax.set_title('Gravitational Lensing: Scalar vs Tensor Elasticity', fontsize=14, pad=15, color='white', weight='bold')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.tick_params(colors='white')
    ax.legend(loc='lower left', facecolor='black', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "tensor_lensing.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_tensor_lensing()