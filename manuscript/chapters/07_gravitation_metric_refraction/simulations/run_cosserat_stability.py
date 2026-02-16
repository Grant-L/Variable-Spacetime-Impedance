import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/07_gravitation_metric_refraction/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_thermodynamic_stability():
    print("Simulating Cauchy vs Cosserat Bulk Modulus Stability...")
    
    mu_shear = np.linspace(0.1, 5, 100)
    
    # In classical Cauchy Aether, to remove longitudinal waves (v_L = 0)
    # lambda + 2*mu = 0 => lambda = -2*mu
    # K = lambda + (2/3)*mu = -2*mu + (2/3)*mu = -4/3 * mu
    K_cauchy = -(4/3) * mu_shear
    
    # In AVE Cosserat Aether, transverse waves are carried by microrotation (\gamma_c)
    # Therefore, lambda and mu can be massive positive numbers (Incompressible)
    lambda_c = 10.0
    K_cosserat = lambda_c + (2/3) * mu_shear
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(mu_shear, K_cosserat, color='#00ffcc', lw=3, label=r'Cosserat Substrate ($K > 0$): Stable')
    ax.plot(mu_shear, K_cauchy, color='#ff3366', lw=3, linestyle='--', label=r'Cauchy Aether ($K = -\frac{4}{3}\mu_{shear}$): Implosion')
    
    ax.axhline(0, color='white', lw=1)
    ax.fill_between(mu_shear, 0, 15, color='#00ffcc', alpha=0.1)
    ax.fill_between(mu_shear, -8, 0, color='#ff3366', alpha=0.1)
    
    ax.text(2.5, 5, "Thermodynamically Stable Universe\n(Transverse waves carried by Spin $\\theta$)", color='white', ha='center', weight='bold')
    ax.text(2.5, -4, "Singularity / Implosion\n(Negative Bulk Modulus)", color='white', ha='center', weight='bold')
    
    ax.set_ylim(-8, 15)
    ax.set_xlim(0.1, 5)
    ax.set_xlabel(r'Macroscopic Shear Modulus ($\mu_{shear}$)', fontsize=12, color='white')
    ax.set_ylabel(r'Bulk Modulus ($K = \lambda + \frac{2}{3}\mu_{shear}$)', fontsize=12, color='white')
    ax.set_title('Resolving the Aether Implosion Paradox', fontsize=14, pad=15, color='white', weight='bold')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "cosserat_stability.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_thermodynamic_stability()