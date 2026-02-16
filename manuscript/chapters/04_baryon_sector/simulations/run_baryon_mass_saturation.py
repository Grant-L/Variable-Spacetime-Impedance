import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/04_baryon_sector/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_baryon_mass_saturation():
    print("Simulating Baryon Mass via Dielectric Saturation...")
    
    v_ratio = np.linspace(0, 0.9999, 1000)
    mass_profile = 1.0 / np.sqrt(1 - v_ratio**4)
    
    m_e = 1.0
    m_mu = 105.66 / 0.511
    m_p = 938.27 / 0.511  # Proton Mass (~1836)
    
    v_e = 0.15 
    m_e_vis = 1.0 / np.sqrt(1 - v_e**4)
    v_mu = (1 - (1/m_mu)**2)**0.25
    v_p = (1 - (1/m_p)**2)**0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(v_ratio, mass_profile, color='#ffcc00', lw=3, label=r'Mass Divergence $\left(1 - (\Delta\phi/V_0)^4\right)^{-1/2}$')
    
    ax.scatter([v_e], [m_e_vis], color='white', s=100, zorder=5)
    ax.scatter([v_mu], [m_mu], color='white', s=100, zorder=5)
    ax.scatter([v_p], [m_p], color='#ff3366', s=150, zorder=5, edgecolor='white')
    
    ax.annotate('Electron ($3_1$)\nSingle Loop', (v_e, m_e_vis), xytext=(15, -5), textcoords='offset points', color='white')
    ax.annotate('Proton ($6^3_2$)\nTri-Loop Frustration\n(m ~ 1836)', (v_p, m_p), xytext=(-130, 10), textcoords='offset points', color='#ff3366', weight='bold')
    
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label=r'Schwinger Yield Limit ($V_0$)')
    ax.fill_betweenx([0.1, 10000], 1.0, 1.05, color='red', alpha=0.1)
    
    ax.set_yscale('log')
    ax.set_ylim(0.8, 10000)
    ax.set_xlim(0, 1.05)
    
    ax.set_xlabel(r'Local Flux Crowding / Lattice Strain ($\Delta\phi / V_0$)', fontsize=12, color='white')
    ax.set_ylabel(r'Stored Inductive Mass ($m / m_e$)', fontsize=12, color='white')
    ax.set_title('Unification of Lepton and Baryon Masses via Saturation', fontsize=14, pad=15, color='white')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "baryon_mass_saturation.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_baryon_mass_saturation()