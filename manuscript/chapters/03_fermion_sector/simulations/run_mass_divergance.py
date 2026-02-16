import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_mass_divergence():
    print("Simulating Dielectric Mass Divergence...")
    
    v_ratio = np.linspace(0, 0.9999, 1000)
    mass_profile = 1.0 / np.sqrt(1 - v_ratio**4)
    
    m_e = 1.0
    m_mu = 105.66 / 0.511
    m_tau = 1776.86 / 0.511
    
    v_e = 0.15 
    m_e_vis = 1.0 / np.sqrt(1 - v_e**4)
    v_mu = (1 - (1/m_mu)**2)**0.25
    v_tau = (1 - (1/m_tau)**2)**0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(v_ratio, mass_profile, color='#00ffcc', lw=3, label=r'Mass Divergence $\left(1 - (\Delta\phi/V_0)^4\right)^{-1/2}$')
    
    ax.scatter([v_e], [m_e_vis], color='white', s=100, zorder=5, edgecolor='black')
    ax.scatter([v_mu], [m_mu], color='white', s=100, zorder=5, edgecolor='black')
    ax.scatter([v_tau], [m_tau], color='white', s=100, zorder=5, edgecolor='black')
    
    ax.annotate('Electron ($3_1$)\nLow Strain', (v_e, m_e_vis), xytext=(15, -5), textcoords='offset points', color='white')
    ax.annotate('Muon ($5_1$)\nDielectric Stress', (v_mu, m_mu), xytext=(-110, 10), textcoords='offset points', color='white')
    ax.annotate('Tau ($7_1$)\nNear-Rupture', (v_tau, m_tau), xytext=(-100, -10), textcoords='offset points', color='white')
    
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label=r'Schwinger Yield Limit ($V_0$)')
    ax.fill_betweenx([0.1, 10000], 1.0, 1.05, color='red', alpha=0.1)
    ax.text(1.01, 10, 'Lattice Rupture\n(Pair Production)', color='red', rotation=90, va='center', fontweight='bold')
    
    ax.set_yscale('log')
    ax.set_ylim(0.8, 10000)
    ax.set_xlim(0, 1.05)
    
    ax.set_xlabel(r'Local Flux Crowding / Lattice Strain ($\Delta\phi / V_0$)', fontsize=12, color='white')
    ax.set_ylabel(r'Stored Inductive Mass ($m / m_e$)', fontsize=12, color='white')
    ax.set_title('Lepton Mass Hierarchy via Dielectric Saturation', fontsize=14, pad=15, color='white')
    
    ax.grid(True, which="both", ls="--", color='#333333', alpha=0.7)
    ax.legend(loc='upper left', facecolor='black', edgecolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "dielectric_mass_resonance.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    simulate_mass_divergence()