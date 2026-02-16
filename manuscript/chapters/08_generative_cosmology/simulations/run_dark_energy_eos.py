import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_dark_energy_eos():
    print("Simulating Thermodynamic Equation of State (w=-1)...")
    
    # Volume growing exponentially
    time = np.linspace(0, 5, 100)
    H_0 = 0.5
    Volume = np.exp(3 * H_0 * time)
    
    # Standard Matter: Energy dilutes (w = 0)
    # Total Energy U is constant, Density rho = U/V
    U_matter = np.full_like(Volume, 100.0)
    rho_matter = U_matter / Volume
    
    # AVE Lattice Genesis (Dark Energy): Density is constant (w = -1)
    # Total Energy U grows with Volume
    rho_vac = 20.0
    U_vac = rho_vac * Volume
    
    # Thermodynamic Pressure P = - dU/dV
    P_matter = -np.gradient(U_matter, Volume) # Evaluates to 0
    P_vac = -np.gradient(U_vac, Volume)       # Evaluates to -rho_vac
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    
    # Left Plot: Energy Density
    ax1.set_facecolor('#050508')
    ax1.plot(Volume, rho_matter, color='#ffff00', lw=3, label=r'Baryonic Matter ($\rho \propto V^{-1}$)')
    ax1.plot(Volume, np.full_like(Volume, rho_vac), color='#00ffcc', lw=3, label=r'Lattice Genesis ($\rho = constant$)')
    ax1.set_xlabel('Cosmic Volume ($V$)', color='white')
    ax1.set_ylabel(r'Energy Density ($\rho$)', color='white')
    ax1.set_title('Density vs Expansion', color='white', weight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, ls="--", color='#333333', alpha=0.7)
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    # Right Plot: Equation of State
    ax2.set_facecolor('#050508')
    ax2.plot(Volume, P_matter / rho_matter, color='#ffff00', lw=3, label=r'Matter: $w = 0$')
    ax2.plot(Volume, P_vac / rho_vac, color='#ff3366', lw=3, label=r'AVE Vacuum: $w = -1$')
    ax2.set_ylim(-1.5, 0.5)
    ax2.set_xlabel('Cosmic Volume ($V$)', color='white')
    ax2.set_ylabel(r'Equation of State ($w = P/\rho$)', color='white')
    ax2.set_title('Thermodynamic Pressure (1st Law)', color='white', weight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, ls="--", color='#333333', alpha=0.7)
    ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    textstr = r"$\mathbf{1st\ Law\ of\ Thermodynamics:}$" + "\n" + r"$dU = -P dV$" + "\n\n" + r"Because Lattice Genesis creates volume" + "\n" + r"at constant density ($U = \rho V$):" + "\n" + r"$d(\rho V) = -P dV \Rightarrow P = -\rho$"
    ax2.text(0.1, 0.2, textstr, transform=ax2.transAxes, color='white', fontsize=11, bbox=dict(facecolor='black', edgecolor='#ff3366', alpha=0.8, pad=8))
    
    filepath = os.path.join(OUTPUT_DIR, "dark_energy_eos.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    ensure_output_dir()
    simulate_dark_energy_eos()