import numpy as np
import matplotlib.pyplot as plt
import os

# Directory setup
OUTPUT_DIR = "manuscript/chapters/03_fermion_sector/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_mass_hierarchy():
    """
    Simulates the AVE Lepton Mass Hierarchy using the N^9 Inductive Scaling Law
    and Dielectric Saturation.
    """
    print("Simulating Lepton Mass Hierarchy...")

    # 1. Physics Constants & Calibration
    # Base Mass (Electron, N=3)
    m_e = 0.511 # MeV
    N_e = 3.0
    
    # 2. Define the N range
    N_plot = np.linspace(1, 9.5, 200)
    particles = {
        'Electron': {'N': 3, 'Mass_Exp': 0.511, 'Color': 'blue'},
        'Muon': {'N': 5, 'Mass_Exp': 105.66, 'Color': 'green'},
        'Tau': {'N': 7, 'Mass_Exp': 1776.86, 'Color': 'red'},
        'Gen 4?': {'N': 9, 'Mass_Exp': None, 'Color': 'gray'}
    }

    # 3. Model A: Standard Inductance (N^2)
    m_std = m_e * (N_plot / N_e)**2

    # 4. Model B: AVE Ideal Geometric Resonance (N^9)
    # Eq 3.17: m_ideal(N) = (E_pair/2) * (N/3)^9 * Omega_res
    # Omega_res = 2 for Excited States (N >= 5), 1 for Ground State
    def get_omega_res(n):
        return np.where(n < 4, 1.0, 2.0)

    m_ideal = m_e * (N_plot / N_e)**9 * get_omega_res(N_plot)

    # 5. Model C: AVE Saturated Dielectric (Real Mass)
    # Omega_sat = sqrt(1 - (V/V_break)^2)
    # Calibration: Tau (N=7) saturation ratio is derived from experiment to be ~0.308
    # Scaling: V^2 scales with N^9
    saturation_ratio = 0.308 * (N_plot / 7.0)**9
    
    # Apply hard cutoff where ratio > 1 (Dielectric Breakdown)
    saturation_factor = np.sqrt(np.maximum(0, 1 - saturation_ratio))
    
    m_real = m_ideal * saturation_factor

    # 6. Plotting
    plt.figure(figsize=(10, 7))
    
    plt.plot(N_plot, m_std, '--', color='gray', label='Standard Inductance ($N^2$)', alpha=0.5)
    plt.plot(N_plot, m_ideal, ':', color='cyan', label='Ideal Geometric Resonance ($N^9$)', linewidth=1.5)
    plt.plot(N_plot, m_real, '-', color='blue', label='AVE Saturated Lattice (Predicted)', linewidth=2.5)

    for name, data in particles.items():
        n = data['N']
        if data['Mass_Exp']:
            plt.plot(n, data['Mass_Exp'], 'o', color=data['Color'], markersize=8, zorder=10)
            plt.text(n, data['Mass_Exp']*1.3, name, ha='center', fontweight='bold', color=data['Color'])
        else:
            m_hyp = m_e * (n/3)**9 * 2
            plt.plot(n, m_hyp, 'X', color=data['Color'], markersize=8, alpha=0.5)
            plt.text(n, m_hyp*0.5, "Forbidden\n(Breakdown)", ha='center', color='gray', fontsize=9)

    # Highlight Breakdown
    breakdown_mask = saturation_ratio > 1
    if np.any(breakdown_mask):
        breakdown_start = N_plot[breakdown_mask][0]
        plt.axvspan(breakdown_start, 9.5, color='red', alpha=0.1, label='Dielectric Breakdown ($V > V_{break}$)')

    plt.yscale('log')
    plt.ylim(0.1, 100000)
    plt.xlim(2, 9.5)
    plt.xlabel('Topological Winding Number ($N$)')
    plt.ylabel('Rest Mass Energy (MeV)')
    plt.title('Derivation of the Lepton Mass Hierarchy ($N^9$ Scaling)')
    plt.grid(True, which='both', linestyle='-', alpha=0.2)
    plt.legend(loc='lower right')
    
    outfile = os.path.join(OUTPUT_DIR, "mass_hierarchy_derived.png")
    plt.savefig(outfile, dpi=300)
    print(f"Saved plot to {outfile}")

    # Numerical Output
    print("\n--- Numerical Validation ---")
    muon_ideal = m_e * (5/3)**9 * 2
    muon_real = muon_ideal * np.sqrt(1 - 0.308 * (5/7)**9)
    print(f"Muon (N=5): Ideal={muon_ideal:.1f}, Real={muon_real:.1f} (Exp: 105.7)")
    
    tau_ideal = m_e * (7/3)**9 * 2
    tau_real = tau_ideal * np.sqrt(1 - 0.308)
    print(f"Tau (N=7): Ideal={tau_ideal:.1f}, Real={tau_real:.1f} (Exp: 1776.9)")

if __name__ == "__main__":
    simulate_mass_hierarchy()