import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "manuscript/chapters/03_fermion_sector/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_mass_hierarchy():
    print("Deriving Lepton Mass Spectrum via Quartic Vacuum Potential...")
    
    # 1. Constants
    m_e_exp = 0.511  # Experimental Electron Mass (MeV)
    m_mu_exp = 105.66 # Experimental Muon Mass (MeV)
    m_tau_exp = 1776.86 # Experimental Tau Mass (MeV)
    
    # 2. Topology Inputs (The "Integers")
    N_e = 3.0   # Electron (Trefoil)
    N_mu = 5.0  # Muon (Pentafoil)
    N_tau = 7.0 # Tau (Septafoil)
    
    # 3. The Model: Inductive Scaling Law (Section 3.3)
    # Mass ~ Energy ~ Volume * EnergyDensity
    # Volume ~ N (Linear Arc Length)
    # EnergyDensity ~ Strain^4 ~ (N^2)^4 = N^8 (Quartic Potential)
    # Result: Mass ~ N^9
    
    def mass_law(N, geometric_factor=1.0):
        # Normalized to Electron Ground State (N=3)
        scaling_ratio = (N / N_e) ** 9
        return m_e_exp * scaling_ratio * geometric_factor

    # 4. Predictions
    # Electron: Reference State (Factor=1)
    m_e_pred = mass_law(N_e, geometric_factor=1.0)
    
    # Muon: First Excited State (N=5)
    # Geometry: Full-Wave Loop (Doubling Factor = 2)
    m_mu_pred = mass_law(N_mu, geometric_factor=2.0)
    
    # Tau: Second Excited State (N=7)
    # Geometry: Full-Wave Loop (Doubling Factor = 2)
    # Note: We apply the "Running Coupling" correction here if refining,
    # but strictly following N^9 yields the raw geometric resonance.
    m_tau_pred = mass_law(N_tau, geometric_factor=2.0)
    
    # 5. Output Results
    print(f"{'-'*40}")
    print(f"Electron (N=3): Ref = {m_e_pred:.3f} MeV")
    
    print(f"Muon     (N=5): Pred = {m_mu_pred:.3f} MeV | Exp = {m_mu_exp:.3f} MeV")
    error_mu = abs(m_mu_pred - m_mu_exp) / m_mu_exp * 100
    print(f"                Error = {error_mu:.2f}%")
    
    print(f"Tau      (N=7): Pred = {m_tau_pred:.3f} MeV | Exp = {m_tau_exp:.3f} MeV")
    error_tau = abs(m_tau_pred - m_tau_exp) / m_tau_exp * 100
    print(f"                Error = {error_tau:.2f}% (Raw Geometric Resonance)")
    print(f"{'-'*40}")

    return (N_e, m_e_pred), (N_mu, m_mu_pred), (N_tau, m_tau_pred)

def plot_hierarchy(e, mu, tau):
    N_range = np.linspace(2.5, 7.5, 100)
    
    # Ideal N^9 Scaling Curve (Normalized to Electron)
    # We include the geometric factor transition smoothly for visualization
    # (Sigmoid blend from 1 to 2 between N=3 and N=5)
    factor = 1.0 + 1.0 / (1.0 + np.exp(-(N_range - 4)*5)) 
    mass_curve = 0.511 * (N_range / 3.0)**9 * factor

    plt.figure(figsize=(10, 6))
    
    # Plot Theoretical Law
    plt.plot(N_range, mass_curve, '--', color='blue', alpha=0.6, label=r'AVE Prediction ($N^9$ Scaling)')
    
    # Plot Data Points
    plt.plot(e[0], e[1], 'go', markersize=10, label='Electron (Ref)')
    plt.plot(mu[0], mu[1], 'ro', markersize=10, label='Muon (Predicted)')
    plt.plot(tau[0], tau[1], 'ro', markersize=10, label='Tau (Predicted)')
    
    # Experimental Benchmarks
    plt.axhline(y=105.66, color='r', linestyle=':', alpha=0.3, label='Exp. Muon Mass')
    plt.axhline(y=1776.86, color='r', linestyle=':', alpha=0.3, label='Exp. Tau Mass')

    plt.yscale('log')
    plt.title(r'Lepton Mass Hierarchy: Quartic Vacuum Potential ($U \propto \phi^4$)', fontsize=14)
    plt.xlabel('Topological Winding Number (N)', fontsize=12)
    plt.ylabel('Rest Mass Energy (MeV)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    output_path = os.path.join(OUTPUT_DIR, "derive_mass_scaling.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    e_data, mu_data, tau_data = calculate_mass_hierarchy()
    plot_hierarchy(e_data, mu_data, tau_data)