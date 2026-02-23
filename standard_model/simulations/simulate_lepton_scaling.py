import numpy as np
import matplotlib.pyplot as plt
import os

# --- EMPIRICAL CODATA MASSES (MeV) ---
m_e = 0.51099895    # Electron
m_mu = 105.6583755  # Muon
m_tau = 1776.86     # Tau

def main():
    print("==========================================================")
    print(" AVE STANDARD MODEL: THE LEPTON HIERARCHY SIMULATOR")
    print("==========================================================\n")

    print(f"Empirical Electron Mass : {m_e:9.4f} MeV")
    print(f"Empirical Muon Mass     : {m_mu:9.4f} MeV")
    print(f"Empirical Tau Mass      : {m_tau:9.4f} MeV\n")

    print("--- Topological Gyroscopic Scaling ---")
    print("Hypothesis: M_topo ∝ 1/R_topo")
    print("Assume baseline normalized Electron Radius R_e = 1.0\n")

    R_e = 1.0
    
    # Calculate required kinematic compression
    R_mu = R_e * (m_e / m_mu)
    R_tau = R_e * (m_e / m_tau)

    print(f"Calculated Electron Radius : {R_e:9.6f} (Normalized)")
    print(f"Calculated Muon Radius     : {R_mu:9.6f} (Compression factor: {m_mu/m_e:.2f}x)")
    print(f"Calculated Tau Radius      : {R_tau:9.6f} (Compression factor: {m_tau/m_e:.2f}x)\n")

    # Generate 1/R curve for visualization (log space due to massive scale difference)
    radii = np.logspace(np.log10(R_tau * 0.5), np.log10(R_e * 2.0), 500)
    masses = m_e / radii

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the 1/R continuous spectrum
    plt.plot(radii, masses, color='grey', linestyle='--', alpha=0.7, label=r"Topological Potential $m \propto 1/R$")
    
    # Plot the specific stabilized kinematic states
    plt.scatter([R_tau], [m_tau], color='red', s=100, zorder=5, label=fr"Tau ($\tau^-$): {m_tau:.1f} MeV")
    plt.scatter([R_mu], [m_mu], color='orange', s=100, zorder=5, label=fr"Muon ($\mu^-$): {m_mu:.1f} MeV")
    plt.scatter([R_e], [m_e], color='blue', s=100, zorder=5, label=fr"Electron ($e^-$): {m_e:.3f} MeV")

    # Annotations
    plt.annotate("Absolute highest physical tension", xy=(R_tau, m_tau), xytext=(R_tau*1.5, m_tau*1.2),
                 arrowprops=dict(facecolor='red', arrowstyle="->"), color='red')
    
    plt.annotate("Intermediate kinematic tension", xy=(R_mu, m_mu), xytext=(R_mu*1.5, m_mu*1.5),
                 arrowprops=dict(facecolor='orange', arrowstyle="->"), color='orange')
    
    plt.annotate("Stable geometric ground state", xy=(R_e, m_e), xytext=(R_e*0.1, m_e*2),
                 arrowprops=dict(facecolor='blue', arrowstyle="->"), color='blue')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.2, color='white')
    
    # Dark mode aesthetics
    plt.gca().set_facecolor('#111111')
    plt.gcf().patch.set_facecolor('#111111')
    plt.gca().tick_params(colors='white')
    for spine in plt.gca().spines.values():
        spine.set_color('#333333')
    
    plt.xlabel(r"Normalized $3_1$ Topological Radius ($R_{topo}$)", color='white', fontsize=12)
    plt.ylabel(r"Inductive Reactance Mass Limit (MeV)", color='white', fontsize=12)
    plt.title(r"The Lepton Hierarchy as $3_1$ Geometric Scaling ($M \propto 1/d$)", color='white', fontsize=14)
    
    legend = plt.legend(facecolor='#222222', edgecolor='#555555', labelcolor='white')
    
    os.makedirs('standard_model/figures', exist_ok=True)
    output_path = 'standard_model/figures/lepton_mass_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    print(f"Success! Output plot generated at: {output_path}")

if __name__ == "__main__":
    main()
