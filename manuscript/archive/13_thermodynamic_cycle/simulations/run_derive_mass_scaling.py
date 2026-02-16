import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "manuscript/chapters/13_thermodynamic_cycle/simulations"

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def calculate_mass_scaling():
    print("Deriving Knot Inductance Scaling Laws...")
    
    # 1. DEFINITIONS
    # Topological Winding Numbers (Knots)
    # Electron (3_1), Muon (5_1 hypot), Tau (7_1 hypot)
    N_knots = np.linspace(1, 9, 50) 
    
    # Experimental Mass Data (MeV)
    # We normalize everything to the Pair Production Energy (E0 = 1.022 MeV)
    # Electron Mass ~ 0.511 -> Pair = 1.022
    E0 = 1.022 
    
    # Data Points (Winding Number, Mass in MeV)
    # N=3 (Electron), N=5 (Muon), N=7 (Tau)
    leptons = {
        "Electron (3_1)": {"N": 3, "Mass": 0.511},
        "Muon (5_1)":     {"N": 5, "Mass": 105.66},
        "Tau (7_1)":      {"N": 7, "Mass": 1776.86}
    }
    
    # 2. MODELS
    
    # Model A: Standard Inductance (The Solenoid)
    # L ~ N^2
    # Mass = E0 * (N/3)^2 * (0.5 for ground state)
    model_standard = E0 * (N_knots / 3.0)**2 * 0.5
    
    # Model B: Geometric Crowding (Volume Constraint)
    # If Volume V ~ 1/N (Compton), and Energy ~ B^2 * V
    # This roughly scales as N^4 to N^5
    model_crowding = E0 * (N_knots / 3.0)**5 * 0.5

    # Model C: VSI Saturated Lattice (The N^9 Hypothesis)
    # L ~ N^2 (Base) * N^3 (Compression) * N^4 (Permeability Non-linearity)
    model_vsi = E0 * (N_knots / 3.0)**9 * 0.5

    return N_knots, model_standard, model_crowding, model_vsi, leptons

def plot_derivation(N, m_std, m_crowd, m_vsi, data):
    plt.figure(figsize=(10, 7))
    
    # Plot Models
    plt.plot(N, m_std, '--', color='gray', label='Standard Inductance ($N^2$)')
    plt.plot(N, m_crowd, '-.', color='orange', label='Geometric Crowding ($N^5$)')
    plt.plot(N, m_vsi, '-', color='blue', linewidth=2, label='VSI Saturated Lattice ($N^9$)')
    
    # Plot Experimental Data
    for name, props in data.items():
        n_val = props["N"]
        m_val = props["Mass"]
        plt.plot(n_val, m_val, 'ro', markersize=8)
        plt.text(n_val, m_val * 1.3, name, ha='center', fontweight='bold')
    
    # Log Scale is essential to see the hierarchy
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.xlabel('Topological Winding Number ($N$)', fontsize=12)
    plt.ylabel('Rest Mass Energy (MeV)', fontsize=12)
    plt.title('Derivation of the Lepton Mass Hierarchy', fontsize=14)
    plt.legend()
    
    # Annotations
    plt.text(4, 10, "The Inductive Gap:\nStandard physics ($N^2$)\ncannot explain the\nMuon/Tau mass spike.", 
             bbox=dict(facecolor='white', alpha=0.8))

    filepath = os.path.join(OUTPUT_DIR, "derive_mass_scaling.png")
    plt.savefig(filepath, dpi=300)
    print(f"Saved Derivation Plot: {filepath}")
    plt.close()

if __name__ == "__main__":
    ensure_output_dir()
    N, m1, m2, m3, d = calculate_mass_scaling()
    plot_derivation(N, m1, m2, m3, d)
    print("Derivation Complete.")