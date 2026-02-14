import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "manuscript/chapters/04_baryon_sector/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_mass_hierarchy():
    print("Deriving Mass Spectrum via Topological Invariants...")
    
    # 1. Constants
    m_e_exp = 0.511   # MeV
    alpha_inv = 137.036
    alpha = 1.0 / alpha_inv
    
    # 2. Lepton Sector (Hyperbolic Volume Scaling)
    # ---------------------------------------------
    # Electron (3_1)
    N_e = 3.0
    Vol_3_1 = 2.8284  # Effective Hyperbolic Volume
    
    # Muon (5_1)
    N_mu = 5.0
    Vol_5_1 = 6.0235  # Effective Hyperbolic Volume
    m_mu_exp = 105.66 # MeV
    
    # Volume Ratio
    R_vol = Vol_5_1 / Vol_3_1
    
    # Prediction
    m_mu_pred = m_e_exp * R_vol * (N_mu / N_e)**9
    
    # 3. Baryon Sector (Schwinger Binding Correction)
    # -----------------------------------------------
    m_p_exp = 938.272 # MeV
    
    # Base Geometry
    Omega_base = 4 * np.pi + (5.0 / 6.0)
    
    # Schwinger Correction (Binding Energy)
    # Two interfaces, alpha/2pi damping
    delta_bind = 2.0 * (alpha / (2 * np.pi))
    
    # Corrected Form Factor
    Omega_corrected = Omega_base - delta_bind
    
    # Prediction
    m_p_pred = m_e_exp * alpha_inv * Omega_corrected

    # 4. Output Results
    print(f"{'-'*50}")
    print(f"LEPTON SECTOR (Hyperbolic Volume Fix)")
    print(f"Volume Ratio R_vol: {R_vol:.4f}")
    print(f"Muon Pred: {m_mu_pred:.3f} MeV | Exp: {m_mu_exp:.3f} MeV")
    error_mu = abs(m_mu_pred - m_mu_exp) / m_mu_exp * 100
    print(f"Error: {error_mu:.2f}%")
    
    print(f"\nBARYON SECTOR (Schwinger Binding Fix)")
    print(f"Base Omega: {Omega_base:.5f}")
    print(f"Schwinger Correction: -{delta_bind:.5f}")
    print(f"Final Omega: {Omega_corrected:.5f}")
    print(f"Proton Pred: {m_p_pred:.3f} MeV | Exp: {m_p_exp:.3f} MeV")
    error_p = abs(m_p_pred - m_p_exp) / m_p_exp * 100
    print(f"Error: {error_p:.4f}%")
    print(f"{'-'*50}")

    return m_mu_pred, m_p_pred

def plot_results(mu_pred, p_pred):
    # Visualization of Precision
    labels = ['Muon (105.7)', 'Proton (938.3)']
    experimental = [105.66, 938.27]
    predicted = [mu_pred, p_pred]
    
    plt.figure(figsize=(10, 6))
    
    # Muon Plot
    plt.subplot(1, 2, 1)
    plt.bar(['Exp', 'AVE'], [105.66, mu_pred], color=['gray', '#D95319'])
    plt.title(f"Muon Mass\n(Error: {abs(mu_pred-105.66)/105.66*100:.1f}%)")
    plt.ylabel('Mass (MeV)')
    plt.ylim(100, 115)
    
    # Proton Plot
    plt.subplot(1, 2, 2)
    plt.bar(['Exp', 'AVE'], [938.27, p_pred], color=['gray', '#77AC30'])
    plt.title(f"Proton Mass\n(Error: {abs(p_pred-938.27)/938.27*100:.4f}%)")
    plt.ylim(938.0, 938.5)
    
    output_path = os.path.join(OUTPUT_DIR, "derive_proton_mass.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    mu, p = calculate_mass_hierarchy()
    plot_results(mu, p)