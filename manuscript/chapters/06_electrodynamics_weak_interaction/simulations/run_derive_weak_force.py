import numpy as np
import matplotlib.pyplot as plt
import os

# Directory setup
OUTPUT_DIR = "manuscript/chapters/06_electrodynamics_weak_interaction/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def derive_weak_force():
    """
    Attempts to derive the Weak Boson Masses (W, Z) from the Proton Mass and Alpha.
    
    Hypothesis:
    1. Base Impedance Scale S = m_p * alpha^-1
    2. m_W = S * (5/8)  (The 5/8 Geometric Partition)
    3. m_Z = m_W / (sqrt(7)/3) (The Geometric Mixing Angle)
    """
    print("Simulating Weak Force Derivation...")

    # 1. Fundamental Inputs (CODATA 2022)
    m_p = 938.272088      # MeV
    alpha_inv = 137.035999
    
    # 2. Experimental Targets (PDG 2024)
    m_W_exp = 80379.0     # MeV (+- 12)
    m_Z_exp = 91187.6     # MeV (+- 2)

    # 3. The "Impedance Bridge" Scale (S)
    # The energy required to stress a proton-sized knot to the vacuum impedance limit.
    S = m_p * alpha_inv
    print(f"Base Impedance Scale (S): {S:.2f} MeV")

    # 4. Derive W Boson (The 5/8 Resonance)
    omega_W = 5/8
    m_W_pred = S * omega_W
    
    # 5. Derive Z Boson (The Geometric Mixing)
    # Hypothesis: cos(theta_W) = sqrt(7)/3
    # This implies a mixing angle theta ~ 28.1 degrees.
    cos_theta_geo = np.sqrt(7) / 3
    m_Z_pred = m_W_pred / cos_theta_geo

    # 6. Output Results
    print("\n--- W Boson Prediction ---")
    print(f"Geometric Factor: 5/8 ({omega_W})")
    print(f"Predicted Mass: {m_W_pred:.2f} MeV")
    print(f"Target Mass:    {m_W_exp:.2f} MeV")
    error_W = (m_W_pred - m_W_exp) / m_W_exp * 100
    print(f"Error:          {error_W:.4f}%")

    print("\n--- Z Boson Prediction ---")
    print(f"Mixing Factor:  3/sqrt(7) ({1/cos_theta_geo:.4f})")
    print(f"Predicted Mass: {m_Z_pred:.2f} MeV")
    print(f"Target Mass:    {m_Z_exp:.2f} MeV")
    error_Z = (m_Z_pred - m_Z_exp) / m_Z_exp * 100
    print(f"Error:          {error_Z:.4f}%")
    
    # 7. Visualization
    labels = ['W Boson', 'Z Boson']
    preds = [m_W_pred, m_Z_pred]
    exps = [m_W_exp, m_Z_exp]
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, preds, width, label='AVE Derived', color='blue', alpha=0.7)
    plt.bar(x + width/2, exps, width, label='Experimental', color='gray', alpha=0.5)
    
    plt.ylabel('Mass (MeV)')
    plt.title('Weak Boson Mass Derivation (From Proton + Alpha)')
    plt.xticks(x, labels)
    plt.ylim(0, 100000)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.text(0, m_W_pred + 2000, f"Err: {error_W:.3f}%", ha='center', fontweight='bold')
    plt.text(1, m_Z_pred + 2000, f"Err: {error_Z:.3f}%", ha='center', fontweight='bold')

    outfile = os.path.join(OUTPUT_DIR, "weak_force_derivation.png")
    plt.savefig(outfile, dpi=300)
    print(f"\nSaved plot to {outfile}")

if __name__ == "__main__":
    derive_weak_force()