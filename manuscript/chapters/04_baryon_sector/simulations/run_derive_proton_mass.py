import numpy as np
import matplotlib.pyplot as plt
import os

# Directory setup
OUTPUT_DIR = "manuscript/chapters/04_baryon_sector/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_proton_mass():
    """
    Attempts to derive the Proton Mass (938.27 MeV) from the Electron Mass (0.511 MeV)
    using the Borromean Linkage Topology and the Fine Structure Constant.
    """
    print("Simulating Proton Mass Derivation...")

    # 1. Fundamental Constants
    m_e = 0.51106           # Electron Mass (MeV) - Ground State Loop
    alpha_inv = 137.03599   # Inverse Fine Structure Constant (Geometric Impedance)
    
    # Target
    m_p_exp = 938.272       # Experimental Proton Mass (MeV)

    # 2. Geometric Hypothesis
    # The Proton is a Borromean Link of 3 loops.
    # We test the hypothesis: m_p = m_e * alpha^-1 * Omega_topo
    # Let's solve for the required Omega_topo to match reality.
    
    required_omega = m_p_exp / (m_e * alpha_inv)
    print(f"Required Geometric Factor (Omega_topo): {required_omega:.5f}")

    # 3. Geometric Candidates for Omega_topo
    # What geometric constants are close to ~13.4?
    
    # Candidate A: 4 * Pi (Spherical Flux Factor)
    omega_A = 4 * np.pi
    
    # Candidate B: 3 Loops * Pi (Simple Loop Summation)
    # 3 loops * 3.14 + Interaction terms?
    # Let's look at the Toroidal Geometry. Volume of Torus = 2 * pi^2 * R * r^2.
    # The Borromean link has 3 interlocking axes.
    # The "Linkage Number" might be related to the number of crossings (6).
    
    # Candidate C: The "Cheshire Cat" Surface
    # 13.4 is suspiciously close to 4*Pi + 1 (12.56 + 1 = 13.56).
    # Or 4*Pi * (1 + delta).
    
    # Candidate D: The Inductive Coupling of 3 orthogonal loops.
    # Mutual inductance M usually scales as k * sqrt(L1*L2).
    # If L scales with alpha^-1, and we have 3 loops...
    
    # Let's test the "Spherical Shell" Hypothesis (Candidate A):
    # m_p ~ Surface Area of the Flux Sphere (4*pi) * Impedance (alpha^-1) * Unit Mass (m_e)?
    
    # Let's calculate the mass for 4*pi
    m_p_model_A = m_e * alpha_inv * (4 * np.pi)
    
    # Let's try to refine Candidate A with the Packing Factor (Kappa ~ 0.437)
    # Maybe m_p = m_e * alpha^-1 * 4*pi * (1 + kappa)?
    m_p_model_B = m_e * alpha_inv * (4 * np.pi) * (1 + 0.437/10) # Just guessing structure
    
    # Let's try a pure topological integer derivation.
    # Proton has Baryon Number B=1. 3 Quarks.
    # What if Omega = 4 * Pi * (1 + 1/(2*Pi))? -> 12.56 * 1.15 = 14.4 (Too high)
    
    # Let's look at the ratio: 1836.15 (mp/me)
    # Ratio / Alpha^-1 = 1836.15 / 137.036 = 13.399
    
    # THIS IS THE KEY NUMBER: 13.399.
    # We need to find a geometric derivation for ~13.4.
    # 4 * Pi = 12.566
    # 4 * Pi + 1 = 13.566 (Too high)
    # 4 * Pi + ln(3)? = 12.56 + 1.09 = 13.65 (Close)
    # 4 * Pi * (1 + 2/30)?
    
    # Let's check 4 * Pi * (1 + 1/15) = 13.404.
    # 1/15? The packing fraction?
    
    # Let's plot the "Spectrum of Geometry" to see what hits.
    
    candidates = {
        '4$\pi$ (Spherical)': 4 * np.pi,
        '13.4 (Empirical Target)': 13.399,
        '4$\pi$ + 1': 4 * np.pi + 1,
        '4$\pi$ + $\ln(2)$': 4 * np.pi + np.log(2),
        '137/10 (Decimated Alpha)': 13.7
    }
    
    print("\n--- Geometric Candidate Search ---")
    for name, val in candidates.items():
        mass_pred = m_e * alpha_inv * val
        error = (mass_pred - m_p_exp) / m_p_exp * 100
        print(f"{name}: Mass = {mass_pred:.2f} MeV (Error: {error:.2f}%)")

    # 4. Plotting
    names = list(candidates.keys())
    values = [m_e * alpha_inv * v for v in candidates.values()]
    errors = [abs(v - m_p_exp) for v in values]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color=['gray', 'green', 'gray', 'gray', 'gray'])
    plt.axhline(m_p_exp, color='red', linestyle='--', linewidth=2, label=f'Experimental Proton Mass ({m_p_exp:.1f} MeV)')
    
    plt.title(f'Proton Mass Candidates (Based on $m_e \\cdot \\alpha^{{-1}} \\cdot \\Omega$)')
    plt.ylabel('Mass (MeV)')
    plt.ylim(800, 1000)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    outfile = os.path.join(OUTPUT_DIR, "proton_mass_search.png")
    plt.savefig(outfile, dpi=300)
    print(f"Saved plot to {outfile}")

    # 5. The "Euler" Coincidence?
    # e^Pi = 23.14 (No)
    # Pi * e = 8.5 (No)
    # 4 * Pi + 2/3 (Quark charge?) = 12.56 + 0.66 = 13.22 (Close!)
    # 4 * Pi + 2/3 + 2/3 - 1/3? (Sum of charges?)
    # 4 * Pi + 5/6 = 13.399??
    # Let's check 4 * Pi + 5/6
    val_quark_sum = 4 * np.pi + (5/6)
    mass_quark_sum = m_e * alpha_inv * val_quark_sum
    print(f"\nHYPOTHESIS CHECK: 4*Pi + 5/6")
    print(f"Value: {val_quark_sum:.4f}")
    print(f"Predicted Mass: {mass_quark_sum:.2f} MeV")
    print(f"Error: {(mass_quark_sum - m_p_exp)/m_p_exp * 100:.3f}%")

if __name__ == "__main__":
    simulate_proton_mass()