import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure assets directory exists
if not os.path.exists('assets'):
    os.makedirs('assets')

def save_plot(filename):
    filepath = os.path.join('assets', filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Generated: {filepath}")

# --- 1. Mexican Hat Potential (Chapter 1) ---
def gen_mexican_hat():
    x = np.linspace(-2, 2, 400)
    y = x**4 - 2*x**2
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=3)
    plt.title("The Vacuum Potential Well $V(|\Psi|^2)$")
    plt.xlabel("Vacuum Order Parameter $|\Psi|$")
    plt.ylabel("Potential Energy")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.text(0, 0.5, "Unstable Vacuum (Symmetric)", ha='center')
    plt.text(1, -1.2, "Stable Vacuum\n(Broken Symmetry)", ha='center', color='blue')
    save_plot("mexican_hat.png")

# --- 2. Born Rule Statistics (Chapter 3) ---
def gen_born_rule():
    x = np.linspace(0, np.pi, 100)
    psi_squared = np.sin(x)**2
    walker_counts = np.random.normal(loc=x[np.argmax(psi_squared)], scale=0.5, size=1000)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, psi_squared, 'r-', linewidth=3, label='Wave Intensity $|\Psi|^2$')
    plt.hist(walker_counts, bins=30, density=True, alpha=0.3, color='cyan', label='Walker Histogram')
    plt.title("Emergence of the Born Rule from Deterministic Walks")
    plt.legend()
    save_plot("born_rule.png")

# --- 3. Galaxy Rotation (Chapter 6) ---
def gen_galaxy_rotation():
    r = np.linspace(0.1, 50, 500)
    M_total = 1.0e11
    scale = 3.0
    M_r = M_total * (1 - np.exp(-r/scale))
    
    v_newton = np.sqrt(1.0 * M_r / r)
    stiffness = 1.0 + 0.8 * (r / 20.0)
    v_lct = np.sqrt(stiffness * M_r / r)
    
    # Normalize
    norm = 220 / v_lct[-1]
    v_newton *= norm
    v_lct *= norm
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, v_newton, 'b--', label='Standard Newtonian (Keplerian)')
    plt.plot(r, v_lct, 'r-', linewidth=2, label='LCT Variable Stiffness')
    plt.fill_between(r, v_newton, v_lct, color='gray', alpha=0.1, label='The "Dark Matter" Gap')
    plt.title("Galaxy Rotation: Stiffness vs. Mass")
    plt.legend()
    save_plot("galaxy_rotation.png")

# --- 4. Hubble Tension (Chapter 6) ---
def gen_hubble_tension():
    t = np.linspace(0.01, 1.0, 1000)
    h_standard = np.linspace(67, 67, 1000) # Simplified constant
    h_lct = np.linspace(67, 73, 1000) # Drift
    
    plt.figure(figsize=(8, 6))
    plt.plot(t, h_standard, 'b--', label='Standard Constant Model')
    plt.plot(t, h_lct, 'r-', linewidth=2, label='LCT Impedance Drift Model')
    plt.fill_between(t, h_standard, h_lct, color='purple', alpha=0.1, label='Hubble Tension Gap')
    plt.title("Cosmological Impedance Evolution")
    plt.ylabel("$H_0$ (km/s/Mpc)")
    plt.xlabel("Cosmic Time (normalized)")
    plt.legend()
    save_plot("hubble_tension_shift.png")

# --- 5. Proton Radius (Chapter 6) ---
def gen_proton_radius():
    r = np.linspace(0.1, 2.0, 500)
    profile = 1.0 / (r**2 + 0.1)
    e_sens = np.exp(-r/0.8)
    m_sens = np.exp(-r/0.4)
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, profile*e_sens, 'b-', label='Electron Interaction (Flow)')
    plt.plot(r, profile*m_sens, 'r-', label='Muon Interaction (Core)')
    plt.axvline(0.877, color='blue', linestyle='--')
    plt.axvline(0.841, color='red', linestyle='--')
    plt.title("Proton Radius: Frequency Dependent Scattering")
    plt.legend()
    save_plot("proton_radius_scattering.png")

# --- Run All ---
if __name__ == "__main__":
    print("Generating Textbook Assets...")
    gen_mexican_hat()
    gen_born_rule()
    gen_galaxy_rotation()
    gen_hubble_tension()
    gen_proton_radius()
    print("Done! Assets are ready for LaTeX compilation.")