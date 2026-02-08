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

# --- Fig 1.1: Mexican Hat (Chapter 1) ---
def gen_mexican_hat():
    x = np.linspace(-2, 2, 400)
    y = x**4 - 2*x**2
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=3)
    plt.title(r"The Vacuum Potential Well $V(|\Psi|^2)$")
    plt.xlabel(r"Vacuum Order Parameter $|\Psi|$")
    plt.ylabel("Potential Energy")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.text(0, 0.5, "Unstable Vacuum", ha='center')
    plt.text(1, -1.2, "Stable VEV", ha='center', color='blue')
    save_plot("mexican_hat.png")

# --- Fig 3.1: Born Rule (Chapter 3) ---
def gen_born_rule():
    x = np.linspace(0, np.pi, 100)
    psi_squared = np.sin(x)**2
    # Inverse transform sampling for histogram
    r = np.random.rand(2000)
    walker_counts = np.arccos(1 - 2*r) 
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, psi_squared, 'r-', linewidth=3, label=r'Wave Intensity $|\Psi|^2$')
    plt.hist(walker_counts, bins=30, density=True, alpha=0.3, color='cyan', label='Walker Histogram')
    plt.title("Emergence of the Born Rule")
    plt.legend()
    save_plot("born_rule.png")

# --- Fig 5.1: Gravitational Decoherence (Chapter 5) ---
def gen_decoherence():
    # Simplified visual approximation of the complex wave simulation
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Interference Pattern (Quantum)
    k = 2.0
    psi = np.sin(k * (X + 2*Y)) + np.sin(k * (X - 2*Y))
    
    # Horizon Scrambling (Thermodynamic)
    # Noise increases as R -> 0 (Event Horizon)
    noise_mask = 1.0 / (R + 0.5)
    scrambled = psi * (1 - np.exp(-R/3)) + np.random.normal(0, 2, X.shape) * np.exp(-R/2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(scrambled, extent=[-10, 10, -10, 10], cmap='magma', origin='lower')
    plt.title("Gravitational Decoherence at the Horizon")
    # Add black hole disc
    circle = plt.Circle((0, 0), 2, color='black')
    plt.gca().add_patch(circle)
    plt.text(0, 0, "Thermodynamic\nScrambling", color='red', ha='center', va='center', fontsize=8)
    plt.axis('off')
    save_plot("gravitational_double_slit.png")

# --- Fig 6.1: Galaxy Rotation (Chapter 6) ---
def gen_galaxy_rotation():
    r = np.linspace(0.1, 50, 500)
    M_r = 1.0e11 * (1 - np.exp(-r/3.0))
    v_newton = np.sqrt(M_r / r)
    stiffness = 1.0 + 0.8 * (r / 20.0)
    v_lct = np.sqrt(stiffness * M_r / r)
    
    # Normalize
    norm = 220 / v_lct[-1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, v_newton * norm, 'b--', label='Standard Newtonian')
    plt.plot(r, v_lct * norm, 'r-', linewidth=3, label='LCT Variable Stiffness')
    plt.fill_between(r, v_newton*norm, v_lct*norm, color='gray', alpha=0.1, label='Dark Matter Gap')
    plt.title("Galaxy Rotation Curves")
    plt.legend()
    save_plot("galaxy_rotation.png")

# --- Fig 6.2: Hubble Tension (Chapter 6) ---
def gen_hubble_tension():
    t = np.linspace(0.01, 1.0, 1000)
    h_standard = np.ones_like(t) * 67
    h_lct = 67 + (73 - 67) * t
    
    plt.figure(figsize=(8, 6))
    plt.plot(t, h_standard, 'b--', label='Standard Model')
    plt.plot(t, h_lct, 'r-', linewidth=3, label='LCT Impedance Drift')
    plt.fill_between(t, h_standard, h_lct, color='purple', alpha=0.1, label='Hubble Tension Gap')
    plt.title("Cosmological Impedance Evolution")
    plt.legend()
    save_plot("hubble_tension_shift.png")

# --- Fig 6.3: Proton Radius (Chapter 6) ---
def gen_proton_radius():
    r = np.linspace(0.1, 2.0, 500)
    profile = 1.0 / (r**2 + 0.1)
    e_sens = np.exp(-r/0.8)
    m_sens = np.exp(-r/0.2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, profile*e_sens, 'b-', label='Electron (Flow)')
    plt.plot(r, profile*m_sens, 'r-', label='Muon (Core)')
    plt.axvline(0.877, color='blue', linestyle='--')
    plt.axvline(0.841, color='red', linestyle='--')
    plt.title("Proton Radius Scattering")
    plt.legend()
    save_plot("proton_radius_scattering.png")

if __name__ == "__main__":
    print("Generating all textbook figures...")
    gen_mexican_hat()
    gen_born_rule()
    gen_decoherence() # Added this!
    gen_galaxy_rotation()
    gen_hubble_tension()
    gen_proton_radius()
    print("Done.")