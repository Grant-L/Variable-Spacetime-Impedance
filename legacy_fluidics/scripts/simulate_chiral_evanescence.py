import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs('assets/sim_outputs', exist_ok=True)
    
    # Define spatial grid (in units of l_node)
    x = np.linspace(0, 10, 1000)
    
    # Left-handed wave: perfectly free propagation. Real wavenumber k.
    # U_L(x) = A * cos(k_real * x)
    k_real = 2.0 * np.pi / 2.0  # Wavelength of 2 discrete nodes
    
    # Right-handed wave: hits the mechanical high-pass filter. 
    # Complex wavenumber causes evanescent exponential decay.
    # U_R(x) = A * exp(-kappa * x) * cos(k_real * x)
    kappa = 1.3  # Strong exponential spatial decay from Cosserat stiffness
    
    # Compute amplitudes
    y_L = np.cos(k_real * x)
    y_R = np.exp(-kappa * x) * np.cos(k_real * x)
    
    # Set up dark-mode plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='black')
    fig.patch.set_facecolor('black')
    
    # -----------------------------------------------------------------
    # Plot Left-Handed (Propagating Wave)
    # -----------------------------------------------------------------
    ax1.set_facecolor('black')
    ax1.plot(x, y_L, color='cyan', linewidth=3)
    ax1.set_title(r"Left-Handed Neutrino ($0_1$): Unhindered Transverse Propagation ($\omega^2 > 0$)", 
                  color='white', pad=15, fontsize=14, fontweight='bold')
    
    # Styling
    ax1.grid(color='gray', linestyle='--', alpha=0.3)
    ax1.tick_params(colors='white')
    ax1.set_ylabel("Torsional Lattice Amplitude", color='white', fontsize=12)
    ax1.set_ylim([-1.2, 1.2])
    
    # Fill under curve for aesthetics
    ax1.fill_between(x, y_L, color='cyan', alpha=0.1)
    
    # -----------------------------------------------------------------
    # Plot Right-Handed (Evanescent Decay)
    # -----------------------------------------------------------------
    ax2.set_facecolor('black')
    ax2.plot(x, y_R, color='magenta', linewidth=3, label="Damped Acoustic Wave")
    
    # Plot the exponential decay envelope bounds
    env_upper = np.exp(-kappa * x)
    env_lower = -np.exp(-kappa * x)
    ax2.plot(x, env_upper, color='red', linestyle='--', linewidth=2, label="Evanescent Decay Envelope", alpha=0.8)
    ax2.plot(x, env_lower, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_title(r"Right-Handed Neutrino ($0_1$): Cosserat Anderson Localization ($\omega^2 < 0$)", 
                  color='white', pad=15, fontsize=14, fontweight='bold')
                  
    # Styling
    ax2.grid(color='gray', linestyle='--', alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.set_xlabel(r"Spatial Distance from Source ($x$ / $\ell_{node}$)", color='white', fontsize=12)
    ax2.set_ylabel("Torsional Lattice Amplitude", color='white', fontsize=12)
    ax2.set_ylim([-1.2, 1.2])
    
    ax2.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    # Fill under curve
    ax2.fill_between(x, y_R, color='magenta', alpha=0.1)
    
    # Create descriptive text block explaining the mechanical parity violation
    textstr = "\n".join((
        r"Parity Violation: Mechanical Filter",
        r"L-Handed: matches substrate grain $\to$ free translation",
        r"R-Handed: shears against substrate stiffness ($\gamma_c$)",
        r"$\omega^2 = c^2k^2 - \gamma_c k < 0 \Rightarrow$ Imaginary Frequency",
        r"Result: Rapid spatial amplitude annihilation."
    ))
    
    # Add text box in upper right of top plot
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan')
    ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', color='white', bbox=props)
    
    plt.tight_layout()
    output_path = 'assets/sim_outputs/chiral_parity_violation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Successfully saved Chiral Parity Violation simulation to {output_path}")

if __name__ == '__main__':
    main()
