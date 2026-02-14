import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def momentum_sweep():
    print("Running Momentum Sweep (Vacuum Rheology Test)...")
    
    # Physics Constants (AVE Framework)
    gamma_c = 5.0  # Critical Shear Threshold
    eta_0 = 0.5    # Base Vacuum Viscosity (Opacity)
    
    # Momentum Levels to Test (k ~ Frequency ~ Shear Rate)
    # k=2 (Low Energy / Radio)
    # k=8 (Medium Energy / Visible)
    # k=20 (High Energy / Gamma)
    momenta = [2.0, 8.0, 20.0] 
    labels = ["Low Momentum (k=2)", "Medium Momentum (k=8)", "High Momentum (k=20)"]
    
    # Grid Setup (2D Slice for clarity)
    z = np.linspace(0, 20, 200)
    x = np.linspace(-5, 5, 100)
    Z, X = np.meshgrid(z, x)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), facecolor='black')
    
    for i, k in enumerate(momenta):
        ax = axes[i]
        
        # 1. SHEAR THINNING CALCULATION (The Axiom)
        # Shear rate is proportional to frequency (k)
        shear_rate = k 
        
        # Effective Viscosity drops as shear increases
        # eta = eta_0 / (1 + (shear/critical)^2)
        eta_eff = eta_0 / (1 + (shear_rate / gamma_c)**2)
        
        # Damping Factor (Attenuation)
        # Low k -> High Viscosity -> High Damping
        damping = np.exp(-eta_eff * Z / 5.0) 
        
        # 2. WAVE FUNCTION
        # Rifled Helix projected onto 2D
        # Envelope * Damping * Oscillation
        envelope = np.exp(-X**2 / 2.0)
        phase = k * Z
        amplitude = envelope * damping * np.cos(phase)
        
        # 3. PLOT
        # Use diverging colormap to show phase structure
        im = ax.contourf(Z, X, amplitude, 100, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Annotations
        ax.set_facecolor('black')
        ax.set_ylabel("Transverse (X)", color='white')
        if i == 2: ax.set_xlabel("Propagation (Z)", color='white')
        
        # Title with Physics stats
        title = f"{labels[i]} | Shear: {shear_rate:.1f} | $\eta_{{eff}}$: {eta_eff:.3f}"
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        
        # Remove ticks for clean look
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Visual cues for attenuation
        if i == 0:
            ax.text(18, 3, "High Damping\n(Viscous)", color='cyan', ha='right')
        if i == 2:
            ax.text(18, 3, "Low Damping\n(Superfluid)", color='cyan', ha='right')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "photon_rheology.png")
    plt.savefig(output_path, dpi=300, facecolor='black')
    print(f"Sweep simulation saved to {output_path}")

if __name__ == "__main__":
    momentum_sweep()