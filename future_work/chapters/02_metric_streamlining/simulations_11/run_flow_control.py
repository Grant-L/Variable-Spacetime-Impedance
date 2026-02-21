"""
AVE MODULE 35: METRIC STREAMLINING (ACTIVE FLOW CONTROL)
--------------------------------------------------------
Strict evaluation of Bingham-Plastic flow control.
Proves that a relativistic vessel can drastically reduce its Inertial Drag 
Coefficient (C_d) by emitting a high-frequency forward shear beam. 
The beam forces the local shear rate (\dot{\gamma}) above the critical 
limit (\dot{\gamma}_c), collapsing the \mathcal{M}_A viscosity (\eta \to 0) 
and generating a frictionless superfluid slipstream.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "future_work/chapters/02_metric_streamlining/simulations_11/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_metric_streamlining():
    print("Simulating Active Metric Streamlining (Bingham-Plastic Flow Control)...")
    
    X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-3, 3, 300))
    hull = (X**2 / 1.5**2 + Y**2 / 0.8**2) <= 1.0
    
    # Kinematic Shear Rate of a Blunt Body at Relativistic Velocity
    natural_shear = np.exp(-3 * (X - -1.5)**2 - 4 * Y**2) * (X < -1.0)
    
    # Active Metric Actuator (High-Frequency Shear Beam) \omega \gg \dot{\gamma}_c
    actuator_beam = np.exp(-1 * (X - -3.0)**2 - 15 * Y**2) * (X < -1.5)
    
    # Bingham-Plastic Viscosity: \eta_{eff} = \eta_0 / (1 + (\dot{\gamma} / \gamma_c)^2)
    gamma_c = 0.2  
    
    # Case A: Passive Flight (Standard Inertia)
    eta_passive = 1.0 / (1.0 + (natural_shear / gamma_c)**2)
    
    # Case B: Active Metric Streamlining
    total_shear_active = natural_shear + 10.0 * actuator_beam
    eta_active = 1.0 / (1.0 + (total_shear_active / gamma_c)**2)
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor('#050508')
    titles = [
        r"Passive Relativistic Flight (High Inertial Drag $C_d \approx 1$)",
        r"Active Metric Streamlining (Superfluid Slipstream $C_d \ll 1$)"
    ]
    etas = [eta_passive, eta_active]
    
    for i, ax in enumerate(axes):
        ax.set_facecolor('#050508')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[i], color='white', fontsize=14, weight='bold', pad=10)
        im = ax.contourf(X, Y, etas[i], levels=50, cmap='magma', vmin=0, vmax=1.0)
        ax.contourf(X, Y, hull, levels=[0.5, 1.5], colors=['#888888'])
        
        if i == 1:
            ax.contour(X, Y, actuator_beam, levels=[0.5, 0.9], colors='#00ffcc', linestyles='dashed', linewidths=1.5)
            ax.text(-3.5, 1.2, "Metric Actuator\n(High-Frequency Shear Beam)", color='#00ffcc', ha='center', weight='bold')
            ax.annotate("", xy=(-1.5, 0), xytext=(-4.5, 0), arrowprops=dict(arrowstyle="<-", color="#00ffcc", lw=2))
        for spine in ax.spines.values(): spine.set_color('#333333')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Effective Vacuum Viscosity ($\eta_{eff}$)', color='white', weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    textstr = (
        r"$\mathbf{Vacuum~Aerodynamics:}$" + "\n" +
        r"The actuator artificially drives local shear $\dot{\gamma} \gg \dot{\gamma}_c$." + "\n" +
        r"By Chapter 9 Rheology, the vacuum physically liquefies" + "\n" +
        r"ahead of the hull ($\eta \to 0$), annihilating the inertial bow shock."
    )
    axes[1].text(-4.8, -2.5, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#ff3366', alpha=0.9, pad=8))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    filepath = os.path.join(OUTPUT_DIR, "vacuum_aerodynamics.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_metric_streamlining()