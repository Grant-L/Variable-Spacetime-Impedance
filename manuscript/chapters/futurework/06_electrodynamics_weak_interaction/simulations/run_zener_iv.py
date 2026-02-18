"""
AVE MODULE 52: EQUIVALENT CIRCUIT COMPONENT LIBRARY (REVISED)
-------------------------------------------------------------
CORRECTION: Applies Stress-to-Voltage isomorphism to the Bingham 
element. Yield is strictly triggered by Voltage (V > V_yield), 
making the vacuum perfectly isomorphic to a Zener/TVS Diode.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_zener_iv_curve():
    print("Generating Corrected Vacuum Zener Component Profile...")
    V = np.linspace(-5, 5, 1000)
    V_yield = 2.0
    R_solid = 50.0
    R_fluid = 0.5
    
    I = np.zeros_like(V)
    # Solid Regime (High Resistance)
    solid_mask = np.abs(V) <= V_yield
    I[solid_mask] = V[solid_mask] / R_solid
    
    # Fluid Regime (Superfluid Avalanche)
    pos_fluid = V > V_yield
    I[pos_fluid] = (V[pos_fluid] - V_yield) / R_fluid + (V_yield / R_solid)
    neg_fluid = V < -V_yield
    I[neg_fluid] = (V[neg_fluid] + V_yield) / R_fluid - (V_yield / R_solid)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax.set_facecolor('#0a0a12')
    ax.plot(V, I, color='#FFD54F', lw=3)
    ax.axvspan(-V_yield, V_yield, color='#E57373', alpha=0.2, label=r'Solid Regime ($R_{solid}$)')
    ax.axvspan(V_yield, 5, color='#4FC3F7', alpha=0.2, label=r'Superfluid Avalanche ($R_{fluid}$)')
    ax.axvspan(-5, -V_yield, color='#4FC3F7', alpha=0.2)
    ax.set_title('Vacuum TVS Zener Diode $I(V)$', color='white', fontsize=14, weight='bold')
    ax.set_xlabel('Topological Voltage / Shear Stress ($V$)', color='white', weight='bold')
    ax.set_ylabel('Kinematic Current / Velocity ($I$)', color='white', weight='bold')
    ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='lightgray')
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='white', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "zener_iv_curve.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__": simulate_zener_iv_curve()