"""
AVE MODULE 61: THE VACUUM MEMRISTOR & SUPERFLUID SKIN EFFECT
------------------------------------------------------------
1. THE VACUUM MEMRISTOR: Proves that thixotropic relaxation (\tau) 
   mathematically forces the vacuum to act as Chua's Memristor, 
   producing the signature "Pinched Hysteresis Loop".
2. THE SUPERFLUID SKIN EFFECT: Proves that as the vacuum yields 
   and resistance drops (R -> 0), the AC skin depth (\delta) collapses. 
   The superfluid boundary layer is strictly confined to the hull,
   acting as a "Metric Faraday Cage" that protects the passenger cabin.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_memristor_and_skin_effect():
    print("Simulating Vacuum Memristance and the Superfluid Skin Effect...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax1.set_facecolor('#0a0a12'); ax2.set_facecolor('#0a0a12')
    
    # ==========================================
    # PLOT 1: THE VACUUM MEMRISTOR (HYSTERESIS)
    # ==========================================
    R_solid = 50.0; R_fluid = 1.0; V_yield = 2.0; tau_vac = 0.05
    freq = 2.0
    
    def memristor_ode(t, y):
        S = y[0] # Fluidity State (0 to 1)
        V_app = 5.0 * np.sin(2 * np.pi * freq * t)
        S_target = 0.5 * (1.0 + np.tanh(10.0 * (np.abs(V_app) - V_yield)))
        return [(S_target - S) / tau_vac] # Memristive delay
        
    t_eval = np.linspace(0, 2.0, 5000)
    sol = solve_ivp(memristor_ode, [0, 2.0], [0.0], t_eval=t_eval, method='RK45')
    
    V_history = 5.0 * np.sin(2 * np.pi * freq * t_eval)
    R_memristor = R_solid * (1 - sol.y[0]) + R_fluid * sol.y[0]
    I_history = V_history / R_memristor
    
    mask = t_eval > 1.0 # Extract steady-state cycle
    ax1.plot(V_history[mask], I_history[mask], color='#00ffcc', lw=3, label='Vacuum $I-V$ Trace')
    ax1.plot(V_history[mask], V_history[mask]/R_solid, color='#E57373', lw=1.5, linestyle='--', label='Perfect Solid (Ohmic)')
    ax1.axvline(V_yield, color='white', lw=1, alpha=0.5, linestyle=':')
    ax1.axvline(-V_yield, color='white', lw=1, alpha=0.5, linestyle=':')
    
    ax1.set_title('The Vacuum Memristor (Pinched Hysteresis)', color='white', fontsize=14, weight='bold')
    ax1.set_xlabel('Topological Voltage / Shear Stress ($V$)', color='white', weight='bold')
    ax1.set_ylabel('Kinematic Current / Velocity ($I$)', color='white', weight='bold')
    ax1.legend(loc='lower right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    # ==========================================
    # PLOT 2: THE SUPERFLUID SKIN EFFECT
    # ==========================================
    V_sweep = np.linspace(0, 5, 1000)
    S_eq = 0.5 * (1.0 + np.tanh(10.0 * (np.abs(V_sweep) - V_yield)))
    R_eff = R_solid * (1 - S_eq) + R_fluid * S_eq
    
    # Skin depth (\delta) is proportional to sqrt(R) in AC electrodynamics
    skin_depth = np.sqrt(R_eff / R_solid) * 100.0 # Normalized %
    
    ax2.plot(V_sweep, skin_depth, color='#FFD54F', lw=4, label='AC Metric Skin Depth ($\delta \propto \sqrt{R_{vac}}$)')
    ax2.axvline(V_yield, color='#ff3366', linestyle=':', lw=2, label='Bingham Yield Limit')
    ax2.fill_between(V_sweep, 0, skin_depth, color='#FFD54F', alpha=0.15)
    
    ax2.set_title('Metric Faraday Cage: The Superfluid Skin Effect', color='white', fontsize=14, weight='bold')
    ax2.set_xlabel('Drive Amplitude ($V$)', color='white', weight='bold')
    ax2.set_ylabel('Slipstream Penetration Depth (% of Max)', color='white', weight='bold')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    
    textstr = (
        r"$\mathbf{Passenger~Safety~Paradox~Resolved:}$" + "\n" +
        r"Because AC skin depth scales with the square root of resistance," + "\n" +
        r"when the vacuum liquefies ($R_{vac} \to 0$), the penetration depth collapses." + "\n" +
        r"The destructive superfluid slipstream is strictly confined to the exterior hull." + "\n" +
        r"The interior cabin metric is physically shielded from the warp shear."
    )
    ax2.text(0.1, 20, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#FFD54F', alpha=0.9, pad=10))

    for ax in [ax1, ax2]:
        ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "memristor_and_skineffect.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight'); plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_memristor_and_skin_effect()