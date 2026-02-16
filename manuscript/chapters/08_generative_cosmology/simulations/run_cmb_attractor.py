import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/08_generative_cosmology/simulations/outputs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def simulate_cmb_equilibrium():
    print("Simulating CMB Thermodynamic Steady-State Attractor...")
    
    # Time array (arbitrary cosmological units)
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    
    # Constants
    H_0 = 1.0 # Expansion rate
    P_genesis = 4.0 # Power density of latent heat injected by new nodes
    
    # Simulate three different initial universes
    # 1. Hot Big Bang (Starts very hot)
    # 2. Cold Void (Starts at Absolute Zero)
    # 3. Equilibrium (Starts at the Attractor State)
    
    u_hot = [15.0]
    u_cold = [0.0]
    
    # Theoretical steady state: du/dt = -4H*u + P = 0 => u_eq = P / 4H
    u_eq_val = P_genesis / (4 * H_0)
    u_eq = [u_eq_val]
    
    for i in range(1, len(t)):
        # du/dt = -4*H_0*u + P_genesis
        du_hot = (-4 * H_0 * u_hot[-1] + P_genesis) * dt
        du_cold = (-4 * H_0 * u_cold[-1] + P_genesis) * dt
        du_eq = (-4 * H_0 * u_eq[-1] + P_genesis) * dt
        
        u_hot.append(u_hot[-1] + du_hot)
        u_cold.append(u_cold[-1] + du_cold)
        u_eq.append(u_eq[-1] + du_eq)
        
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax.set_facecolor('#050508')
    
    ax.plot(t, u_hot, color='#ff3366', lw=3, label='Initial State: Hot Big Bang')
    ax.plot(t, u_cold, color='#00ffcc', lw=3, label='Initial State: Absolute Zero Void')
    ax.plot(t, u_eq, color='white', lw=2, linestyle='--', label=r'AVE Steady-State ($T_{CMB} \approx 2.7$ K)')
    
    ax.set_title('The CMB as the Thermodynamic Attractor of Genesis', color='white', fontsize=14, weight='bold')
    ax.set_xlabel('Cosmological Time', color='white', fontsize=12)
    ax.set_ylabel('Radiation Energy Density ($u_{rad} \propto T^4$)', color='white', fontsize=12)
    
    textstr = (
        r"$\mathbf{Thermodynamic\ Balance:}$" + "\n" +
        r"$\dot{u}_{rad} = -4H_0 u_{rad} + \mathcal{P}_{latent} = 0$" + "\n\n" +
        "Latent Heat of vacuum crystallization exactly\n" +
        "balances adiabatic expansion cooling."
    )
    ax.text(0.5, 0.4, textstr, transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', edgecolor='white', alpha=0.8, pad=8))
    
    ax.tick_params(colors='white')
    ax.grid(True, color='#333333', ls='--')
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    filepath = os.path.join(OUTPUT_DIR, "cmb_thermodynamic_attractor.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    ensure_output_dir()
    simulate_cmb_equilibrium()