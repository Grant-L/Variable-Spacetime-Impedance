"""
AVE MODULE 54: DISTRIBUTED LC TRANSMISSION LINE
-----------------------------------------------
Models a 1D cross-section of the spatial vacuum grid as a distributed 
LC transmission line. Demonstrates that a finite propagation velocity (c) 
emerges naturally from the distributed inductance and capacitance 
of the network nodes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

OUTPUT_DIR = "manuscript/chapters/13_spacetime_circuit_analysis/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_transmission_line():
    print("Simulating Vacuum LC Transmission Line...")
    
    N_nodes = 100
    L_val = 1.0
    C_val = 1.0
    
    def tline_ode(t, y):
        dy = np.zeros(2 * N_nodes)
        
        # Input Pulse at Node 0 
        V_in = np.exp(-((t - 5.0)**2) / 2.0)
        I_in_0 = (V_in - y[0]) / 1.0 # Source impedance R_s = 1.0
        
        dy[0] = (I_in_0 - y[1]) / C_val
        for i in range(1, N_nodes):
            dy[2*i] = (y[2*i - 1] - y[2*i + 1]) / C_val
        for i in range(N_nodes - 1):
            dy[2*i + 1] = (y[2*i] - y[2*i + 2]) / L_val
        dy[2*(N_nodes-1) + 1] = (y[2*(N_nodes-1)] - 0.0) / L_val
                
        return dy

    t_eval = np.linspace(0, 100.0, 1000)
    sol = solve_ivp(tline_ode, [0, 100.0], np.zeros(2 * N_nodes), t_eval=t_eval)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#0a0a12'); ax.set_facecolor('#0a0a12')
    
    nodes_to_plot = [10, 30, 50, 70, 90]
    colors = ['#E57373', '#FFD54F', '#4FC3F7', '#64B5F6', '#9575CD']
    
    for idx, node in enumerate(nodes_to_plot):
        ax.plot(sol.t, sol.y[2 * node], color=colors[idx], lw=2.0, label=f'Node {node}')
        
    ax.set_title(r'Signal Propagation in a Distributed LC Network ($v_g = 1/\sqrt{LC}$)', color='white', fontsize=13)
    ax.set_xlabel('Time (Arbitrary Units)', color='white')
    ax.set_ylabel('Amplitude (V)', color='white')
    ax.grid(True, ls=":", color='#333333')
    ax.tick_params(colors='lightgray')
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='gray', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#333333')

    filepath = os.path.join(OUTPUT_DIR, "ave_transmission_line.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_transmission_line()