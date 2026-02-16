import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/07_gravitation_metric_refraction/simulations/outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_ponderomotive_force():
    print("Simulating Ponderomotive Equivalence Principle...")
    
    x = np.linspace(0.5, 10, 500)
    GM_c2 = 1.0
    
    # Refractive gradient (increasing density toward 0)
    n_x = 1.0 + (2 * GM_c2 / x)
    
    # Wave packet energy U = m_i * c^2 / n(x)
    m_i = 1.0
    U_x = m_i / n_x
    
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#050508')
    ax1.set_facecolor('#050508')
    
    color1 = '#00ffcc'
    ax1.set_xlabel('Distance from Mass Source', color='white', fontsize=12)
    ax1.set_ylabel('Refractive Index $n(x)$', color=color1, fontsize=12)
    ax1.plot(x, n_x, color=color1, lw=3, label='Vacuum Density $n(x)$')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#ff3366'
    ax2.set_ylabel('Wave Packet Energy $U(x)$', color=color2, fontsize=12)
    ax2.plot(x, U_x, color=color2, lw=3, linestyle='--', label='Stored Energy $U \propto m_i / n(x)$')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.suptitle('The Equivalence Principle via Wave Mechanics', color='white', fontsize=16, weight='bold')
    
    textstr = (
        r"$\mathbf{Ponderomotive\ Force:}$" + "\n" +
        r"$F_{grav} = -\nabla U = -\nabla \left( \frac{m_i c^2}{n(x)} \right)$" + "\n" +
        r"Force is identically proportional to $m_i$." + "\n" +
        r"Therefore: $m_g \equiv m_i$"
    )
    ax1.text(0.4, 0.4, textstr, transform=ax1.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', edgecolor='white', alpha=0.8, pad=8))

    filepath = os.path.join(OUTPUT_DIR, "ponderomotive_equivalence.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    simulate_ponderomotive_force()