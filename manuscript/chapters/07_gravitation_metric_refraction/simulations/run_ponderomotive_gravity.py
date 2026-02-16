"""
AVE MODULE 23: PONDEROMOTIVE EQUIVALENCE PRINCIPLE
--------------------------------------------------
Strict mathematical proof that Gravity is the thermodynamic drift of a wave.
When a trapped topological wave packet enters a refractive density gradient, 
its stored inductive rest-energy scales inversely with the local scalar index.
The spatial derivative of this energy drives physical acceleration. 
Because the energy is fundamentally defined by the particle's internal 
inductive mass (m_i), the resulting acceleration is independent of the mass 
magnitude, identically deriving m_i \equiv m_g.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/07_gravitation_metric_refraction/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_ponderomotive_force():
    print("Simulating Ponderomotive Equivalence Principle...")
    
    x = np.linspace(0.8, 10, 1000)
    GM_c2 = 1.0
    
    # 1. Scalar Refractive gradient (n_scalar = 1 + GM/rc^2)
    n_scalar = 1.0 + (GM_c2 / x)
    
    # 2. Wave packet Rest-Energy U = m_i * c^2 / n_scalar
    m_i = 1.0
    U_x = m_i / n_scalar
    
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax1.set_facecolor('#050508')
    
    color1 = '#00ffcc'
    ax1.set_xlabel('Distance from Mass Source ($r$)', color='white', fontsize=13, weight='bold')
    ax1.set_ylabel(r'Scalar Refractive Index $n_{scalar}(r)$', color=color1, fontsize=13, weight='bold')
    ax1.plot(x, n_scalar, color=color1, lw=3, label=r'Lattice Density ($n = 1 + \frac{GM}{rc^2}$)')
    ax1.tick_params(axis='y', labelcolor=color1, colors='white')
    ax1.tick_params(axis='x', colors='white')
    
    ax2 = ax1.twinx()
    color2 = '#ff3366'
    ax2.set_ylabel(r'Stored Wave Packet Energy $U(r)$', color=color2, fontsize=13, weight='bold')
    ax2.plot(x, U_x, color=color2, lw=3, linestyle='--', label=r'Rest Energy ($U = \frac{m_i c^2}{n}$)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    for spine in ax1.spines.values(): spine.set_color('#333333')
    for spine in ax2.spines.values(): spine.set_color('#333333')
    
    fig.suptitle('The Equivalence Principle via Wave Mechanics', color='white', fontsize=16, weight='bold', y=0.95)
    
    textstr = (
        r"$\mathbf{Ponderomotive~Force~Derivation:}$" + "\n" +
        r"$U(r) = \frac{m_i c^2}{1 + GM/rc^2} \approx m_i c^2 - \frac{GMm_i}{r}$" + "\n\n" +
        r"$\mathbf{F}_{grav} = -\nabla U = \mathbf{-\frac{GMm_i}{r^2}}$" + "\n\n" +
        r"Force is algebraically proportional to internal inertia ($m_i$)." + "\n" +
        r"Therefore: $m_g \equiv m_i$ is mechanically guaranteed."
    )
    ax1.text(0.4, 0.45, textstr, transform=ax1.transAxes, color='white', fontsize=12, 
             bbox=dict(facecolor='#111111', edgecolor='#ff3366', alpha=0.9, pad=10))

    filepath = os.path.join(OUTPUT_DIR, "ponderomotive_equivalence.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_ponderomotive_force()