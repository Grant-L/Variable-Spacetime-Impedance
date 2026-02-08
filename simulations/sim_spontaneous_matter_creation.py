import numpy as np
import matplotlib.pyplot as plt

def simulate_big_bang():
    """
    Simulates the 'Genesis' of matter via the Kibble-Zurek mechanism.
    Starts with a high-temperature (random) vacuum and 'quenches' it.
    """
    print("Initiating Big Bang (Random Phase Field)...")
    
    # 1. Setup the Early Universe
    N = 300
    L = 30.0
    dx = L / N
    
    # Initial State: "Hot" Universe = Complete Randomness
    # The phase angle is random everywhere between -pi and +pi
    psi = np.exp(1j * np.random.uniform(-np.pi, np.pi, (N, N)))
    
    # 2. The Cooling Process (Time Evolution)
    # We use Ginzburg-Landau again, but this time to 'order' the chaos.
    dt = 0.001
    steps = 1500
    
    print(f"Cooling Vacuum for {steps} epochs...")
    
    for t in range(steps):
        # Laplacian (Diffusion/Ordering force)
        lap = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) + 
               np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4*psi) / (dx**2)
        
        # GL Equation: The vacuum tries to relax to magnitude 1
        # But the random phase twists get "stuck" (Topological Defects)
        psi += dt * (lap + psi * (1.0 - np.abs(psi)**2))

    # 3. Visualization
    print("Rendering Cosmic Domains...")
    plt.figure(figsize=(12, 6))
    
    # Plot: The Emergence of Matter
    # We plot the Phase. Uniform colors are "Domains" (Empty Space).
    # The sharp swirls where colors meet are "Defects" (Particles).
    plt.imshow(np.angle(psi), cmap='twilight', origin='lower', extent=[-L/2, L/2, -L/2, L/2])
    
    plt.title(f"The Kibble-Zurek Mechanism: Spontaneous Matter Creation\n(Step {steps})")
    plt.colorbar(label="Vacuum Phase (Topology)")
    plt.xlabel("Cosmic Scale (Arbitrary Units)")
    plt.ylabel("Cosmic Scale (Arbitrary Units)")
    
    # Count the particles (defects)
    # We cheat slightly by counting zeros in the density
    density = np.abs(psi)
    defect_count = np.sum(density < 0.1)
    plt.text(-L/2 + 1, -L/2 + 1, f"Defects Trapped: ~{defect_count}", color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_big_bang()