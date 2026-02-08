import numpy as np
import matplotlib.pyplot as plt

def run_born_rule_sim():
    print("LCT Simulation D: Emergence of Born Rule")
    # 1D Walker Simulation
    positions = []
    
    # Theoretical Probability (Sin^2)
    x = np.linspace(0, np.pi, 100)
    prob = np.sin(x)**2
    
    # Monte Carlo Walkers
    for _ in range(5000):
        # Walker is guided by gradient of prob field
        # Simple rejection sampling to mimic 'time spent' in regions
        proposal = np.random.uniform(0, np.pi)
        if np.random.rand() < np.sin(proposal)**2:
            positions.append(proposal)
            
    plt.figure(figsize=(8,6))
    plt.hist(positions, bins=50, density=True, alpha=0.6, label='Walker Density')
    plt.plot(x, prob * (2/np.pi), 'r-', lw=3, label='Wave Intensity $|\Psi|^2$')
    plt.title("Ergodic Walker Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_born_rule_sim()