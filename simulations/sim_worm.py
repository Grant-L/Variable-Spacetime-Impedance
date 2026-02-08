import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def simulate_wormhole_impedance():
    # Setup Paths (100 units long)
    x = np.linspace(0, 100, 500)
    
    # Physics Parameters
    # External: Stretched Lattice (High Inductance -> Slow)
    strain = 3.0 
    v_out = 1.0 / np.sqrt(1 + strain) 
    
    # Internal: Compressed Lattice (Low Inductance -> Fast)
    compression = 0.75
    v_in = 1.0 / np.sqrt(1 - compression)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('black')
    
    # Top Plot: External Space
    ax1.set_facecolor('black'); ax1.axis('off')
    ax1.set_title("External Space (Stretched Lattice)", color='white')
    # Draw Stretched Grid
    for gx in np.arange(0, 100, 5.0):
        ax1.axvline(gx, color='red', alpha=0.3, linewidth=1, linestyle='--')
    line_out, = ax1.plot([], [], 'r-', linewidth=3)
    
    # Bottom Plot: Wormhole
    ax2.set_facecolor('black'); ax2.axis('off')
    ax2.set_title("Wormhole Throat (Compressed Lattice)", color='white')
    # Draw Compressed Grid
    for gx in np.arange(0, 100, 1.5):
        ax2.axvline(gx, color='cyan', alpha=0.3, linewidth=0.5)
    line_in, = ax2.plot([], [], 'c-', linewidth=3)
    
    def update(frame):
        t = frame * 0.5
        # Calculate shifted pulses
        pulse_out = np.exp(-(x - (10 + v_out*t))**2 / 9.0)
        pulse_in = np.exp(-(x - (10 + v_in*t))**2 / 9.0)
        
        line_out.set_data(x, pulse_out)
        line_in.set_data(x, pulse_in)
        return [line_out, line_in]

    ani = FuncAnimation(fig, update, frames=100, blit=True)
    ani.save("wormhole_impedance.gif", writer=PillowWriter(fps=20))
    plt.show()

if __name__ == "__main__":
    simulate_wormhole_impedance()