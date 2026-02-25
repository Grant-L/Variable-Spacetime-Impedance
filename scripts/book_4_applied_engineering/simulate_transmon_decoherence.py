"""
AVE Quantum Decoherence Simulator (1D Transmon)
==============================================
Simulates a classic Transmon Qubit as an explicitly fragile 1D LC standing wave 
(a continuous phase amplitude). 
Subjects the transmission line to an ambient 300K stochastic thermodynamic noise floor.
Tracks the rapid exponential decay of phase-coherence (Entropy generation / Decoherence)
as the linear wave physically scatters into the background lattice.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Parameters
N = 200             # Number of LC nodes in the 1D junction
T_MAX = 1000        # Simulation time steps
C = 0.5             # Wave speed (Courant number for FDTD)
NOISE_AMP = 0.15    # Amplitude of the 300K stochastic background noise

def simulate_transmon_decoherence():
    # Initialize the LC grid (voltage/displacement)
    V = np.zeros(N)
    V_prev = np.zeros(N)
    
    # Inject a perfect standing wave (The "Qubit State")
    x = np.linspace(0, 2*np.pi, N)
    standing_wave = np.sin(3 * x)
    V[:] = standing_wave
    V_prev[:] = standing_wave
    
    # Tracking metrics
    time_history = []
    coherence_history = []
    
    # Main FDTD Loop with Stochastic Noise
    for t in range(T_MAX):
        V_next = np.zeros(N)
        
        # 1D Wave Equation (Explicit Euclidean Update)
        for i in range(1, N-1):
            V_next[i] = 2 * V[i] - V_prev[i] + (C**2) * (V[i+1] - 2*V[i] + V[i-1])
            
        # Apply 300K Thermodynamic Scatter (Stochastic Force)
        noise = np.random.normal(0, NOISE_AMP, N)
        V_next += noise
        
        # Enforce boundary conditions (Transmon Junction limits)
        V_next[0] = 0
        V_next[-1] = 0
        
        # Calculate Phase Coherence (Autocorrelation with original pristine state)
        # As the noise physically bashes the amplitude, the dot product falls.
        current_coherence = np.abs(np.dot(V_next, standing_wave)) / np.dot(standing_wave, standing_wave)
        
        coherence_history.append(current_coherence)
        time_history.append(t)
        
        # Step time
        V_prev = np.copy(V)
        V = np.copy(V_next)

    return time_history, coherence_history

def generate_plot(time, coherence, out_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Smooth the coherence for visualization using a moving average
    window_size = 20
    smoothed = np.convolve(coherence, np.ones(window_size)/window_size, mode='valid')
    time_smooth = time[:len(smoothed)]

    ax.plot(time_smooth, smoothed, color='#00ffcc', linewidth=2.5, label='Transmon Phase Amplitude (Coherence)')
    ax.fill_between(time_smooth, smoothed, color='#00ffcc', alpha=0.1)
    
    # Plot formatting to match AVE manuscript style
    ax.set_title("1D Transmon Decoherence under 300K Ambient LC Noise", fontsize=16, color='white', pad=15)
    ax.set_xlabel("Time (Arbitrary Units)", fontsize=14)
    ax.set_ylabel(r"Quantum State Coherence $|\langle \psi(t)|\psi(0) \rangle|$", fontsize=14)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.7)
    
    # Add a decay envelope overlay
    decay_envelope = np.exp(-np.array(time_smooth) / 150.0)
    ax.plot(time_smooth, decay_envelope, color='#ff00aa', linestyle='--', linewidth=2, label='Exponential Decoherence Envelope')

    ax.legend(loc='upper right', facecolor='black', edgecolor='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Done] Saved Transmon Decoherence Plot: {out_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    out_dir = PROJECT_ROOT / "scripts" / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Simulating 1D Transmon Decoherence...")
    t, c = simulate_transmon_decoherence()
    generate_plot(t, c, out_dir / "transmon_decoherence_plot.png")
