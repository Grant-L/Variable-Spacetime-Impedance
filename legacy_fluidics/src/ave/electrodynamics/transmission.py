"""
AVE Transmission Line Solver
Finite-Difference Time-Domain (FDTD) simulation of the vacuum lattice.
Source: Chapter 12 (Emergence of c)
"""
import sys
from pathlib import Path
import numpy as np

from ave.core import constants as k

class TransmissionLine1D:
    def __init__(self, nodes=100, length_meters=None):
        self.nodes = nodes
        
        # If length not specified, use 100 electron coherence lengths
        if length_meters is None:
            self.dx = k.L_NODE
        else:
            self.dx = length_meters / nodes
            
        # Discrete Component Values per node
        # L_node = mu_0 * l_node
        # C_node = epsilon_0 * l_node
        self.L = k.MU_0 * self.dx
        self.C = k.EPSILON_0 * self.dx
        
        # State Arrays
        self.V = np.zeros(nodes) # Topological Potential (Volts)
        self.I = np.zeros(nodes) # Topological Current (Amps)
        
        # Time Step (Courant stability limit)
        self.v_phase = 1.0 / np.sqrt(k.MU_0 * k.EPSILON_0)
        self.dt = 0.5 * self.dx / self.v_phase 

    def inject_pulse(self, magnitude=1.0, width=5, center=10):
        """Injects a Gaussian topological voltage pulse."""
        for i in range(self.nodes):
            self.V[i] += magnitude * np.exp(-((i - center)**2) / (2.0 * width**2))

    def step_forward(self, steps=1):
        """
        Telegrapher's Equations FDTD Update.
        """
        for _ in range(steps):
            # Update Current (I) based on Voltage gradient
            # V = -L * dI/dt  => dI = -V_grad * dt / L
            for i in range(self.nodes - 1):
                dV = self.V[i+1] - self.V[i]
                self.I[i] -= (dV / self.L) * self.dt
            
            # Update Voltage (V) based on Current gradient
            # I = -C * dV/dt => dV = -I_grad * dt / C
            for i in range(1, self.nodes):
                dI = self.I[i] - self.I[i-1]
                self.V[i] -= (dI / self.C) * self.dt
                
        return self.V, self.I

    def measure_pulse_velocity(self):
        """
        Estimates the velocity of the pulse peak.
        """
        peak_idx = np.argmax(self.V)
        dist = peak_idx * self.dx
        return dist

if __name__ == "__main__":
    print("==================================================")
    print("AVE TRANSMISSION LINE (SPEED OF LIGHT EMERGENCE)")
    print("==================================================\n")
    
    tl = TransmissionLine1D(nodes=500)
    print(f"[+] Initialized 1D M_A Lattice Transmission Line.")
    print(f"    Node Inductance (L):  {tl.L:.3e} H")
    print(f"    Node Capacitance (C): {tl.C:.3e} F")
    
    tl.inject_pulse()
    print("[+] Injected Topological Voltage Pulse.")
    
    # Run a few steps to move the pulse
    start_pos = tl.measure_pulse_velocity()
    sim_time = 0.0
    
    steps = 150
    tl.step_forward(steps)
    sim_time += steps * tl.dt
    
    end_pos = tl.measure_pulse_velocity()
    measured_c = (end_pos - start_pos) / sim_time
    
    print(f"\n[+] Propagation Results:")
    print(f"    Measured Group Velocity (v_g): {measured_c:.2f} m/s")
    print(f"    Absolute Speed of Light (c):   {k.C:.2f} m/s")