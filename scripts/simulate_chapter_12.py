import numpy as np
import matplotlib.pyplot as plt
from ave.electrodynamics import circuits

def run_transmission_test():
    """
    Protocol: Inject pulse into discrete LC grid. 
    Verify propagation velocity == c.
    Reproduces Figure 12.3
    """
    # Setup 100-node grid using ave.mechanics.moduli parameters
    grid = circuits.TransmissionLine1D(nodes=100)
    
    # Inject Gaussian Pulse
    grid.inject_pulse(t=0, magnitude=0.5)
    
    # Run Time Steps
    results = grid.step_forward(steps=1000)
    
    # Analyze velocity
    velocity = grid.measure_pulse_velocity()
    c_theory = const.c
    
    assert np.isclose(velocity, c_theory, rtol=1e-5)
    print("Spacetime Transmission Velocity Verified as 'c'")