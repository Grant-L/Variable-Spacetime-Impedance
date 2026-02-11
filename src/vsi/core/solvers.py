import numpy as np
from core.constants import L_LATTICE, C_LATTICE, DX, LCT_C

def compute_laplacian(u):
    """
    Standard discrete 2D Laplacian for a nodal mesh.
    Represents the coupling between adjacent vacuum nodes[cite: 80, 538].
    """
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u)

def update_vacuum_field(u, u_prev, v_map, dt):
    """
    The Discrete Wave Equation Solver[cite: 84, 99].
    v_map: Local phase velocity (1/sqrt(L*C)), allowing for metric strain[cite: 570, 616].
    """
    lap = compute_laplacian(u)
    # The LCT update rule: u_next = 2u - u_prev + (v*dt/dx)^2 * laplacian
    u_next = 2 * u - u_prev + (v_map * dt / DX)**2 * lap
    return u_next