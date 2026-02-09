"""
LCT Hardware Constants: 
Defining the Vacuum as a Discrete LC Lattice.
"""
import scipy.constants as const
import numpy as np

# --- Standard Physical Constants ---
C_LIGHT = const.c
G_GRAV = const.G
H_BAR = const.hbar
EPSILON_0 = const.epsilon_0  # Lattice Capacitance equivalent 
MU_0 = const.mu_0            # Lattice Inductance equivalent [cite: 38, 69]

# --- LCT Hardware Variables (Table 1) ---
L_LATTICE = MU_0             # [H/m] Vacuum Permeability [cite: 38]
C_LATTICE = EPSILON_0        # [F/m] Vacuum Permittivity [cite: 38]
DX = 1.616255e-35            # [m] Lattice Pitch (Planck Length) [cite: 38]

# --- Emergent Hardware Properties ---
# Characteristic Impedance: Z0 = sqrt(L/C) [cite: 38]
Z_0 = np.sqrt(L_LATTICE / C_LATTICE) 

# Nyquist Limit: w_cutoff = 2 / sqrt(LC) [cite: 38, 134]
W_CUTOFF = 2.0 / np.sqrt(L_LATTICE * C_LATTICE)

# --- Derived LCT Scaling Factors ---
# Mass as Bandwidth Saturation: Relates mass to frequency cutoff [cite: 144]
PLANCK_MASS_LCT = (H_BAR * W_CUTOFF) / (C_LIGHT**2)