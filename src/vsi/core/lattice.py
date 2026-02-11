import numpy as np

class LatticeNode:
    """
    Represents a fundamental node in the Discrete Amorphous Manifold (MA).
    
    Attributes:
        L (float): Nodal Inductance (Henry).
        C (float): Nodal Capacitance (Farad).
        lp (float): Lattice Pitch (meters).
    """
    def __init__(self, L_node=1.256e-6, C_node=8.854e-12, lp=1.616e-35):
        self.L = L_node
        self.C = C_node
        self.lp = lp
        
        # The hardware slew rate limit (Speed of Light)
        # c = 1 / sqrt(LC)
        self.c_limit = 1.0 / np.sqrt(self.L * self.C)
        
        # The Hardware Saturation Frequency (Nyquist Limit)
        # w_sat = c / lp
        self.w_sat = self.c_limit / self.lp

    def get_dispersion_velocity(self, k, mode='flux'):
        """
        Calculates Group Velocity (vg) for a given wavenumber (k).
        
        Args:
            k (float): Wavenumber (radians/meter).
            mode (str): 'flux' (linear perturbation) or 'defect' (topological knot).
            
        Returns:
            float: Group velocity in m/s.
        """
        # Normalize k against the Nyquist limit (pi/lp)
        k_nyquist = np.pi / self.lp
        
        if k >= k_nyquist:
            return 0.0 # Hard hardware cutoff (GZK limit)

        if mode == 'flux':
            # --- FLUX MODE (Photons) ---
            # Flux is a sub-saturation perturbation. The lattice behaves as
            # a linear transmission line. Dispersion is negligible until
            # the immediate vicinity of the Nyquist limit.
            # Relation: omega = c * k
            # vg = d(omega)/dk = c
            return self.c_limit

        elif mode == 'defect':
            # --- DEFECT MODE (Matter) ---
            # A topological defect (knot) imposes a continuous load.
            # The node enters the saturation regime.
            # We map the wavenumber k to the Intrinsic Spin Frequency (w_spin).
            # Relation: v_g = c * sqrt(1 - (w_spin / w_sat)^2)
            
            # Simple coupling assumption: w_spin scales linearly with k
            w_spin = k * self.c_limit
            
            # Calculate the relativistic gamma factor inverse
            # If w_spin approaches w_sat, velocity drops to zero (Mass).
            saturation_ratio = (w_spin / self.w_sat)**2
            if saturation_ratio >= 1.0:
                return 0.0
            
            return self.c_limit * np.sqrt(1.0 - saturation_ratio)

        else:
            raise ValueError(f"Unknown propagation mode: {mode}")