"""
AVE Lepton Sector (Dynamic Matter Library)
Implements fundamental fermions as live, dynamic topological solitons.
Source: Chapter 5 (Topological Matter) & Chapter 12.5 (The Confinement Bubble)
"""
import sys
from pathlib import Path
import numpy as np
import math

from ave.core import constants as k
from ave.core import geometry
from ave.matter.solitons import TopologicalSoliton

class Electron(TopologicalSoliton):
    """
    The fundamental ground-state fermion.
    A continuous 3_1 Trefoil Knot bounded exactly by the Golden Torus.
    """
    def __init__(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), is_antimatter=False):
        charge = k.E_CHARGE if is_antimatter else -k.E_CHARGE
        super().__init__(rest_mass_kg=k.M_E, charge_coulombs=charge, position=position, velocity=velocity, orientation=orientation)
        
        self.is_antimatter = is_antimatter
        self.name = "Positron" if is_antimatter else "Electron"
        
        # Golden Torus Geometry
        self.torus = geometry.GoldenTorus()
        self.R = self.torus.physical_major_radius()
        self.r = self.torus.physical_minor_radius()
        
        # Chirality dictates the phase twist direction (+/- Z axis)
        self.chirality = -1.0 if is_antimatter else 1.0
        
        # Absolute structural core radius
        self.r_core = k.ALPHA_GEOM * k.L_NODE
        
        # Ropelength limit for stability checks
        self.knot_length_m = 16.37 * k.L_NODE

    def get_parametric_core(self, resolution=500):
        """
        Generates the dense parametric points of a (3,2) Torus Knot.
        Uses the exact R and r bounded by the dielectric limit.
        """
        t = np.linspace(0, 2 * math.pi, resolution)
        
        p = 3.0 # Poloidal windings
        q = 2.0 # Toroidal windings
        
        x = (self.R + self.r * np.cos(p * t)) * np.cos(q * t)
        y = (self.R + self.r * np.cos(p * t)) * np.sin(q * t)
        z = -self.r * np.sin(p * t) * self.chirality
        
        # Local space curve (rotation & translation handled by parent class)
        return np.column_stack((x, y, z))

    def get_local_impedance_profile(self, distance_from_center):
        """
        Calculates the spatial varactor limit mapping the Confinement Bubble.
        Demonstrates the Pauli Exclusion limit natively.
        Source: Eq 12.10 - 12.12
        """
        if distance_from_center <= self.r_core:
            # Inside the saturated core, C -> Infinity, Z -> 0 Ohms
            return 0.0, -1.0  # (Impedance, Reflection Coefficient)
            
        strain_ratio = self.r_core / distance_from_center
        C_eff = k.EPSILON_0 / math.sqrt(1.0 - strain_ratio**2)
        Z_local = math.sqrt(k.MU_0 / C_eff)
        
        # Reflection against ambient vacuum (Z_0)
        gamma = (Z_local - k.Z_0) / (Z_local + k.Z_0)
        return Z_local, gamma

    def check_stability(self):
        """
        Evaluates Leaky Cavity instability (Chapter 4.3.1).
        Checks if internal static tension exceeds the absolute 1D Yield Limit.
        """
        E_joules = self.m0 * k.C**2
        internal_tension_N = E_joules / self.knot_length_m
        is_stable = internal_tension_N <= k.T_EM
        return is_stable, internal_tension_N, k.T_EM

class Positron(Electron):
    """Physically identical to the electron, but inverted topological helicity."""
    def __init__(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0)):
        super().__init__(position, velocity, orientation, is_antimatter=True)

class Muon(Electron):
    """
    A 2nd-generation resonance loaded onto the identical Golden Torus topology.
    It is strictly unstable because its inductive core shatters the local yield limit.
    """
    def __init__(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), is_antimatter=False):
        super().__init__(position, velocity, orientation, is_antimatter)
        self.m0 = 206.768283 * k.M_E
        self.name = "Anti-Muon" if is_antimatter else "Muon"
        
        # Dynamically recalculate base class inheritance limits
        from ave.core import conversion
        self.L0 = conversion.mass_to_inductance(self.m0)
        self.omega_c = (self.m0 * k.C**2) / k.H_BAR
        self._update_kinematics()

class Tau(Electron):
    def __init__(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), is_antimatter=False):
        super().__init__(position, velocity, orientation, is_antimatter)
        self.m0 = 3477.23 * k.M_E
        self.name = "Anti-Tau" if is_antimatter else "Tau"
        
        from ave.core import conversion
        self.L0 = conversion.mass_to_inductance(self.m0)
        self.omega_c = (self.m0 * k.C**2) / k.H_BAR
        self._update_kinematics()

def calculate_theoretical_alpha():
    """Legacy compatibility for test suite"""
    torus = geometry.GoldenTorus()
    return 1.0 / torus.topological_impedance()


if __name__ == "__main__":
    print("==================================================")
    print("AVE MATTER LIBRARY: ELECTRON INSTANTIATION")
    print("==================================================\n")
    
    # 1. Instantiate the Electron at 0.8c
    e_minus = Electron(position=(0, 0, 0), velocity=(0.8 * k.C, 0, 0))
    print(f"[+] Instantiated {e_minus.name} (3_1 Trefoil)")
    print(f"    -> Rest Mass:       {e_minus.m0:.3e} kg")
    print(f"    -> Velocity:        {np.linalg.norm(e_minus.vel)/k.C:.2f} c")
    print(f"    -> Lorentz Gamma:   {e_minus.gamma:.4f}")
    print(f"    -> Relativistic M:  {e_minus.dynamic_mass:.3e} kg")
    
    # 2. Check Static Tension
    stable, tension, yield_limit = e_minus.check_stability()
    print(f"\n[+] Structural Integrity Check")
    print(f"    -> Static Tension:  {tension:.4f} N (Limit: {yield_limit:.4f} N)")
    print(f"    -> State:           {'STABLE' if stable else 'UNSTABLE (LEAKY CAVITY)'}")
    
    # 3. Simulate Relativistic Zitterbewegung
    print("\n[+] Dynamic LC Resonance State (Stepping forward dt=1e-22 s)")
    e_minus.step_kinematics(dt_lab=1e-22)
    state = e_minus.get_lc_resonance_state()
    print(f"    -> Phase Angle:   {state['phase_angle']:.4f} rad (Proper Time active)")
    print(f"    -> Inductive:     {state['E_magnetic_inductive']:.4e} J")
    print(f"    -> Capacitive:    {state['E_electric_capacitive']:.4e} J")

    # 4. Render a 1D slice of the 3D Impedance Fields
    print("\n[+] Scanning Topological Field Gradients (Pauli Exclusion Limit)")
    print(f"{'Distance (l_node)':<20} | {'Local Impedance (Z)'}")
    print("-" * 45)
    
    distances = [10.0 * k.L_NODE, 2.0 * k.L_NODE, 1.1 * k.L_NODE, e_minus.r_core]
    for d in distances:
        z_ohms, gamma_refl = e_minus.get_local_impedance_profile(d)
        marker = " <-- PAULI BOUNDARY (Perfect Mirror)" if z_ohms < 50.0 else ""
        print(f"{d/k.L_NODE:<20.2f} | {z_ohms:>8.2f} Ohms (Γ={gamma_refl:>5.2f}) {marker}")

    print("\n==================================================")