"""
AVE Matter Library: Neutrino Sector
Implements the physics of Chiral Unknots (0_1 Topology).
Source: Chapter 7
"""
import sys
from pathlib import Path
import numpy as np
import math

# Add src directory to path if running as script
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k
from ave.matter.solitons import TopologicalSoliton

class Neutrino(TopologicalSoliton):
    """
    The 0_1 Unknot.
    As an unknot, it completely lacks the Golden Torus dielectric self-intersection.
    Therefore, its mass is strictly a linear acoustic torsional harmonic, millions
    of times lighter than the electron.
    """
    def __init__(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), flavor="electron", is_antimatter=False):
        
        # Acoustic mass harmonics (Rough approximations for simulation limits)
        flavor_masses = {
            "electron": 0.12e-6 * k.M_E, # ~0.06 eV
            "muon": 0.33e-6 * k.M_E,     # ~0.17 eV
            "tau": 36.0e-6 * k.M_E       # ~18.3 eV
        }
        
        if flavor not in flavor_masses:
            raise ValueError("Flavor must be 'electron', 'muon', or 'tau'")
            
        m0 = flavor_masses[flavor]
        
        super().__init__(rest_mass_kg=m0, charge_coulombs=0.0, position=position, velocity=velocity, orientation=orientation)
        
        self.flavor = flavor
        self.is_antimatter = is_antimatter
        self.name = f"{'Anti-' if is_antimatter else ''}{flavor.capitalize()} Neutrino"
        
        # Chirality dictates propagation permission through the trace-reversed vacuum
        self.chirality = "right" if is_antimatter else "left"

    def get_parametric_core(self, resolution=100):
        """
        The 0_1 Unknot is a simple loop.
        Because it does not self-intersect or weave, it does not trigger Axiom 4 saturation.
        """
        t = np.linspace(0, 2 * math.pi, resolution)
        r_unknot = 1.0 * k.L_NODE
        
        x = r_unknot * np.cos(t)
        y = r_unknot * np.sin(t)
        z = np.zeros_like(t)
        
        return np.column_stack((x, y, z))

    def check_chirality_permission(self):
        """
        Evaluates the Chiral Exclusion Principle (Chapter 7).
        Left-handed neutrinos propagate freely. Right-handed neutrinos 
        shear against the gamma_c Cosserat vacuum microrotation and are evanescent.
        """
        if self.chirality == "left":
            return True, "Propagating Wave (omega^2 > 0)"
        else:
            return False, "Evanescent Wave / Anderson Localization (omega^2 < 0)"


if __name__ == "__main__":
    print("==================================================")
    print("AVE MATTER LIBRARY: NEUTRINO INSTANTIATION")
    print("==================================================\n")
    
    nu_e = Neutrino(velocity=(0.999999 * k.C, 0, 0), flavor="electron")
    print(f"[+] Instantiated {nu_e.name} (0_1 Unknot)")
    print(f"    -> Rest Mass:       {nu_e.m0:.3e} kg")
    print(f"    -> Charge:          {nu_e.charge:.1f} C")
    print(f"    -> Velocity:        {np.linalg.norm(nu_e.vel)/k.C:.5f} c")
    
    permitted, reason = nu_e.check_chirality_permission()
    print(f"    -> Vacuum State:    {reason}")
    
    anti_nu_e = Neutrino(velocity=(0.999999 * k.C, 0, 0), flavor="electron", is_antimatter=True)
    print(f"\n[+] Instantiated {anti_nu_e.name}")
    permitted, reason = anti_nu_e.check_chirality_permission()
    print(f"    -> Vacuum State:    {reason}")
    print("\n==================================================")