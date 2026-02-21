"""
AVE Matter Library: Neutrino Sector
Implements the physics of Chiral Unknots (0_1 Topology).
Source: Chapter 7
"""
import sys
from pathlib import Path
import numpy as np
import math

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
        
        # The Neutrino is the 0_1 unknot. Lacking the 3D impedance self-intersection
        # of the Golden Torus, it does not acquire inertial mass from macroscopic structural strain.
        # Instead, its mass is strictly the resonant planar acoustic echo of the vacuum lattice.
        # These harmonics drop exponentially by powers of the Cosserat Poisson ratio (nu_vac = 2/7).
        
        from ave.mechanics import moduli
        nu_vac = moduli.get_poisson_ratio()
        
        # Exact topological harmonic indices
        # Tau: 4th harmonic
        # Muon: 5th harmonic
        # Electron: 8th harmonic
        flavor_harmonics = {
            "electron": 8,
            "muon": 5,
            "tau": 4
        }
        
        if flavor not in flavor_harmonics:
            raise ValueError("Flavor must be 'electron', 'muon', or 'tau'")
            
        n_harmonic = flavor_harmonics[flavor]
        
        # The mass eigenvalue is the electron mass scaled by the acoustic vacuum resonance
        m0 = k.M_E * (nu_vac ** n_harmonic)
        
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

def check_chirality_permission(chirality_str):
    """
    Module-level helper to evaluate the Chiral Exclusion Principle.
    """
    if chirality_str.lower() == "left":
        return True
    return False

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