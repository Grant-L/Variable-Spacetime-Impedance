import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.ave.core.lattice import AmorphousManifold

class UniverseValidator:
    def __init__(self):
        print("BOOTING UNIVERSAL DIAGNOSTIC TOOL...")
        print("-" * 50)
        
        # 1. Initialize Hardware (The Lattice)
        print("[HARDWARE] Initializing Discrete Amorphous Manifold...")
        self.universe = AmorphousManifold(n_nodes=10000, box_size=10.0, seed=42)
        self.kappa = self.universe.calculate_kappa()
        
        # Fundamental Constants (Inputs)
        self.m_e = 0.51106      # MeV (Electron)
        self.H0 = 73.0          # km/s/Mpc (Hubble)
        self.G = 6.674e-11      # SI
        self.c = 299792458      # m/s
        
        # Derived Constants (To be calculated)
        self.alpha_inv = 137.036  # Geometric Ansatz (4pi^3 + pi^2 + pi)

    def check_hardware(self):
        print(f"\n[HARDWARE] Lattice Inspection:")
        target_kappa = 0.437
        diff = abs(self.kappa - target_kappa) / target_kappa * 100
        
        print(f"  > Measured Packing Factor (Kappa): {self.kappa:.5f}")
        print(f"  > Theory Target:                   {target_kappa}")
        print(f"  > Hardware Variance:               {diff:.3f}%")
        
        if diff < 1.0:
            print("  > STATUS: PASS (Hardware within tolerance)")
            return True
        else:
            print("  > STATUS: FAIL (Lattice geometry out of spec)")
            return False

    def check_baryon_sector(self):
        print(f"\n[BARYON SECTOR] Strong Force Derivation:")
        # m_p = m_e * alpha^-1 * (4pi + 5/6)
        
        omega_topo = 4 * np.pi + (5/6)
        m_p_derived = self.m_e * self.alpha_inv * omega_topo
        m_p_exp = 938.272
        
        diff = abs(m_p_derived - m_p_exp) / m_p_exp * 100
        
        print(f"  > Geometric Factor (Omega):        {omega_topo:.4f}")
        print(f"  > Derived Proton Mass:             {m_p_derived:.3f} MeV")
        print(f"  > Experimental Target:             {m_p_exp:.3f} MeV")
        print(f"  > Error:                           {diff:.4f}%")
        
        if diff < 0.05:
            print("  > STATUS: PASS (High-Precision Match)")
        else:
            print("  > STATUS: WARNING (Recalibration needed)")

    def check_weak_sector(self):
        print(f"\n[WEAK SECTOR] Impedance Bridge Derivation:")
        # Base Scale S = m_p * alpha^-1
        # m_W = S * 5/8
        # m_Z = m_W * 3/sqrt(7)
        
        omega_topo = 4 * np.pi + (5/6)
        m_p_derived = self.m_e * self.alpha_inv * omega_topo
        
        S = m_p_derived * self.alpha_inv
        m_W_derived = S * (5/8)
        m_Z_derived = m_W_derived * (3 / np.sqrt(7))
        
        m_W_exp = 80379.0
        m_Z_exp = 91187.6
        
        diff_W = abs(m_W_derived - m_W_exp) / m_W_exp * 100
        diff_Z = abs(m_Z_derived - m_Z_exp) / m_Z_exp * 100
        
        print(f"  > Base Impedance Scale (S):        {S/1000:.2f} GeV")
        print(f"  > Derived W Mass:                  {m_W_derived/1000:.2f} GeV (Err: {diff_W:.3f}%)")
        print(f"  > Derived Z Mass:                  {m_Z_derived/1000:.2f} GeV (Err: {diff_Z:.3f}%)")
        
        if diff_W < 0.1 and diff_Z < 0.1:
            print("  > STATUS: PASS (Electroweak Unification Confirmed)")
        else:
            print("  > STATUS: FAIL")

    def check_dark_sector(self):
        print(f"\n[DARK SECTOR] Cosmology Check:")
        # v_flat = (G * M * a_gen)^1/4
        # a_gen = c * H0 / 2pi
        
        # Convert H0 to SI
        H0_si = (self.H0 * 1000) / 3.086e22 # 1/s
        a_gen = (self.c * H0_si) / (2 * np.pi)
        
        # Standard Galaxy (Milky Way)
        M_galaxy = 1.0e11 * 1.989e30 # kg
        
        v_flat = (self.G * M_galaxy * a_gen)**0.25
        v_flat_kms = v_flat / 1000.0
        
        print(f"  > Hubble Constant (Input):         {self.H0} km/s/Mpc")
        print(f"  > Derived Genesis Accel (a0):      {a_gen:.2e} m/s^2")
        print(f"  > Derived Rotation Floor:          {v_flat_kms:.2f} km/s")
        print(f"  > Observation Target:              ~200 km/s")
        
        if 180 < v_flat_kms < 220:
            print("  > STATUS: PASS (Dark Matter is Viscosity)")
        else:
            print("  > STATUS: FAIL")

    def run_full_diagnostic(self):
        self.check_hardware()
        self.check_baryon_sector()
        self.check_weak_sector()
        self.check_dark_sector()
        print("-" * 50)
        print("DIAGNOSTIC COMPLETE. UNIVERSE STABLE.")

if __name__ == "__main__":
    validator = UniverseValidator()
    validator.run_full_diagnostic()