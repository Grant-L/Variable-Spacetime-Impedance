import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ave.core import constants as k
from ave.mechanics import moduli
from ave.matter import baryons
from ave.matter import bosons    # <--- ADDED BOSONS
from ave.cosmology import expansion

def run_verification():
    print("========================================")
    print("AVE SIMULATION VERIFICATION TRACE")
    print("========================================\n")

    print(f"[Axiom 4] Geometric Alpha Inverse:\n  Theory: {k.alpha_geom_inv:.6f} | Target: 137.036304\n  Status: {'[PASS]' if abs(k.alpha_geom_inv - 137.0363) < 0.001 else '[FAIL]'}\n")
    
    rho = moduli.calculate_bulk_density()
    print(f"[Mechanics] Vacuum Bulk Density:\n  Theory: {rho:.4e} kg/m^3 | Target: 7.92e+06 kg/m^3\n  Status: {'[PASS]' if abs(rho - 7.91e6) < 1e5 else '[FAIL]'}\n")

    nu = moduli.calculate_kinematic_viscosity()
    print(f"[Mechanics] Vacuum Kinematic Viscosity:\n  Theory: {nu:.4e} m^2/s | Target: 8.45e-07 m^2/s\n  Status: {'[PASS]' if abs(nu - 8.45e-7) < 1e-8 else '[FAIL]'}\n")

    f_strong = baryons.calculate_strong_force_tension()
    print(f"[Matter] Strong Force Tension:\n  Theory: {f_strong:.2f} N | Target: ~160,000 N\n  Status: {'[PASS]' if abs(f_strong - 160000) < 5000 else '[FAIL]'}\n")

    wz_ratio = bosons.calculate_weak_mixing_angle_mass_ratio()
    print(f"[Electroweak] W/Z Mass Ratio:\n  Theory: {wz_ratio:.6f} | Target: 0.881917\n  Status: {'[PASS]' if abs(wz_ratio - 0.8819) < 0.0001 else '[FAIL]'}\n")

    H_0_km = expansion.calculate_hubble_constant_limit()
    print(f"[Cosmology] Hubble Constant (H_inf):\n  Theory: {H_0_km:.2f} km/s/Mpc | Target: 69.32 km/s/Mpc\n  Status: {'[PASS]' if abs(H_0_km - 69.32) < 0.1 else '[FAIL]'}\n")

if __name__ == "__main__":
    run_verification()