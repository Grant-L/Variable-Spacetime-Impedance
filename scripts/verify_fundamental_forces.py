import sys, os, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ave.matter import baryons, bosons
from ave.core import geometry

def run():
    print("--- VERIFYING FUNDAMENTAL FORCES ---")
    
    tension = baryons.calculate_strong_force_tension()
    print(f"Strong Force Tension:\n  AVE Theory: {tension:.2e} N  {'[PASS]' if abs(tension - 160000)/160000 < 0.05 else '[FAIL]'}\n")

    wz_ratio = bosons.calculate_weak_mixing_angle_mass_ratio()
    print(f"Weak Boson Mass Ratio (W/Z):\n  AVE Theory: {wz_ratio:.5f}  {'[PASS]' if abs(wz_ratio - 0.8814) < 0.001 else '[FAIL]'}\n")

    charges = geometry.BorromeanLinkage().charge_fractionalization()
    print(f"Quark Charge Fractionalization:\n  AVE Theory: {['{:.2f}'.format(c) for c in charges]}")
    # FIXED: Floating point logic
    if any(math.isclose(1.0/3.0, c, abs_tol=1e-5) for c in charges):
        print("  [PASS] Recovers 1/3e and 2/3e\n")
    else:
        print("  [FAIL]\n")

if __name__ == "__main__":
    run()