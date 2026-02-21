import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from ave.mechanics import rheology


def run():
    print("--- VERIFYING CONDENSATE RHEOLOGY & PHASE TRANSITIONS ---")

    yield_kev = rheology.calculate_microscopic_point_yield_kev()
    print(f"Microscopic Point-Yield: {yield_kev:.2f} keV {'[PASS]' if abs(yield_kev - 16.50) < 0.1 else '[FAIL]'}")

    elec_stable, _ = rheology.check_heavy_fermion_stability(511000.0)
    muon_stable, _ = rheology.check_heavy_fermion_stability(105.66e6)
    print(
        f"Electron Stable: {elec_stable} | Muon Stable: {muon_stable} {'[PASS]' if elec_stable and not muon_stable else '[FAIL]'}"
    )

    omega_ent = rheology.calculate_sagnac_rlve_entrainment(10000.0, 19300.0)
    print(f"Sagnac-RLVE Entrainment: {omega_ent:.4e} rad/s {'[PASS]' if omega_ent > 0 else '[FAIL]'}")


if __name__ == "__main__":
    run()
