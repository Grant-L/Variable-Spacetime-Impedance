import sys
import numpy as np
sys.path.append('src')
import ave.solvers.coupled_resonator as cr

# Patch 4s coeffs in memory
cr._4S_P_COEFFS = [1.0, -0.75, 0.125, -1.0/192.0]
cr._4S_P2_COEFFS = [0.0] * 7
for _i in range(4):
    for _j in range(4):
        cr._4S_P2_COEFFS[_i + _j] += cr._4S_P_COEFFS[_i] * cr._4S_P_COEFFS[_j]

# Redefine the poly lambda because it captured the old list reference
cr._4S_POLY = lambda z: [cr._4S_P2_COEFFS[k] * z**k for k in range(7)]
cr._SUBSHELL_PARAMS[(4, 0)] = (cr._4S_NORM, cr._4S_POLY, cr._4S_RATE)

J_ana = cr._coulomb_J_sub(1, 0, 1.0, 4, 0, 1.0)
print(f"Fixed _coulomb_J_sub(1s, 4s): {J_ana:.4f} eV")

subs = cr._fill_subshells(19)
ze = cr._scf_z_eff_v2(19, subs)
E_neutral = cr.atom_total_energy_v3(19, subs, ze)

subs_ion = list(subs)
subs_ion[-1] = (4, 0, 0)
subs_ion = subs_ion[:-1]
ze_ion = cr._scf_z_eff_v2(19, subs_ion)
E_ion = cr.atom_total_energy_v3(19, subs_ion, ze_ion)

print(f"K IE: {E_ion - E_neutral:.3f} eV (Exp: 4.34 eV)")
