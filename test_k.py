import ave.solvers.coupled_resonator as cr

subs = cr._fill_subshells(19)
print("K (neutral):", subs)
ze = cr._scf_z_eff_v2(19, subs)
print("Ze neutral:", ze)
E = cr.atom_total_energy_v2(19, subs, ze)
print("E neutral:", E)

# Ion
subs_ion = list(subs)
subs_ion[-1] = (4, 0, 0) # Potassium 4s1 goes to 4s0
subs_ion = subs_ion[:-1]
print("K+ (ion):", subs_ion)
ze_ion = cr._scf_z_eff_v2(19, subs_ion)
print("Ze ion:", ze_ion)
E_ion = cr.atom_total_energy_v2(19, subs_ion, ze_ion)
print("E ion:", E_ion)
print("IE:", E_ion - E)
