import ave.solvers.coupled_resonator as cr

subs = cr._fill_subshells(19)
ze = cr._scf_z_eff_v2(19, subs)

subs_ion = list(subs)
subs_ion[-1] = (4, 0, 0)
subs_ion = subs_ion[:-1]
ze_ion = cr._scf_z_eff_v2(19, subs_ion)

def print_energy_breakdown(Z, subs, ze_list):
    E_nonint = 0.0
    E_J = 0.0
    electrons = []
    
    for idx, (n, l, count) in enumerate(subs):
        ze = ze_list[idx]
        for _ in range(count):
            E_comp = (ze**2 - 2.0 * Z * ze) * cr._RY_EV / n**2
            E_nonint += E_comp
            electrons.append((n, l, ze))
            
    for i in range(len(electrons)):
        for j in range(i + 1, len(electrons)):
            n_i, l_i, ze_i = electrons[i]
            n_j, l_j, ze_j = electrons[j]
            E_J += cr._coulomb_J_sub(n_i, l_i, ze_i, n_j, l_j, ze_j)
            
    print(f"Non-interacting: {E_nonint:.2f}")
    print(f"Coulomb J:       {E_J:.2f}")
    print(f"Total:           {E_nonint + E_J:.2f}")

print("Neutral K:")
print_energy_breakdown(19, subs, ze)
print("Ion K+:")
print_energy_breakdown(19, subs_ion, ze_ion)
