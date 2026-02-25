import os
import re

repo = "/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance"

replacements = [
    ("scripts/book_3_macroscopic_continuity/simulate_black_hole_core.py", '"Localized Metric Strain $h_\perp$"', 'r"Localized Metric Strain $h_\perp$"'),
    ("scripts/book_3_macroscopic_continuity/simulate_oort_cloud_trap.py", '"Solar Dielectric Strain Field ($h_\perp$)"', 'r"Solar Dielectric Strain Field ($h_\perp$)"'),
    ("scripts/book_4_applied_engineering/simulate_achromatic_lens.py", '"AVE Lens: Zero Boundary Reflection ($Z \equiv Z_0$)"', 'r"AVE Lens: Zero Boundary Reflection ($Z \equiv Z_0$)"'),
    ("scripts/book_4_applied_engineering/simulate_ponder_acoustic_tank.py", 
        'f"2. Macroscopic Metric Thrust Rectification ($F_{{out}} \propto V^2 \cdot Q_{{acoustic}}$)"', 
        'rf"2. Macroscopic Metric Thrust Rectification ($F_{{out}} \propto V^2 \cdot Q_{{acoustic}}$)"'),
    ("scripts/book_4_applied_engineering/simulate_ponder_acoustic_tank.py", '"Time ($\mu$s)"', 'r"Time ($\mu$s)"'),
    ("scripts/book_4_applied_engineering/simulate_ponder_phased_array.py", '"Localized Energy Density ($\mu$J/$m^3$)"', 'r"Localized Energy Density ($\mu$J/$m^3$)"'),
    ("scripts/book_4_applied_engineering/simulate_mutual_inductance.py", "'General Relativity (Lense-Thirring $\Omega_{LT}$)'", "r'General Relativity (Lense-Thirring $\Omega_{LT}$)'"),
    ("scripts/book_5_topological_biology/simulate_nested_sleep_pods.py", '"Local Refractive Index $n_\perp(x)$"', 'r"Local Refractive Index $n_\perp(x)$"'),
    ("manuscript/scripts/simulate_electron_topology.py", '"Electron Defect: Topologically Locked $\mathcal{M}_A$ Phase Dislocation"', 'r"Electron Defect: Topologically Locked $\mathcal{M}_A$ Phase Dislocation"'),
    ("manuscript/scripts/simulate_higgs_rupture.py", '"Unitary Rupture Threshold ($\epsilon_{sat}$)"', 'r"Unitary Rupture Threshold ($\epsilon_{sat}$)"'),
    ("manuscript/scripts/simulate_higgs_rupture.py", '"Lattice Compliance ($C / \epsilon_0$)"', 'r"Lattice Compliance ($C / \epsilon_0$)"'),
    ("manuscript/scripts/simulate_higgs_rupture.py", '"Topological Rest Mass ($L / \mu_0$)"', 'r"Topological Rest Mass ($L / \mu_0$)"'),
    ("manuscript/scripts/simulate_electroweak_unification.py", '"Metric Restoring Impedance ($\Omega$)"', 'r"Metric Restoring Impedance ($\Omega$)"'),
    ("manuscript/scripts/simulate_chiral_network.py", '"Left-Handed Topology Input (Antimatter)\\nMechanically Blocked ($Z \to \infty$)"', 'r"Left-Handed Topology Input (Antimatter)\nMechanically Blocked ($Z \to \infty$)"'),
    ("manuscript/scripts/simulate_chiral_network.py", '"Because the discrete $\mathcal{M}_A$ LC network is physically constructed of right-handed helical Inductors,\\n"', 'r"Because the discrete $\mathcal{M}_A$ LC network is physically constructed of right-handed helical Inductors,\n"'),
    ("manuscript/scripts/simulate_chiral_network.py", '"geometric linkage frustration ($Z_{chiral} \to \infty$) and are deterministically scattered (attenuated) over sub-fermi bounds."', 'r"geometric linkage frustration ($Z_{chiral} \to \infty$) and are deterministically scattered (attenuated) over sub-fermi bounds."'),
    ("manuscript/scripts/simulate_atomic_orbitals.py", '"wave-equation for the discrete LC continuum ($c=\sqrt{1/\mu\epsilon}$).\\n"', 'r"wave-equation for the discrete LC continuum ($c=\sqrt{1/\mu\epsilon})$.\n"'),
    ("manuscript/scripts/simulate_atomic_orbitals.py", '"Wavefunctions ($\Psi$) are literal macroscopic mechanical acoustic phonons.\\n"', 'r"Wavefunctions ($\Psi$) are literal macroscopic mechanical acoustic phonons.\n"')
]

for filepath, search, replace in replacements:
    full_path = os.path.join(repo, filepath)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            content = f.read()
        
        updated = content.replace(search, replace)
        
        if updated != content:
            with open(full_path, "w") as f:
                f.write(updated)
            print(f"Patched: {filepath}")
        else:
            print(f"Not found: {filepath} -> {search[:30]}...")
