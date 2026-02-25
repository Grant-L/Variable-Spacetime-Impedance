"""
SPICE Netlist Generator for Amino Acids
=======================================
Reads AVE organic topological mappings from `spice_organic_mapper.py`
and procedurally generates standard .cir SPICE netlists for 
amino acids (Glycine, Alanine, etc).

To run these models, you use ngspice:
$ ngspice
ngspice 1 -> source output.cir
ngspice 2 -> run
ngspice 3 -> plot v(out)
"""

import sys
import os
from pathlib import Path

# Fix path to import ave modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ave.mechanics.spice_organic_mapper import (
    get_inductance,
    get_capacitance,
    AMINO_SOURCE_FREQ,
    AMINO_SOURCE_VOLT,
    CARBOXYL_LOAD_R
)


def generate_amino_netlist(name, filename, r_group_lines):
    """
    Generates a SPICE circuit model for an amino acid using a standard backbone template
    and injecting the specific R-Group filter network.
    """
    with open(filename, "w") as f:
        f.write(f"* AVE SPICE Model: {name}\n")
        f.write("* Backbone topology mapped to L/C tensors\n\n")

        # 1. SOURCE
        f.write("* --- The Amino Source (NH3+) ---\n")
        f.write(f"V_amino in 0 SIN(0 {AMINO_SOURCE_VOLT} {AMINO_SOURCE_FREQ})\n")
        f.write(f"L_nh3 in n_amino {get_inductance('N')}pH\n")
        f.write(f"C_nc n_amino n_alpha {get_capacitance('C-N')}fF\n\n")

        # 2. CHASSIS
        f.write("* --- The Alpha Carbon (C-alpha) ---\n")
        f.write(f"L_alpha n_alpha n_alpha_out {get_inductance('C')}pH\n\n")

        # 3. R-GROUP
        f.write(f"* --- R-Group Filter Stub ({name}) ---\n")
        for line in r_group_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write(f"C_cc n_alpha_out n_carboxyl_c {get_capacitance('C-C')}fF\n\n")

        # 4. SINK
        f.write("* --- The Carboxyl Sink (COO-) ---\n")
        f.write(f"L_carboxyl_c n_carboxyl_c n_carboxyl_split {get_inductance('C')}pH\n")
        
        f.write(f"C_co_double n_carboxyl_split n_o_double {get_capacitance('C=O')}fF\n")
        f.write(f"L_o_double n_o_double 0 {get_inductance('O')}pH\n")
        
        f.write(f"C_co_single n_carboxyl_split out {get_capacitance('C-O')}fF\n")
        f.write(f"L_o_single out n_term {get_inductance('O')}pH\n")

        f.write(f"R_load n_term 0 {CARBOXYL_LOAD_R}\n\n")

        # 5. DIRECTIVES
        f.write("* --- AC Simulation Directives ---\n")
        f.write(".ac dec 100 1G 1000G\n")
        f.write(".end\n")

    print(f"[Done] Generated SPICE Netlist: {filename}")

def build_all_amino_acids(out_dir):
    """Defines the explicit R-Group circuitry and generates the files."""
    
    # 1. Glycine (-H)
    gly_r = [
        f"C_rgroup_bond n_alpha n_rgroup {get_capacitance('C-H')}fF",
        f"L_rgroup_mass n_rgroup 0 {get_inductance('H')}pH"
    ]
    generate_amino_netlist("Glycine", out_dir / "glycine_ave.cir", gly_r)

    # 2. Alanine (-CH3)
    ala_r = [
        f"C_rgroup_bond n_alpha n_r_carbon {get_capacitance('C-C')}fF",
        f"L_r_carbon n_r_carbon n_r_h_split {get_inductance('C')}pH",
        f"C_ch1 n_r_h_split n_rh1 {get_capacitance('C-H')}fF",
        f"L_rh1 n_rh1 0 {get_inductance('H')}pH",
        f"C_ch2 n_r_h_split n_rh2 {get_capacitance('C-H')}fF",
        f"L_rh2 n_rh2 0 {get_inductance('H')}pH",
        f"C_ch3 n_r_h_split n_rh3 {get_capacitance('C-H')}fF",
        f"L_rh3 n_rh3 0 {get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Alanine", out_dir / "alanine_ave.cir", ala_r)

    # 3. Valine (-CH(CH3)2)
    val_r = [
        f"C_r_beta_bond n_alpha n_r_beta {get_capacitance('C-C')}fF",
        f"L_r_beta n_r_beta n_r_beta_split {get_inductance('C')}pH",
        f"C_beta_h n_r_beta_split n_beta_h {get_capacitance('C-H')}fF",
        f"L_beta_h n_beta_h 0 {get_inductance('H')}pH",
        f"C_beta_g1 n_r_beta_split n_gamma1 {get_capacitance('C-C')}fF",
        f"L_gamma1 n_gamma1 0 {get_inductance('C') + 3*get_inductance('H')}pH", # Lumped methyl
        f"C_beta_g2 n_r_beta_split n_gamma2 {get_capacitance('C-C')}fF",
        f"L_gamma2 n_gamma2 0 {get_inductance('C') + 3*get_inductance('H')}pH"  # Lumped methyl
    ]
    generate_amino_netlist("L-Valine", out_dir / "valine_ave.cir", val_r)

    # 4. Serine (-CH2OH)
    ser_r = [
        f"C_r_beta_bond n_alpha n_r_beta {get_capacitance('C-C')}fF",
        f"L_r_beta n_r_beta n_r_beta_split {get_inductance('C')}pH",
        f"C_beta_h1 n_r_beta_split n_bh1 {get_capacitance('C-H')}fF",
        f"L_bh1 n_bh1 0 {get_inductance('H')}pH",
        f"C_beta_h2 n_r_beta_split n_bh2 {get_capacitance('C-H')}fF",
        f"L_bh2 n_bh2 0 {get_inductance('H')}pH",
        f"C_beta_o n_r_beta_split n_gamma_o {get_capacitance('C-O')}fF",
        f"L_gamma_o n_gamma_o n_gamma_o_split {get_inductance('O')}pH",
        f"C_oh n_gamma_o_split n_term_h {get_capacitance('O-H')}fF",
        f"L_term_h n_term_h 0 {get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Serine", out_dir / "serine_ave.cir", ser_r)

    # 5. Cysteine (-CH2SH)
    cys_r = [
        f"C_r_beta_bond n_alpha n_r_beta {get_capacitance('C-C')}fF",
        f"L_r_beta n_r_beta n_r_beta_split {get_inductance('C')}pH",
        f"C_beta_h1 n_r_beta_split n_bh1 {get_capacitance('C-H')}fF",
        f"L_bh1 n_bh1 0 {get_inductance('H')}pH",
        f"C_beta_h2 n_r_beta_split n_bh2 {get_capacitance('C-H')}fF",
        f"L_bh2 n_bh2 0 {get_inductance('H')}pH",
        f"C_beta_s n_r_beta_split n_gamma_s {get_capacitance('C-S')}fF",
        f"L_gamma_s n_gamma_s n_gamma_s_split {get_inductance('S')}pH",
        f"C_sh n_gamma_s_split n_term_h {get_capacitance('S-H')}fF",
        f"L_term_h n_term_h 0 {get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Cysteine", out_dir / "cysteine_ave.cir", cys_r)

    # 6. Phenylalanine (-CH2-Phenyl)
    phe_r = [
        f"C_r_beta_bond n_alpha n_r_beta {get_capacitance('C-C')}fF",
        f"L_r_beta n_r_beta n_r_beta_split {get_inductance('C')}pH",
        f"C_beta_h1 n_r_beta_split n_bh1 {get_capacitance('C-H')}fF",
        f"L_bh1 n_bh1 0 {get_inductance('H')}pH",
        f"C_beta_h2 n_r_beta_split n_bh2 {get_capacitance('C-H')}fF",
        f"L_bh2 n_bh2 0 {get_inductance('H')}pH",
        f"C_beta_ring n_r_beta_split n_ring {get_capacitance('C-C')}fF",
        f"L_ring n_ring 0 {6*get_inductance('C') + 5*get_inductance('H')}pH" # Lumped aromatic inertia
    ]
    generate_amino_netlist("L-Phenylalanine", out_dir / "phenylalanine_ave.cir", phe_r)

    # 7. L-Leucine (-CH2-CH(CH3)2)
    leu_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + get_inductance('H')}pH",
        f"C_gamma_d1 n_gamma_split n_delta1 {get_capacitance('C-C')}fF",
        f"L_delta1 n_delta1 0 {get_inductance('C') + 3*get_inductance('H')}pH",
        f"C_gamma_d2 n_gamma_split n_delta2 {get_capacitance('C-C')}fF",
        f"L_delta2 n_delta2 0 {get_inductance('C') + 3*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Leucine", out_dir / "leucine_ave.cir", leu_r)

    # 8. L-Isoleucine (-CH(CH3)-CH2-CH3)
    ile_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + get_inductance('H')}pH",
        f"C_beta_gamma1 n_beta_split n_gamma1 {get_capacitance('C-C')}fF",
        f"L_gamma1 n_gamma1 0 {get_inductance('C') + 3*get_inductance('H')}pH",
        f"C_beta_gamma2 n_beta_split n_gamma2 {get_capacitance('C-C')}fF",
        f"L_gamma2 n_gamma2 n_gamma2_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma2_delta n_gamma2_split n_delta {get_capacitance('C-C')}fF",
        f"L_delta n_delta 0 {get_inductance('C') + 3*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Isoleucine", out_dir / "isoleucine_ave.cir", ile_r)

    # 9. L-Threonine (-CH(OH)-CH3)
    thr_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + get_inductance('H')}pH",
        f"C_beta_o n_beta_split n_gamma_o {get_capacitance('C-O')}fF",
        f"L_gamma_o n_gamma_o 0 {get_inductance('O') + get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma_c {get_capacitance('C-C')}fF",
        f"L_gamma_c n_gamma_c 0 {get_inductance('C') + 3*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Threonine", out_dir / "threonine_ave.cir", thr_r)

    # 10. L-Methionine (-CH2-CH2-S-CH3)
    met_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma_s n_gamma_split n_delta_s {get_capacitance('C-S')}fF",
        f"L_delta_s n_delta_s n_s_split {get_inductance('S')}pH",
        f"C_s_epsilon n_s_split n_epsilon {get_capacitance('C-S')}fF",
        f"L_epsilon n_epsilon 0 {get_inductance('C') + 3*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Methionine", out_dir / "methionine_ave.cir", met_r)

    # 11. L-Proline (Cyclic stub approximation)
    pro_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma_delta n_gamma_split n_delta {get_capacitance('C-C')}fF",
        f"L_delta n_delta 0 {get_inductance('C') + 2*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Proline", out_dir / "proline_ave.cir", pro_r)

    # 12. L-Tyrosine (-CH2-Phenyl-OH)
    tyr_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_ring n_beta_split n_ring {get_capacitance('C-C')}fF",
        f"L_ring n_ring n_ring_split {6*get_inductance('C') + 4*get_inductance('H')}pH",
        f"C_ring_o n_ring_split n_oh {get_capacitance('C-O')}fF",
        f"L_oh n_oh 0 {get_inductance('O') + get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Tyrosine", out_dir / "tyrosine_ave.cir", tyr_r)

    # 13. L-Tryptophan (-CH2-Indole)
    trp_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_ring n_beta_split n_ring {get_capacitance('C-C')}fF",
        f"L_ring n_ring 0 {9*get_inductance('C') + 6*get_inductance('H') + get_inductance('N')}pH" # Indole mass
    ]
    generate_amino_netlist("L-Tryptophan", out_dir / "tryptophan_ave.cir", trp_r)

    # 14. L-Aspartate (-CH2-COO-)
    asp_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma_coo n_gamma 0 {get_inductance('C') + 2*get_inductance('O')}pH" # Lumped carboxyl
    ]
    generate_amino_netlist("L-Aspartate", out_dir / "aspartate_ave.cir", asp_r)

    # 15. L-Glutamate (-CH2-CH2-COO-)
    glu_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma_delta n_gamma_split n_delta {get_capacitance('C-C')}fF",
        f"L_delta_coo n_delta 0 {get_inductance('C') + 2*get_inductance('O')}pH"
    ]
    generate_amino_netlist("L-Glutamate", out_dir / "glutamate_ave.cir", glu_r)

    # 16. L-Asparagine (-CH2-CONH2)
    asn_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma_amide n_gamma 0 {get_inductance('C') + get_inductance('O') + get_inductance('N') + 2*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Asparagine", out_dir / "asparagine_ave.cir", asn_r)

    # 17. L-Glutamine (-CH2-CH2-CONH2)
    gln_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma_delta n_gamma_split n_delta {get_capacitance('C-C')}fF",
        f"L_delta_amide n_delta 0 {get_inductance('C') + get_inductance('O') + get_inductance('N') + 2*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Glutamine", out_dir / "glutamine_ave.cir", gln_r)

    # 18. L-Histidine (-CH2-Imidazole)
    his_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_ring n_beta_split n_ring {get_capacitance('C-C')}fF",
        f"L_ring n_ring 0 {3*get_inductance('C') + 2*get_inductance('N') + 3*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Histidine", out_dir / "histidine_ave.cir", his_r)

    # 19. L-Lysine (-CH2-CH2-CH2-CH2-NH3+)
    lys_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma_delta n_gamma_split n_delta {get_capacitance('C-C')}fF",
        f"L_delta n_delta n_delta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_delta_eps n_delta_split n_eps {get_capacitance('C-C')}fF",
        f"L_eps n_eps n_eps_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_eps_zeta n_eps_split n_zeta {get_capacitance('C-N')}fF",
        f"L_zeta_nh3 n_zeta 0 {get_inductance('N') + 3*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Lysine", out_dir / "lysine_ave.cir", lys_r)

    # 20. L-Arginine (-CH2-CH2-CH2-NH-C(NH2)2+)
    arg_r = [
        f"C_r_beta n_alpha n_beta {get_capacitance('C-C')}fF",
        f"L_beta n_beta n_beta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_beta_gamma n_beta_split n_gamma {get_capacitance('C-C')}fF",
        f"L_gamma n_gamma n_gamma_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_gamma_delta n_gamma_split n_delta {get_capacitance('C-C')}fF",
        f"L_delta n_delta n_delta_split {get_inductance('C') + 2*get_inductance('H')}pH",
        f"C_delta_eps n_delta_split n_eps {get_capacitance('C-N')}fF",
        f"L_eps_guanidino n_eps 0 {get_inductance('C') + 3*get_inductance('N') + 5*get_inductance('H')}pH"
    ]
    generate_amino_netlist("L-Arginine", out_dir / "arginine_ave.cir", arg_r)

if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "assets" / "sim_outputs" / "spice_models"
    os.makedirs(out_dir, exist_ok=True)
    
    build_all_amino_acids(out_dir)
    print("Run these files using ngspice to view the biological AC resonance profiles.")
