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


def generate_glycine_spice(filepath="glycine_ave.cir"):
    """
    Generates a SPICE circuit model for Glycine (H).
    Backbone: [NH3+] - [CH2] - [COO-]
    R-Group: Just an H atom (simplest filter stub).
    """

    with open(filepath, "w") as f:
        f.write("* AVE SPICE Model: Glycine (Non-Chiral Baseline)\n")
        f.write("* Backbone topology mapped to L/C tensors\n\n")

        # ---------------------------------------------------------------------
        # 1. THE SOURCE: Amino Group (NH3+)
        # ---------------------------------------------------------------------
        f.write("* --- The Amino Source (NH3+) ---\n")
        f.write(f"V_amino in 0 SIN(0 {AMINO_SOURCE_VOLT} {AMINO_SOURCE_FREQ})\n")
        
        # The Nitrogen Mass
        l_nit = get_inductance("N")
        f.write(f"L_nh3 in n_amino {l_nit}pH\n")
        
        # N-C alpha bond
        c_nc = get_capacitance("C-N")
        f.write(f"C_nc n_amino n_alpha {c_nc}fF\n\n")

        # ---------------------------------------------------------------------
        # 2. THE CHASSIS: Alpha Carbon (C_alpha)
        # ---------------------------------------------------------------------
        f.write("* --- The Alpha Carbon (C-alpha) ---\n")
        l_calpha = get_inductance("C")
        f.write(f"L_alpha n_alpha n_alpha_out {l_calpha}pH\n")

        # The R-Group Stub (Just an H for Glycine)
        f.write("* R-Group Filter Stub (H attached to C-alpha)\n")
        c_ch = get_capacitance("C-H")
        l_h = get_inductance("H")
        f.write(f"C_rgroup_bond n_alpha n_rgroup {c_ch}fF\n")
        # In a real circuit, a purely passive terminal atom acts as a shunt capacitor to space 
        # or an inductor terminating to local parasitic ground.
        f.write(f"L_rgroup_mass n_rgroup 0 {l_h}pH\n\n")

        # C_alpha to Carboxyl Carbon bond
        c_cc = get_capacitance("C-C")
        f.write(f"C_cc n_alpha_out n_carboxyl_c {c_cc}fF\n\n")

        # ---------------------------------------------------------------------
        # 3. THE SINK: Carboxyl Group (COO-)
        # ---------------------------------------------------------------------
        f.write("* --- The Carboxyl Sink (COO-) ---\n")
        l_c_carboxyl = get_inductance("C")
        f.write(f"L_carboxyl_c n_carboxyl_c n_carboxyl_split {l_c_carboxyl}pH\n")
        
        # Double bonded Oxygen (O)
        c_co_double = get_capacitance("C=O")
        l_o_double = get_inductance("O")
        f.write(f"C_co_double n_carboxyl_split n_o_double {c_co_double}fF\n")
        f.write(f"L_o_double n_o_double 0 {l_o_double}pH\n")
        
        # Single bonded Oxygen (O-)
        c_co_single = get_capacitance("C-O")
        l_o_single = get_inductance("O")
        f.write(f"C_co_single n_carboxyl_split out {c_co_single}fF\n")
        f.write(f"L_o_single out n_term {l_o_single}pH\n")

        # The Thermodynamic Load Resistor (Z_0)
        f.write(f"R_load n_term 0 {CARBOXYL_LOAD_R}\n\n")

        # ---------------------------------------------------------------------
        # 4. SIMULATION DIRECTIVES
        # ---------------------------------------------------------------------
        f.write("* --- AC Simulation Directives ---\n")
        f.write(".ac dec 100 1G 1000G\n")
        f.write(".end\n")

    print(f"[Done] Generated SPICE Netlist: {filepath}")


def generate_alanine_spice(filepath="alanine_l_ave.cir"):
    """
    Generates a SPICE circuit model for L-Alanine.
    Backbone: [NH3+] - [CH(CH3)] - [COO-]
    R-Group: Methyl group (CH3).
    """

    with open(filepath, "w") as f:
        f.write("* AVE SPICE Model: L-Alanine (Chiral)\n")
        f.write("* Backbone topology mapped to L/C tensors\n\n")

        # 1. SOURCE
        f.write("* --- The Amino Source (NH3+) ---\n")
        f.write(f"V_amino in 0 SIN(0 {AMINO_SOURCE_VOLT} {AMINO_SOURCE_FREQ})\n")
        f.write(f"L_nh3 in n_amino {get_inductance('N')}pH\n")
        f.write(f"C_nc n_amino n_alpha {get_capacitance('C-N')}fF\n\n")

        # 2. CHASSIS
        f.write("* --- The Alpha Carbon (C-alpha) ---\n")
        f.write(f"L_alpha n_alpha n_alpha_out {get_inductance('C')}pH\n")

        # The R-Group Stub (CH3 for Alanine) - Adds significantly more mass/capacitance
        f.write("* R-Group Filter Stub (-CH3 attached to C-alpha)\n")
        f.write(f"C_rgroup_bond n_alpha n_r_carbon {get_capacitance('C-C')}fF\n")
        f.write(f"L_r_carbon n_r_carbon n_r_h_split {get_inductance('C')}pH\n")
        
        # 3 Hydrogens acting as a parallel capacitive array to ground
        f.write(f"C_ch1 n_r_h_split n_rh1 {get_capacitance('C-H')}fF\n")
        f.write(f"L_rh1 n_rh1 0 {get_inductance('H')}pH\n")
        f.write(f"C_ch2 n_r_h_split n_rh2 {get_capacitance('C-H')}fF\n")
        f.write(f"L_rh2 n_rh2 0 {get_inductance('H')}pH\n")
        f.write(f"C_ch3 n_r_h_split n_rh3 {get_capacitance('C-H')}fF\n")
        f.write(f"L_rh3 n_rh3 0 {get_inductance('H')}pH\n\n")

        f.write(f"C_cc n_alpha_out n_carboxyl_c {get_capacitance('C-C')}fF\n\n")

        # 3. SINK
        f.write("* --- The Carboxyl Sink (COO-) ---\n")
        f.write(f"L_carboxyl_c n_carboxyl_c n_carboxyl_split {get_inductance('C')}pH\n")
        
        f.write(f"C_co_double n_carboxyl_split n_o_double {get_capacitance('C=O')}fF\n")
        f.write(f"L_o_double n_o_double 0 {get_inductance('O')}pH\n")
        
        f.write(f"C_co_single n_carboxyl_split out {get_capacitance('C-O')}fF\n")
        f.write(f"L_o_single out n_term {get_inductance('O')}pH\n")

        f.write(f"R_load n_term 0 {CARBOXYL_LOAD_R}\n\n")

        # 4. DIRECTIVES
        f.write("* --- AC Simulation Directives ---\n")
        f.write(".ac dec 100 1G 1000G\n")
        f.write(".end\n")

    print(f"[Done] Generated SPICE Netlist: {filepath}")

if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "assets" / "sim_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    generate_glycine_spice(out_dir / "glycine_ave.cir")
    generate_alanine_spice(out_dir / "alanine_l_ave.cir")
    print("Run these files using ngspice to view the biological AC resonance profiles.")
