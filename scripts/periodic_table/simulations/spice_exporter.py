"""
AVE MODULE: SPICE NETLIST GENERATOR
---------------------------------------------
Translates the 3D topological arrays produced by `simulate_element.py` into 
executable formal SPICE netlists (.cir). 

This allows users to load elements (Hydrogen through Nitrogen) directly 
into LTspice, Ngspice, or Xyce as classical Resonant LC Networks to 
simulate their AC properties, Q-factors, and scattering/resonance parameters.
"""
import os
import numpy as np

# A baseline SPICE sub-circuit representing a single localized Nucleon (defect).
# It's an ultra-high Q parallel LC tank. 
# We use standard 1uH and 1pF values to define a normalized 159.1 MHz resonant baseline.
NUCLEON_SUBCKT = """
* -----------------------------------------------------------------
* NUCLEON (Baseline 6^3_2 Topological Defect)
* -----------------------------------------------------------------
.SUBCKT NUCLEON IN OUT
L_CORE IN OUT 1uH
C_CORE IN OUT 1pF
.ENDS NUCLEON
"""

# The scaling factor bridging AVE Topological d_ij directly to the SPICE Coupling 'K' Coefficient.
# Ensures LTSpice limits K values strictly between 0 and 0.999.
SPICE_K_SCALAR = 0.5 

def generate_spice_netlist(element_name, z, a, nodes, output_dir):
    """
    Parses the (x,y,z) spatial coordinates for the component nucleons
    and generates an identical SPICE LC tank array mapping mutual inductance
    based purely on Euclidean distance (1/d_ij).
    """
    if len(nodes) == 0:
        return
        
    netlist = []
    
    # Header
    netlist.append(f"* Applied Vacuum Engineering (AVE) - SPICE Netlist")
    netlist.append(f"* Element: {element_name} (Z={z}, A={a})")
    netlist.append(f"* Auto-generated topological mutual impedance array")
    netlist.append(f"* Nodes: {len(nodes)}")
    netlist.append("\n")
    
    # Subcircuit definitions
    netlist.append(NUCLEON_SUBCKT)
    netlist.append("\n")
    
    # Instantiate Nucleons as parallel LC Tanks
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* MACROSCOPIC TOPOLOGY (Nucleon Array)")
    netlist.append("* -----------------------------------------------------------------")
    
    # SPICE needs a global ground (0) to simulate accurately.
    # We will tie one side of all tanks to ground, and cascade the other sides.
    # We inject an AC sweep source and a Transient Pulse to ping the impedance of the matrix.
    # Pulse format: PULSE(Vinitial Vpeak Tdelay Trise Tfall Ton Tperiod)
    netlist.append("V_STIM NODE_1 0 AC 1 PULSE(0 100k 10n 1n 1n 50n 100n)\n")
    
    for i in range(len(nodes)):
        # Syntax: X<name> <terminal 1> <terminal 2> <subcircuit_name>
        # We tie all nodes sequentially to create an open lattice we can sweep.
        node_id = i + 1
        netlist.append(f"X_NUC_{node_id} NODE_{node_id} 0 NUCLEON")
    
    netlist.append("\n")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* SPATIAL MUTUAL INDUCTANCE (K-FACTORS)")
    netlist.append("* -----------------------------------------------------------------")
    
    # Calculate Spatial Distance and translate to SPICE Mutual Inductance (K coeff)
    # the format is K1 L1 L2 <value>
    k_index = 1
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Calculate 3D Euclidean Distance
            dist = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
            
            # Translate to physical K factor (Must be strictly < 1.0)
            k_val = SPICE_K_SCALAR / dist
            k_val = min(k_val, 0.999) # LTspice enforcement limit
            
            # The inductors exist INSIDE the subcircuit blocks (X_NUC_n).
            # To couple them in LTSpice we reference the buried inductor: X_NUC_1.L_CORE
            netlist.append(f"K_{k_index} X_NUC_{i+1}.L_CORE X_NUC_{j+1}.L_CORE {k_val:.6f}")
            k_index += 1
            
    netlist.append("\n")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* SIMULATION DIRECTIVES")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* Transient Analysis: 100kV Step Pulse (1ns rise) to test dielectric rupture limits")
    netlist.append(".TRAN 0.1n 200n")
    netlist.append("* Broadband AC sweep: 1 MHz to 1 GHz to ping macro-resonance Q-factors")
    netlist.append(".AC DEC 100 1MEG 1G")
    netlist.append(".OPTIONS METHOD=GEAR")
    netlist.append("\n.END\n")
    
    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{element_name.lower().replace(' ', '_').replace('-', '_')}.cir"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("\n".join(netlist))
        
    print(f"[*] SPICE Netlist Generated: {filepath} (Contains {len(nodes)} NUCLEON subcircuits and {k_index-1} Mutual K mappings)")

def generate_fusion_netlist(fusion_name, nodes_a, name_a, nodes_b, name_b, output_dir):
    """
    Generates a specialized SPICE netlist modeling the collision of two distinct topological networks.
    Demonstrates the energetic cost of forcing phase-lock via a continuous AC Ramp (Ponderomotive force).
    """
    netlist = []
    
    # Header
    netlist.append(f"* Applied Vacuum Engineering (AVE) - DUAL NETWORK FUSION")
    netlist.append(f"* Collision: {name_a} + {name_b} -> {fusion_name}")
    netlist.append(f"* Auto-generated topological mutual impedance transient array")
    netlist.append("\n")
    netlist.append(NUCLEON_SUBCKT)
    netlist.append("\n")
    
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* MACROSCOPIC TOPOLOGY (Dual Array)")
    netlist.append("* -----------------------------------------------------------------")
    
    # We apply an extreme AC ramp to Array A to force it into Array B
    netlist.append("* V_PONDEROMOTIVE provides the kinetic forcing required to overcome 1/d repulsion")
    netlist.append("V_POND NODE_A_1 0 SINE(0 1MEG 1G) AC 1\n")
    
    all_nodes = nodes_a + nodes_b
    
    for i in range(len(nodes_a)):
        node_id = i + 1
        netlist.append(f"X_NUC_A_{node_id} NODE_A_{node_id} 0 NUCLEON")
        
    for i in range(len(nodes_b)):
        node_id = i + 1
        netlist.append(f"X_NUC_B_{node_id} NODE_B_{node_id} 0 NUCLEON")
        
    netlist.append("\n")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* SPATIAL MUTUAL INDUCTANCE (K-FACTORS)")
    netlist.append("* -----------------------------------------------------------------")
    
    k_index = 1
    # Internal coupling Array A
    for i in range(len(nodes_a)):
        for j in range(i + 1, len(nodes_a)):
            dist = np.linalg.norm(np.array(nodes_a[i]) - np.array(nodes_a[j]))
            k_val = min(SPICE_K_SCALAR / dist, 0.999)
            netlist.append(f"K_{k_index} X_NUC_A_{i+1}.L_CORE X_NUC_A_{j+1}.L_CORE {k_val:.6f}")
            k_index += 1
            
    # Internal coupling Array B
    for i in range(len(nodes_b)):
        for j in range(i + 1, len(nodes_b)):
            dist = np.linalg.norm(np.array(nodes_b[i]) - np.array(nodes_b[j]))
            k_val = min(SPICE_K_SCALAR / dist, 0.999)
            netlist.append(f"K_{k_index} X_NUC_B_{i+1}.L_CORE X_NUC_B_{j+1}.L_CORE {k_val:.6f}")
            k_index += 1
            
    # Transient bridging coupling (A to B)
    # We model them at a collision distance of R=1.5d
    collision_offset = np.array([1.5, 0, 0])
    for i in range(len(nodes_a)):
        for j in range(len(nodes_b)):
            pt_a = np.array(nodes_a[i])
            pt_b = np.array(nodes_b[j]) + collision_offset
            dist = np.linalg.norm(pt_a - pt_b)
            k_val = min(SPICE_K_SCALAR / dist, 0.999)
            netlist.append(f"K_{k_index} X_NUC_A_{i+1}.L_CORE X_NUC_B_{j+1}.L_CORE {k_val:.6f}")
            k_index += 1
            
    netlist.append("\n")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* SIMULATION DIRECTIVES")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* Transient Analysis to track the AC ramp phase-locking threshold")
    netlist.append(".TRAN 0.1n 500n")
    netlist.append("\n.END\n")
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{fusion_name.lower().replace(' ', '_').replace('-', '_')}.cir"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("\n".join(netlist))
        
    print(f"[*] SPICE Fusion Netlist Generated: {filepath}")

