"""
AVE MODULE: PONDER-01 LTSpice Emitter Generation
---------------------------------------------
Generates a highly asymmetric SPICE Netlist simulating the POUNDER-01
acoustic rectification testbench using an explicit Helium-4
(Alpha Core Tetrahedron) topology as the ultra-sharp emitter point-source, 
acting against a flat geometric collector plane. 

Models the spatial mutual inductance (K-factors) across the gap to prove 
the asymmetric buildup of Ponderomotive potential in hardware solvers.
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from periodic_table.simulations.simulate_element import get_nucleon_coordinates
from periodic_table.simulations.spice_exporter import NUCLEON_SUBCKT, SPICE_K_SCALAR

def generate_ponder_01_ltspice():
    print("[*] Generating PONDER-01 Helium-4 Emitter LTSpice Netlist...")
    
    # 1. Fetch Emitter Nodes (Helium-4)
    # The He-4 nucleus provides a mathematically perfect, ultra-dense sharp point-charge topology.
    emitter_nodes = get_nucleon_coordinates(2, 4)
    
    # We'll normalize the He-4 center of mass to (0, 0, 0)
    he4_center = np.mean(emitter_nodes, axis=0)
    emitter_nodes = [tuple(np.array(n) - he4_center) for n in emitter_nodes]
    
    # 2. Generate flat Collector Plane nodes
    # A simple 3x3 grid of nodes spaced by d=0.85
    d_spacing = 0.85
    collector_nodes = []
    
    # Offset the collector down the Z-axis by the Ponder-01 gap distance (Ratio 1000:1 mapping scaled down for SPICE limits)
    z_gap = -50.0 * d_spacing 
    
    for x in [-d_spacing*5, 0, d_spacing*5]:
        for y in [-d_spacing*5, 0, d_spacing*5]:
            collector_nodes.append((x, y, z_gap))
            
    # Compile total nodes
    num_emitter = len(emitter_nodes)
    num_collector = len(collector_nodes)
    all_nodes = emitter_nodes + collector_nodes
    
    print(f"[*] Topological Arrays: Emitter (He-4) [{num_emitter} nodes] -> Collector [{num_collector} nodes]")
    print(f"[*] Gap Distance: {abs(z_gap):.2f} metric units")
    
    netlist = []
    
    # Header
    netlist.append("* Applied Vacuum Engineering - PONDER-01 Testbench")
    netlist.append("* Asymmetric AC Geometry: Helium-4 Emitter vs Flat Collector")
    netlist.append("\n")
    netlist.append(NUCLEON_SUBCKT)
    netlist.append("\n")
    
    # 3. Instantiate the topological tanks 
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* MACROSCOPIC TOPOLOGY")
    netlist.append("* -----------------------------------------------------------------")
    
    # The emitter is driven by the 30kV VHF signal (100MHz)
    netlist.append("V_VHF NODE_E 0 AC 1 SINE(0 30k 100MEG)\n")
    
    for i in range(num_emitter):
        # Tie emitter side directly to the VHF High-Voltage source
        netlist.append(f"X_EMIT_{i+1} NODE_E 0 NUCLEON")
        
    for i in range(num_collector):
        # Tie collector side directly to Ground
        netlist.append(f"X_COLL_{i+1} 0 0 NUCLEON")
        
    netlist.append("\n")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* SPATIAL MUTUAL INDUCTANCE (K-FACTORS)")
    netlist.append("* -----------------------------------------------------------------")
    
    k_index = 1
    
    # Coupling internal to Emitter (Helium-4 coherence)
    for i in range(num_emitter):
        for j in range(i + 1, num_emitter):
            dist = np.linalg.norm(np.array(emitter_nodes[i]) - np.array(emitter_nodes[j]))
            if dist > 0:
                k_val = min(SPICE_K_SCALAR / dist, 0.999)
                netlist.append(f"K_{k_index} X_EMIT_{i+1}.L_CORE X_EMIT_{j+1}.L_CORE {k_val:.6f}")
                k_index += 1
                
    # Coupling internal to Collector Plane
    for i in range(num_collector):
        for j in range(i + 1, num_collector):
            dist = np.linalg.norm(np.array(collector_nodes[i]) - np.array(collector_nodes[j]))
            if dist > 0:
                k_val = min(SPICE_K_SCALAR / dist, 0.999)
                netlist.append(f"K_{k_index} X_COLL_{i+1}.L_CORE X_COLL_{j+1}.L_CORE {k_val:.6f}")
                k_index += 1
                
    # MACROSCOPIC COUPLING: Emitter driving Collector across the gap
    netlist.append("* -- Rectification Gradient (Gap Trans-Admittance) --")
    for i in range(num_emitter):
        for j in range(num_collector):
            dist = np.linalg.norm(np.array(emitter_nodes[i]) - np.array(collector_nodes[j]))
            if dist > 0:
                # The gradient scaling acts as the Ponderomotive momentum transfer coefficient
                k_val = min(SPICE_K_SCALAR / dist, 0.999)
                netlist.append(f"K_{k_index} X_EMIT_{i+1}.L_CORE X_COLL_{j+1}.L_CORE {k_val:.6f}")
                k_index += 1
                
    netlist.append("\n")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append("* SIMULATION DIRECTIVES")
    netlist.append("* -----------------------------------------------------------------")
    netlist.append(".TRAN 0.1n 50n")
    netlist.append("* Monitor the inductive momentum coupling across the gap")
    netlist.append(".PROBE V(NODE_E)")
    netlist.append("\n.END\n")
    
    outdir = os.path.join(project_root, "periodic_table", "simulations", "spice_netlists")
    os.makedirs(outdir, exist_ok=True)
    target = os.path.join(outdir, "ponder_01_he4_emitter.cir")
    
    with open(target, 'w') as f:
        f.write("\n".join(netlist))
        
    print(f"[*] LTSpice PONDER-01 Testbench Exported: {target}")

if __name__ == "__main__":
    generate_ponder_01_ltspice()
