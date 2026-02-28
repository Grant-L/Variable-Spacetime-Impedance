import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# MNA (Modified Nodal Analysis) AC Solver for AVE SPICE files
def parse_and_solve_cir(filepath, freqs):
    nodes = {'0': 0}
    components = []
    v_in_node = None
    v_out_node = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            
            comp_type = parts[0][0].upper()
            n1, n2 = parts[1], parts[2]
            
            if n1 not in nodes: nodes[n1] = len(nodes)
            if n2 not in nodes: nodes[n2] = len(nodes)
            
            if comp_type in ['R', 'L', 'C']:
                val_str = parts[3]
                if val_str.endswith('fF'):
                    val_str = val_str[:-2]
                elif val_str.endswith('pH'):
                    val_str = val_str[:-2]
                elif val_str.endswith('Ohm'):
                    val_str = val_str[:-3]
                    
                try:
                    val = float(val_str)
                    components.append((comp_type, nodes[n1], nodes[n2], val))
                except Exception as e:
                    print(f"Failed to parse component: {line} -> {e}")
            elif comp_type == 'V':
                # V_amino in 0 SIN(...)
                v_in_node = nodes[n1]
                
    if 'out' in nodes:
        v_out_node = nodes['out']
    
    num_nodes = len(nodes)
    
    # We want to solve Y * V = J at each frequency
    # V[v_in_node] is forced to 1.0 (assuming 1V AC source)
    # V[0] is forced to 0.0
    
    # For a forced node, we can remove it from the unknown vector and move its Y-matrix columns to the RHS
    unknown_nodes = [i for i in range(num_nodes) if i not in (0, v_in_node)]
    unknown_idx = {n: i for i, n in enumerate(unknown_nodes)}
    N_u = len(unknown_nodes)
    
    H_mag = []
    for f in freqs:
        w = 2 * np.pi * f
        Y = np.zeros((N_u, N_u), dtype=complex)
        J = np.zeros(N_u, dtype=complex)
        
        for ctype, n1, n2, val in components:
            if ctype == 'R':
                y = 1.0 / val
            elif ctype == 'C':
                y = 1j * w * val
            elif ctype == 'L':
                y = 1.0 / (1j * w * val) if w != 0 else 1e9
                
            # Add to full matrix
            # If n1 is unknown, add to diag. If n1 and n2 unknown, subtract from off-diag
            if n1 in unknown_idx:
                i = unknown_idx[n1]
                Y[i, i] += y
                
                # if n2 is known (like 0 or v_in), it acts as a current injection
                if n2 == v_in_node:
                    J[i] += y * 1.0 # V_in is 1.0
                    
            if n2 in unknown_idx:
                j = unknown_idx[n2]
                Y[j, j] += y
                
                if n1 == v_in_node:
                    J[j] += y * 1.0
                    
            if n1 in unknown_idx and n2 in unknown_idx:
                i, j = unknown_idx[n1], unknown_idx[n2]
                Y[i, j] -= y
                Y[j, i] -= y
                
        # Solve
        V_u = np.linalg.solve(Y, J)
        if v_out_node in unknown_idx:
            v_out = V_u[unknown_idx[v_out_node]]
        elif v_out_node == 0:
            v_out = 0.0
        elif v_out_node == v_in_node:
            v_out = 1.0
        else:
            v_out = 0.0
            
        H_mag.append(abs(v_out))
            
    return np.array(H_mag)

from scipy.signal import find_peaks

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    cir_dir = PROJECT_ROOT / "assets" / "sim_outputs" / "spice_models"
    cir_files = glob.glob(str(cir_dir / "*_ave.cir"))
    
    # 300 cm-1 to 4000 cm-1 (convert to Hz: c * cm-1 * 100)
    c = 299792458
    wavenumbers = np.linspace(300, 4000, 2000)
    freqs = wavenumbers * c * 100
    
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 10))
    
    results = {}
    peak_data = []
    
    print(f"Parsing and simulating {len(cir_files)} amino acid geometries...")
    for f in sorted(cir_files):
        name = os.path.basename(f).replace("_ave.cir", "").capitalize()
        H = parse_and_solve_cir(f, freqs)
        H_db = 10 * np.log10(np.clip(H**2, 1e-30, None))
        results[name] = H_db
        
        # Extract the absolute deepest resonance notch
        dominant_idx = np.argmin(H_db)
        dominant_wn = wavenumbers[dominant_idx]
        peak_db = H_db[dominant_idx]
        
        peak_data.append((name, dominant_wn, peak_db))
            
        plt.plot(wavenumbers, H_db, label=name, alpha=0.7, linewidth=1.5)
        
    print("\n=======================================================")
    print("  Dominant Topological Resonance (Primary Absorption Notch)")
    print("=======================================================")
    print(f"{'Amino Acid':<15} | {'Notch (cm⁻¹)':<16} | {'Transmission Depth (dB)'}")
    print("-" * 55)
    # Sort by wavenumber
    for name, wn, db in sorted(peak_data, key=lambda x: x[1] if not np.isnan(x[1]) else 0):
        print(f"{name:<15} | {wn:>10.1f} cm⁻¹     | {db:>8.1f} dB")

        
    plt.title("20 Amino Acids: Fully Predictive Topological Resonance Spectra", fontsize=16)
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    plt.ylabel("|H|² (dB)", fontsize=14)
    plt.ylim(-100, 40)
    plt.xlim(300, 4000)
    
    # Highlight regions
    plt.axvspan(600, 1600, alpha=0.05, color='white')
    plt.text(1100, 35, 'Fingerprint Region', fontsize=12, color='#888', ha='center')
    plt.axvspan(2500, 3800, alpha=0.05, color='cyan')
    plt.text(3150, 35, 'Stretch Region', fontsize=12, color='#668899', ha='center')
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=2)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    out_path = cir_dir.parent / "amino_acid_batch_resonance.png"
    plt.savefig(out_path, dpi=300)
    print(f"\\nBatch sweep complete. Plot saved to: {out_path}")
