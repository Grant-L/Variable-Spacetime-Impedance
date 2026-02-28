import numpy as np
import matplotlib.pyplot as plt
import os
import control
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set aesthetic plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.0
plt.style.use('dark_background')

# ============================================================================
# AVE Topological Transmission Line Model for Deterministic Protein Folding
# ============================================================================
# In the Zero-Parameter Mathematical Graph, secondary structure (Helix vs Sheet)
# is not determined by complex atomic force fields or AI statistical matching.
# It is determined strictly by the macroscopic AC Impedance Match of the amino
# acid sequence functioning as an LC transmission line. 
#
# A sequence that matches the impedance of the vacuum (Z0) exhibits minimal 
# reflection (S11) and smoothly coiling Phase. It cleanly wraps into an 
# Alpha-Helix to minimize strain.
#
# A sequence with severe periodic mismatch generates massive reactive strain 
# (high S11 Reflection). To survive physical rupture, the network violently 
# unwinds and flattens into an extended Beta-Sheet / Random Coil.
# ============================================================================

def create_lc_filter(L, C, R=0.1):
    """
    Creates a continuous-time Transfer Function for an RLC node block.
    """
    # H(s) = 1 / (s^2 * L * C + s * R * C + 1)
    num = [1]
    den = [L * C, R * C, 1]
    return control.TransferFunction(num, den)

def get_amino_acid_component(aa_name):
    """
    Returns the effective topological (L, C) parameters for the R-group
    based on the established AVE node definitions.
    """
    topology_map = {
        # Glycine: No side chain, minimal capacitance, highly flexible (mismatch prone in bulk)
        'Gly': {'L': 1.0e-9, 'C': 0.1e-12, 'R': 0.05, 'desc': 'Flexible / Sheet'},
        
        # Alanine: Methyl group, low topological drag, perfect helix packer
        'Ala': {'L': 2.0e-9, 'C': 1.0e-12, 'R': 0.1, 'desc': 'Helix Former'},
        
        # Leucine: Larger alkyl, strongly supports helix spacing
        'Leu': {'L': 3.5e-9, 'C': 1.8e-12, 'R': 0.15, 'desc': 'Helix Former'},
        
        # Valine: Branched at C-beta, causes severe steric / impedance mismatch
        'Val': {'L': 4.0e-9, 'C': 0.5e-12, 'R': 0.25, 'desc': 'Branched / Sheet'},
        
        # Proline: Cyclic bonded to backbone, acts as a rigid inductive lock (Kink)
        'Pro': {'L': 8.0e-9, 'C': 3.0e-12, 'R': 0.5, 'desc': 'Rigid Kink'},
        
        # Serine: Hydroxyl group, shifts capacitance
        'Ser': {'L': 2.5e-9, 'C': 2.5e-12, 'R': 0.2, 'desc': 'Polar / Coil'},
    }
    
    if aa_name not in topology_map:
        raise ValueError(f"Topological parameters for {aa_name} not yet defined.")
        
    return topology_map[aa_name]

def build_protein_transmission_line(sequence):
    """
    Cascades the individual amino acid transfer functions into a single 
    macro-molecular topological chain.
    """
    sys = control.TransferFunction([1], [1]) # Start with identity
    
    # We model the transmission line as a cascade of series filters
    # In a full rigorous SPICE model this would be cascaded ABCD matrices, 
    # but for Bode plot stability analysis, serial cascading demonstrates the pole 
    # accumulation geometry.
    for aa in sequence:
        comps = get_amino_acid_component(aa)
        node_tf = create_lc_filter(comps['L'], comps['C'], comps['R'])
        sys = control.series(sys, node_tf)
        
    return sys

def calculate_s11_reflection(transfer_function, w):
    """
    Extracts the magnitude response and correlates it to the geometric 
    impedance mismatch (Reactive Strain / S11 mapping).
    """
    mag, phase, omega = control.bode(transfer_function, w, plot=False)
    
    # In a matched line, Magnitude approaches 1 (0 dB). 
    # Deviation from 1 represents the scattered/reflected topological energy.
    # S11 ~ |1 - |H(jw)|| directly correlates to physical chain tension.
    reflected_strain = np.abs(1.0 - mag)
    
    return mag, phase, reflected_strain

# Define the sequences to test
N_LENGTH = 10 # Number of residues in the chain

sequences = {
    'Polyalanine (Alpha-Helix)': ['Ala'] * N_LENGTH,
    'Polyglycine (Beta-Sheet/Coil)': ['Gly'] * N_LENGTH,
    'Polyvaline (Beta-Sheet)': ['Val'] * N_LENGTH,
    'Polyproline (Rigid Helix II)': ['Pro'] * N_LENGTH
}

# Frequency sweep (1 MHz to 100 GHz broadly spanning molecular vibrations)
w = np.logspace(6, 11, 2000)
f = w / (2 * np.pi)

# ============================================================================
# Execution & Plot Generation
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('AVE Deterministic Protein Folding:\nMacroscopic Transmission Line Mismatch (S11 Strain)', fontsize=18, y=0.95)

colors = {
    'Polyalanine (Alpha-Helix)': '#00ffff', # Cyan
    'Polyglycine (Beta-Sheet/Coil)': '#ff00ff', # Magenta
    'Polyvaline (Beta-Sheet)': '#ffaa00',   # Orange
    'Polyproline (Rigid Helix II)': '#00ff00' # Green
}

print("Running SPICE Transmission Line Means Test...\n")

for name, seq in sequences.items():
    print(f"Building chain: {name}")
    sys_chain = build_protein_transmission_line(seq)
    
    mag, phase, strain = calculate_s11_reflection(sys_chain, w)
    
    # Plot 1: Magnitude Response (Bode)
    ax1.semilogx(f, 20 * np.log10(mag), label=name, color=colors[name], linewidth=2.5)
    
    # Plot 2: Total Reactive Strain (S11 substitute)
    # This dictates the geometric folding. Low strain = Helix wrapper. High strain = Unwound Sheet.
    ax2.semilogx(f, strain, label=name, color=colors[name], linewidth=2.5)
    
    # Print the integrated strain proxy for tabular verification
    total_integrated_strain = np.sum(strain)
    print(f" -> Integrated Topological Strain: {total_integrated_strain:.2f} \n")

# Format Bode Magnitude
ax1.set_ylabel('Magnitude (dB)', fontsize=14)
ax1.set_title('Topological Frequency Response (Cascaded R-Group RC Filter)', fontsize=14)
ax1.legend(loc='lower left', fontsize=12)
ax1.set_ylim(-150, 40)

# Format Strain
ax2.set_xlabel('Frequency (Hz)', fontsize=14)
ax2.set_ylabel(r'Reactive Structural Strain ($\sim S_{11}$)', fontsize=14)
ax2.set_title('Geometric Unwinding Tension (High Strain forces Beta-Sheet flattening)', fontsize=14)
ax2.set_ylim(0, 1.1)

# Highlight structural thresholds
ax2.axhline(0.2, color='white', linestyle='--', alpha=0.5, label='Alpha-Helix Max Tolerance')
ax2.fill_between(f, 0.2, 1.1, color='red', alpha=0.1, label='Beta-Sheet Flattening Zone')
ax2.legend(loc='lower right', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.92])

# Save artifact
# --- Standard AVE output directory ---
def _find_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(_find_repo_root(), "assets", "sim_outputs")
os.makedirs(output_dir, exist_ok=True)
# --- End standard output directory ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "protein_spice_folding_strain.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Rendered diagnostic graph to: {output_path}")



plt.close()
print("Simulation complete.")