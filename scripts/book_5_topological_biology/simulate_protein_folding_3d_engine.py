import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set aesthetic plot parameters for the AVE Engine
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.style.use('dark_background')

# ============================================================================
# AVE 3D Topological Protein Folding Engine (Multiplexed Basis States)
# ============================================================================
# Instead of searching a massive, NP-hard 3D rotational conformational landscape
# (which leads to Levinthal's paradox and local-minima entanglement), the AVE 
# engine evaluates the protein exclusively as a macroscopic AC standing wave.
#
# Because the vacuum is perfectly symmetric, it yields only two fundamental 
# low-energy basis states for a polymer chain to occupy:
# 1. The fully isotropic 3D Helix wrapper.
# 2. The fully anisotropic 2D Beta-Sheet flattener.
#
# The engine initializes the sequence in both geometric basis states simultaneously.
# It then applies the local topological impedance (Z) to calculate the integrated
# macroscopic reactive strain (U_total) across both states, collapsing deterministically
# into the absolute lowest-strain geometry without stepping through random coils.
# ============================================================================

class AVETopologicalFolder:
    def __init__(self, sequence):
        self.sequence = sequence
        self.n_residues = len(sequence)
        
        # Empirical bond lengths (Angstroms)
        self.L_Ca_Ca = 3.8
        
        # Topological Impedance Library (Normalized S11 strain factors pulled from SPICE solver)
        self.impedance_map = {
            'A': 0.8,  # Ala: Low Drag, Perfect Helix Former
            'L': 0.9,  # Leu: Low Drag
            'K': 0.95, # Lys: Flexible
            'E': 0.95, # Glu: Flexible
            'G': 4.5,  # Gly: Highly mismatched, massive strain in Helix
            'V': 3.8,  # Val: Branched, massive strain in Helix
            'P': 5.0,  # Pro: Rigid Kink, breaks Helix symmetry
            'S': 2.5,  # Ser: Polar disruption
            'C': 1.1   # Cys: Moderate
        }

    def _get_z_topo(self, aa):
        if aa not in self.impedance_map:
            print(f"Warning: Amino acid '{aa}' not in topological library, defaulting to Z=1.5")
            return 1.5
        return self.impedance_map[aa]

    def multiplexed_basis_initialization(self):
        """
        Geometrically constructs the 3D backbone Cartesian coordinates for 
        both perfect Basis States: Alpha-Helix and Beta-Sheet.
        """
        self.coords_helix = np.zeros((self.n_residues, 3))
        self.coords_sheet = np.zeros((self.n_residues, 3))
        
        # --- CONSTANTS FOR IDEAL ALPHA-HELIX ---
        # 3.6 residues per turn, 5.4 A pitch
        R_helix = 2.3  # Radius (Angstroms)
        pitch_helix = 5.4
        turns_per_res = 1.0 / 3.6
        dZ_helix = pitch_helix / 3.6
        
        # --- CONSTANTS FOR IDEAL BETA-SHEET ---
        # Extended zig-zag, roughly 3.4 A per residue advance
        dZ_sheet = 3.4
        amp_sheet = 1.0
        
        for i in range(self.n_residues):
            # Construct Alpha-Helix
            theta = 2.0 * np.pi * turns_per_res * i
            self.coords_helix[i, 0] = R_helix * np.cos(theta)
            self.coords_helix[i, 1] = R_helix * np.sin(theta)
            self.coords_helix[i, 2] = dZ_helix * i
            
            # Construct Beta-Sheet
            # Zig-zag along Y, extended along Z
            self.coords_sheet[i, 0] = 0.0
            self.coords_sheet[i, 1] = amp_sheet if i % 2 == 0 else -amp_sheet
            self.coords_sheet[i, 2] = dZ_sheet * i

    def calculate_macroscopic_strain(self):
        """
        Calculates U_total. 
        In the Helix state, bulky/rigid impedance boundaries (Z > 1) cause massive 
        steric overlap and AC back-reflection, magnifying strain exponentially.
        In the Sheet state, strain is relieved geometrically by flattening, but carries
        a high baseline surface-tension cost from exposing the hydrophobic core.
        """
        self.strain_helix = np.zeros(self.n_residues)
        self.strain_sheet = np.zeros(self.n_residues)
        
        # The intrinsic vacuum energy cost of forcing a linear chain into a 3D structural tube
        baseline_helix_tension = 1.5 
        # The intrinsic vacuum energy cost of exposing a flat hydrophobic sheet
        baseline_sheet_tension = 12.0 
        
        for i, aa in enumerate(self.sequence):
            z = self._get_z_topo(aa)
            
            # HELIX STRAIN RULE:
            # Low Z (< 1.0) means it fits perfectly, minimizing strain (U -> 0)
            # High Z (> 1.0) creates violent steric AC mismatch inside tight cylinder
            if z <= 1.0:
                self.strain_helix[i] = baseline_helix_tension * z
            else:
                self.strain_helix[i] = baseline_helix_tension * (z ** 3.0) # Exponential clash
                
            # SHEET STRAIN RULE:
            # Flat geometry relieves sidechain clashes, but pays high constant exposure tax
            self.strain_sheet[i] = baseline_sheet_tension * (z ** 0.5)
            
        self.U_total_helix = np.sum(self.strain_helix)
        self.U_total_sheet = np.sum(self.strain_sheet)

    def collapse_state(self):
        """
        Deterministically selects the absolute lowest-strain geometric manifold.
        """
        if self.U_total_helix < self.U_total_sheet:
            self.final_state = 'Alpha-Helix'
            self.final_coords = self.coords_helix
            self.final_U = self.U_total_helix
        else:
            self.final_state = 'Beta-Sheet'
            self.final_coords = self.coords_sheet
            self.final_U = self.U_total_sheet

    def run(self):
        self.multiplexed_basis_initialization()
        self.calculate_macroscopic_strain()
        self.collapse_state()
        
        print(f"Sequence: {self.sequence}")
        print(f"Basis H (Helix) Strain: {self.U_total_helix:.2f}")
        print(f"Basis S (Sheet) Strain: {self.U_total_sheet:.2f}")
        print(f"-> Deterministic Collapse: {self.final_state}\n")

# ============================================================================
# Execution & Rendering
# ============================================================================

# Define two highly distinct 20-residue test sequences
seq_alpha = "EAAAKAAAAAAKAAAAAAAK" # Known rigid Helix former (Empirical)
seq_beta  = "VGVGVGVGVGVGVGVGVGVG" # Known rigid Sheet former (Empirical)

engine_alpha = AVETopologicalFolder(seq_alpha)
engine_alpha.run()

engine_beta = AVETopologicalFolder(seq_beta)
engine_beta.run()

# Plotting the Deterministic Collapse
fig = plt.figure(figsize=(18, 8))
fig.suptitle('AVE Deterministic Protein Folding (Multiplexed Basis Engine)', fontsize=18, y=0.98)

# -- Panel 1: Sequence A (Alpha) 3D Plot --
ax1 = fig.add_subplot(1, 4, 1, projection='3d')
x, y, z = engine_alpha.final_coords.T
ax1.plot(x, y, z, color='#00ffff', linewidth=3, marker='o', markersize=6)
ax1.set_title(f"Sequence A: {engine_alpha.final_state}", color='#00ffff')
ax1.set_axis_off()
ax1.set_zlim(-5, 55)

# -- Panel 2: Sequence A Energy Bar --
ax2 = fig.add_subplot(1, 4, 2)
bars = ax2.bar(['Basis $\mathcal{H}$\n(Helix)', 'Basis $\mathcal{S}$\n(Sheet)'], 
               [engine_alpha.U_total_helix, engine_alpha.U_total_sheet], 
               color=['#00ffff', '#444444'])
ax2.set_ylabel('Total Thermodynamic Strain ($U_{total}$)')
ax2.set_title(f'Strain Resolution')
ax2.axhline(engine_alpha.U_total_helix, color='white', linestyle='--', alpha=0.5)

# Add value text
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 10, round(yval, 1), ha='center', va='bottom', color='white')

# -- Panel 3: Sequence B (Beta) Energy Bar --
ax3 = fig.add_subplot(1, 4, 3)
bars2 = ax3.bar(['Basis $\mathcal{H}$\n(Helix)', 'Basis $\mathcal{S}$\n(Sheet)'], 
               [engine_beta.U_total_helix, engine_beta.U_total_sheet], 
               color=['#444444', '#ff00ff'])
ax3.set_title(f'Strain Resolution')
ax3.axhline(engine_beta.U_total_sheet, color='white', linestyle='--', alpha=0.5)

for bar in bars2:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 10, round(yval, 1), ha='center', va='bottom', color='white')


# -- Panel 4: Sequence B (Beta) 3D Plot --
ax4 = fig.add_subplot(1, 4, 4, projection='3d')
x2, y2, z2 = engine_beta.final_coords.T
ax4.plot(x2, y2, z2, color='#ff00ff', linewidth=3, marker='o', markersize=6)
ax4.set_title(f"Sequence B: {engine_beta.final_state}", color='#ff00ff')
ax4.set_axis_off()
ax4.set_zlim(-5, 55)

# Ensure same scaling for visual comparison
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax4.set_xlim(-5, 5)
ax4.set_ylim(-5, 5)

ax3.set_ylim(0, max(engine_alpha.U_total_sheet, engine_beta.U_total_helix) * 1.1)
ax2.set_ylim(0, max(engine_alpha.U_total_sheet, engine_beta.U_total_helix) * 1.1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save artifact
output_dir = "scripts/book_5_topological_biology/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "protein_folding_3d_collapse.png")
plt.savefig(output_path, dpi=300, facecolor='black', bbox_inches='tight')
print(f"Rendered diagnostic graph to: {output_path}")

asset_path = "assets/sim_outputs/protein_folding_3d_collapse.png"
plt.savefig(asset_path, dpi=300, facecolor='black', bbox_inches='tight')

plt.close()
print("Simulation complete.")

