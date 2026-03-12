import sys
sys.path.append('.') # Allow executing from repo root
import jax
import jax.numpy as jnp
from string import ascii_uppercase
from s11_fold_engine_v3_jax import fold_s11_jax, fold_cotranslational, _s11_loss_jit, _s11_loss
from s11_fold_engine_v3_jax import compute_z_topo, compute_cys_mask, compute_aromatic_mask, compute_gly_mask, compute_pro_mask, compute_cg_mask
from s11_fold_engine_v3_jax import STUB_LENGTH, STUB_TYPE
from urllib.request import urlopen
import time
import numpy as np

def fetch_pdb_coords(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    coords, seq = [], []
    current_res, current_coords = None, {"N": None, "CA": None, "C": None}
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
             'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    for line in urlopen(url).read().decode('utf-8').split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() in ["N", "CA", "C"] and line[16] in [' ', 'A']:
            res_num = int(line[22:26])
            atom_name = line[12:16].strip()
            if res_num != current_res:
                if current_res is not None and all(current_coords.values()):
                    coords.append([current_coords["N"], current_coords["CA"], current_coords["C"]])
                    seq.append(current_res_name)
                current_res = res_num
                current_res_name = d3to1.get(line[17:20].strip(), 'X')
                current_coords = {"N": None, "CA": None, "C": None}
            current_coords[atom_name] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    if all(current_coords.values()):
        coords.append([current_coords["N"], current_coords["CA"], current_coords["C"]])
        seq.append(current_res_name)
    return np.array(coords), "".join(seq)

def rmsd(c1, c2):
    c1_ca = c1[:, 1, :]
    c2_ca = c2[:, 1, :]
    c1_c = c1_ca - np.mean(c1_ca, axis=0)
    c2_c = c2_ca - np.mean(c2_ca, axis=0)
    C = np.dot(c1_c.T, c2_c)
    U, _, Vt = np.linalg.svd(C)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    c1_rot = np.dot(c1_c, R)
    return np.sqrt(np.mean(np.sum((c1_rot - c2_c)**2, axis=1)))

pdb_id = "1PIN" # WW Domain
print(f"Fetching {pdb_id}...")
native_coords, seq = fetch_pdb_coords(pdb_id)
N = len(seq)
if N > 34:
    seq = seq[:34]
    native_coords = native_coords[:34]
    N = 34
print(f"Sequence ({N}): {seq}")

# Setup biological vectors
z_topo = compute_z_topo(seq)
cys_mask = compute_cys_mask(seq)
arom_mask = compute_aromatic_mask(seq)
gly_mask = compute_gly_mask(seq)
pro_mask = compute_pro_mask(seq)
stub_len = jnp.array([float(STUB_LENGTH.get(aa, 0)) for aa in seq])
stub_type_arr = jnp.array([float(STUB_TYPE.get(aa, 0.0)) for aa in seq])
cg_mask = compute_cg_mask(seq)

print(f"\nRunning Phase 9+10 Cotranslational Optimizer -> 3000 Step Refinement...")
t0 = time.time()
folded_ca, folded_hist, folded_trace, folded_bb = fold_cotranslational(
    seq, steps_per_residue=200, k0=8, lr=0.03, window=30
)
folded_coords_flat = folded_bb.flatten()
final_loss = folded_trace[-1]
t1 = time.time()
print(f"Cotranslational Done in {t1-t0:.1f}s, Loss = {final_loss:.4f}")

folded_coords = np.array(folded_coords_flat).reshape(N, 3, 3)
ct_rmsd = rmsd(folded_coords, native_coords)
print(f"Cotranslational RMSD: {ct_rmsd:.2f} Å")

refined_coords = folded_coords

def analyze_loss_components(coords_flat, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N, chi1=None, chi2=None, cg_mask=None, stub_len=None, stub_type_arr=None, tunnel_window=0):
    from ave.solvers.s11_fold_engine_v3_jax import (
        _get_atom_positions, universal_reflection, P_C, junction_T_operator,
        N_COH_RESIDUES, _SIGMOID_SHARPNESS, R_BURIAL, _r_Ca, d0, LAMBDA_BOND,
        CHI_SCALE, DELTA_CHI, M_N_CA, M_CA_C, M_C_N, N_E_N_CA, N_E_CA_C, N_E_C_N,
        D_N_CA, D_CA_C, D_C_N, LAMBDA_BB_STERIC, R_NN, R_CC, R_CN, R_STERIC_CC
    )
    
    # Run the exact same geometry to extract raw tensors to print!
    pass # Implementation later


    # Reconstruct exact physics variables here using pure JAX
    z_mag = jnp.where(cg_mask, z_topo, 1.0)
    
    atom_N, atom_Ca, atom_C, o_pos, h_pos, cb_pos, cg_pos = _get_atom_positions(
        coords_flat, cys_mask, arom_mask, gly_mask, pro_mask, N, chi1, chi2, cg_mask,
        stub_len=stub_len, stub_type_arr=stub_type_arr
    )
    
    # ... Abridged exact copy ...
    # Wait, the best way to do this is to just call `_s11_loss` with a diagnostic flag!

print("\n--- Diagnostic: Loss Intercept ---")
print("Evaluating Refined Array Tensors...")

components = _s11_loss(
    folded_coords_flat, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N,
    chi1=None, chi2=None, cg_mask=cg_mask, stub_len=stub_len, 
    stub_type_arr=stub_type_arr, return_components=True
)

for key, val in components.items():
    print(f"  {key:<20}: {float(val):.5f}")

print("\nEvaluating Native (1PIN) Tensors...")
# Need to flatten the native coords
native_coords_flat = jnp.array(native_coords).reshape(-1)

components_native = _s11_loss(
    native_coords_flat, z_topo, cys_mask, arom_mask, gly_mask, pro_mask, N,
    chi1=None, chi2=None, cg_mask=cg_mask, stub_len=stub_len, 
    stub_type_arr=stub_type_arr, return_components=True
)

for key, val in components_native.items():
    print(f"  {key:<20}: {float(val):.5f}")

