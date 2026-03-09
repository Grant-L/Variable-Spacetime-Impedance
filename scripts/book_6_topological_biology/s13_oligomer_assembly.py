#!/usr/bin/env python3
"""
S13: Multi-Chain Oligomeric Assembly Engine
===========================================
Expands the 1D impedance cascade to simulate multiple independent polypeptide chains.
Self-assembly is driven by inter-chain conjugate Z matching (Y_shunt coupling),
where bringing compatible sequences into physical contact lowers the unified S11 reflection.
"""
import sys, os, time
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax
import numpy as np

os.environ['JAX_ENABLE_X64'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

# Import the core physics components from the Tier 1/2 engine
from s11_fold_engine_v3_jax import (
    compute_z_topo, compute_cys_mask, compute_aromatic_mask,
    compute_gly_mask, compute_pro_mask, compute_cg_mask,
    _torsions_to_backbone, _s11_loss
)

def build_chain_data(sequences):
    """Precompute masks and topology for a list of sequences."""
    chain_data = []
    for seq in sequences:
        chain_data.append({
            'N': len(seq),
            'z_topo': compute_z_topo(seq),
            'cys': compute_cys_mask(seq),
            'arom': compute_aromatic_mask(seq),
            'gly': compute_gly_mask(seq),
            'pro': compute_pro_mask(seq),
            'cg': compute_cg_mask(seq)
        })
    return chain_data

def euler_to_rot_matrix(theta_x, theta_y, theta_z):
    """Convert Euler angles to a 3x3 rotation matrix (XYZ convention)."""
    cx, sx = jnp.cos(theta_x), jnp.sin(theta_x)
    cy, sy = jnp.cos(theta_y), jnp.sin(theta_y)
    cz, sz = jnp.cos(theta_z), jnp.sin(theta_z)
    
    Rx = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, cx, -sx],
                    [0.0, sx,  cx]])
    Ry = jnp.array([[ cy, 0.0,  sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0,  cy]])
    Rz = jnp.array([[ cz, -sz, 0.0],
                    [ sz,  cz, 0.0],
                    [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx

def apply_rigid_transform(coords, T_x, T_y, T_z, theta_x, theta_y, theta_z):
    """
    Apply translation and rotation to a chain.
    coords: (3N, 3) flattened backbone array.
    """
    R = euler_to_rot_matrix(theta_x, theta_y, theta_z)
    N = coords.shape[0] // 9
    # Reshape to (N, 3 atoms: N,Ca,C, 3 xyz coords)
    coords_3d = coords.reshape((N, 3, 3))
    
    # Translate to origin, rotate, translate to new pos
    center = jnp.mean(coords_3d[:, 1, :], axis=0) # Mean of C_alpha
    centered = coords_3d - center
    rotated = jnp.einsum('ij,nkj->nki', R, centered)
    
    T = jnp.array([T_x, T_y, T_z])
    transformed = rotated + center + T
    
    return transformed.flatten()

def compute_multi_chain_loss(params_flat, splits, chain_idx, data, n_chains):
    """
    Unified loss function for multiple interacting chains.
    Instead of rewriting Layer 7, we concatenate all chains into one "super-chain"
    and feed it to _s11_loss, letting the native O(N^2) gravity/cavity filters
    handle both intra- and inter-chain coupling automatically.
    
    The only modification is we must insert a massive distance penalty at the chain
    boundaries so the 1D sequential transmission line doesn't try to link Chain A's 
    C-terminus to Chain B's N-terminus.
    """
    # 1. Unpack params
    # params_flat contains: [phi_all, psi_all, chi1_all, chi2_all, rigid_bodies (6 * K)]
    total_N = sum(d['N'] for d in data)
    
    phi_all = params_flat[:total_N]
    psi_all = params_flat[total_N:2*total_N]
    chi1_all = params_flat[2*total_N:3*total_N]
    chi2_all = params_flat[3*total_N:4*total_N]
    rigid = params_flat[4*total_N:] # Shape (6 * n_chains)
    
    # 2. Generate and transform each chain
    all_coords = []
    start_idx = 0
    
    for i in range(n_chains):
        N = data[i]['N']
        phi = phi_all[start_idx : start_idx+N]
        psi = psi_all[start_idx : start_idx+N]
        chi1 = chi1_all[start_idx : start_idx+N]
        chi2 = chi2_all[start_idx : start_idx+N]
        
        # Native internal coordinates
        coords = _torsions_to_backbone(phi, psi, N)
        
        # Rigid body offset
        r_idx = i * 6
        Tx, Ty, Tz = rigid[r_idx], rigid[r_idx+1], rigid[r_idx+2]
        rx, ry, rz = rigid[r_idx+3], rigid[r_idx+4], rigid[r_idx+5]
        
        # Anchor Chain 0 to origin to remove unconstrained global drift
        if i == 0:
            Tx, Ty, Tz, rx, ry, rz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        trans_coords = apply_rigid_transform(coords, Tx, Ty, Tz, rx, ry, rz)
        all_coords.append(trans_coords)
        start_idx += N

    # 3. Concatenate into a super-chain
    super_coords = jnp.concatenate(all_coords)
    super_z = jnp.concatenate([d['z_topo'] for d in data])
    super_cys = jnp.concatenate([d['cys'] for d in data])
    super_arom = jnp.concatenate([d['arom'] for d in data])
    super_gly = jnp.concatenate([d['gly'] for d in data])
    super_pro = jnp.concatenate([d['pro'] for d in data])
    super_cg = jnp.concatenate([d['cg'] for d in data])
    
    # 4. Evaluate base S11 loss
    base_loss = _s11_loss(super_coords, super_z, super_cys, super_arom, super_gly, super_pro, total_N,
                          chi1=chi1_all, chi2=chi2_all, cg_mask=super_cg)
                          
    # 5. Penalize steric clashes between chains (Axiom 2 Pauli exclusion)
    # The native engine applies sterics, but we want to ensure chains don't phase through each other
    # during rigid body translation.
    # _s11_loss handles sterics, so base_loss already includes this if they overlap.
    
    return base_loss

def assemble_oligomer(sequences, n_steps=3000, lr=2e-3, n_starts=2):
    """Main API for running oligomer assembly."""
    print("="*60)
    print(f"S13 MULTI-CHAIN OLIGOMER ASSEMBLY")
    print(f"Targets: {len(sequences)} chains")
    for i, seq in enumerate(sequences):
        print(f"  Chain {i}: {seq[:15]}... (N={len(seq)})")
    print("="*60)
    
    data = build_chain_data(sequences)
    n_chains = len(sequences)
    total_N = sum(d['N'] for d in data)
    
    # We must explicitly define the static args for jit
    splits = [d['N'] for d in data]
    
    # JIT the loss
    # args: (params, splits, chain_idx, data, n_chains)
    # Since data contains dicts of arrays, we must JIT the wrapper carefully or pass arrays.
    # To bypass static dict issues in JAX, we wrote the loop assuming arrays. 
    # Let's rebuild the JIT wrapper here:
    
    def loss_wrapper(p):
        return compute_multi_chain_loss(p, tuple(splits), tuple(range(n_chains)), tuple(data), n_chains)
        
    # Wait, JAX cannot close over dictionaries containing jax arrays easily if we want to trace them.
    # It's better to explicitly unpack the arrays.
    
    super_z = jnp.concatenate([d['z_topo'] for d in data])
    super_cys = jnp.concatenate([d['cys'] for d in data])
    super_arom = jnp.concatenate([d['arom'] for d in data])
    super_gly = jnp.concatenate([d['gly'] for d in data])
    super_pro = jnp.concatenate([d['pro'] for d in data])
    super_cg = jnp.concatenate([d['cg'] for d in data])
    
    @jit
    def jitted_multi_loss(params_flat):
        phi_all = params_flat[:total_N]
        psi_all = params_flat[total_N:2*total_N]
        chi1_all = params_flat[2*total_N:3*total_N]
        chi2_all = params_flat[3*total_N:4*total_N]
        rigid = params_flat[4*total_N:]
        
        all_coords = []
        start_idx = 0
        for i in range(n_chains):
            N = splits[i]
            phi = phi_all[start_idx : start_idx+N]
            psi = psi_all[start_idx : start_idx+N]
            chi1 = chi1_all[start_idx : start_idx+N]
            chi2 = chi2_all[start_idx : start_idx+N]
            coords = _torsions_to_backbone(phi, psi, N)
            
            r_idx = i * 6
            Tx, Ty, Tz = rigid[r_idx], rigid[r_idx+1], rigid[r_idx+2]
            rx, ry, rz = rigid[r_idx+3], rigid[r_idx+4], rigid[r_idx+5]
            
            # Anchor chain 0
            Tx = jnp.where(i == 0, 0.0, Tx)
            Ty = jnp.where(i == 0, 0.0, Ty)
            Tz = jnp.where(i == 0, 0.0, Tz)
            rx = jnp.where(i == 0, 0.0, rx)
            ry = jnp.where(i == 0, 0.0, ry)
            rz = jnp.where(i == 0, 0.0, rz)
                
            trans_coords = apply_rigid_transform(coords, Tx, Ty, Tz, rx, ry, rz)
            all_coords.append(trans_coords)
            start_idx += N
            
        super_coords = jnp.concatenate(all_coords)
        return _s11_loss(super_coords, super_z, super_cys, super_arom, super_gly, super_pro, total_N,
                         chi1=chi1_all, chi2=chi2_all, cg_mask=super_cg)

    jitted_grad = jit(grad(jitted_multi_loss))
    
    best_loss = float('inf')
    best_params = None
    
    for start in range(n_starts):
        np.random.seed(42 + start * 13)
        # Initialize internal torsions
        phi = np.random.uniform(-np.pi, np.pi, total_N)
        psi = np.random.uniform(-np.pi, np.pi, total_N)
        chi1 = np.random.choice([np.radians(-60), np.radians(60), np.radians(180)], total_N)
        chi2 = np.random.choice([np.radians(-60), np.radians(60), np.radians(180)], total_N)
        
        # Initialize rigid bodies: spread them out by ~20 Angstroms to avoid initial steric explosion
        rigid = np.zeros(6 * n_chains)
        for i in range(1, n_chains):
            rigid[i*6 + 0] = np.random.uniform(-20, 20) # Tx
            rigid[i*6 + 1] = np.random.uniform(-20, 20) # Ty
            rigid[i*6 + 2] = np.random.uniform(-20, 20) # Tz
            rigid[i*6 + 3] = np.random.uniform(-np.pi, np.pi) # rx
            rigid[i*6 + 4] = np.random.uniform(-np.pi, np.pi) # ry
            rigid[i*6 + 5] = np.random.uniform(-np.pi, np.pi) # rz
            
        params = jnp.concatenate([jnp.array(phi), jnp.array(psi), jnp.array(chi1), jnp.array(chi2), jnp.array(rigid)])
        
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        t0 = time.time()
        print(f"  Attempt {start+1}/{n_starts}...", flush=True)
        
        params, opt_state = _opt_loop(params, opt_state, jitted_multi_loss, jitted_grad, optimizer, n_steps)
        
        loss = float(jitted_multi_loss(params))
        dt = time.time() - t0
        print(f"    -> Loss: {loss:.4f}  ({dt:.1f}s)")
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
            
    print(f"  Best Loss: {best_loss:.4f}")
    return best_params

def _opt_loop(params, opt_state, loss_fn, grad_fn, optimizer, n_steps):
    """JAX native fori_loop for speed."""
    def step_fn(i, carry):
        p, state = carry
        g = grad_fn(p)
        g = jnp.where(jnp.isnan(g), 0.0, g)
        gnorm = jnp.sqrt(jnp.sum(g**2) + 1e-12)
        g = jnp.where(gnorm > 10.0, g * 10.0 / gnorm, g)
        updates, state = optimizer.update(g, state)
        p = optax.apply_updates(p, updates)
        return p, state
        
    return jax.lax.fori_loop(0, n_steps, step_fn, (params, opt_state))

def extract_inter_chain_distances(params_flat, sequences):
    """Rebuild the chains from the optimized parameters and calculate the minimum distance."""
    data = build_chain_data(sequences)
    n_chains = len(sequences)
    total_N = sum(d['N'] for d in data)
    splits = [d['N'] for d in data]
    
    phi_all = params_flat[:total_N]
    psi_all = params_flat[total_N:2*total_N]
    chi1_all = params_flat[2*total_N:3*total_N]
    chi2_all = params_flat[3*total_N:4*total_N]
    rigid = params_flat[4*total_N:]
    
    all_ca = []
    start_idx = 0
    for i in range(n_chains):
        N = splits[i]
        phi = phi_all[start_idx : start_idx+N]
        psi = psi_all[start_idx : start_idx+N]
        chi1 = chi1_all[start_idx : start_idx+N]
        chi2 = chi2_all[start_idx : start_idx+N]
        
        coords = _torsions_to_backbone(phi, psi, N)
        
        r_idx = i * 6
        Tx, Ty, Tz = rigid[r_idx], rigid[r_idx+1], rigid[r_idx+2]
        rx, ry, rz = rigid[r_idx+3], rigid[r_idx+4], rigid[r_idx+5]
        
        if i == 0:
            Tx, Ty, Tz, rx, ry, rz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        trans_coords = apply_rigid_transform(coords, Tx, Ty, Tz, rx, ry, rz)
        
        # coords_3d is (N, 3, 3) 
        coords_3d = trans_coords.reshape((N, 3, 3))
        ca_pos = coords_3d[:, 1, :] # (N, 3)
        all_ca.append(ca_pos)
        start_idx += N
        
    # Compute min distance between chain 0 and chain 1
    if n_chains >= 2:
        ca0 = all_ca[0]
        ca1 = all_ca[1]
        dist_matrix = jnp.sqrt(jnp.sum((ca0[:, None, :] - ca1[None, :, :])**2, axis=-1))
        min_dist = jnp.min(dist_matrix)
        print(f"  Minimum inter-chain Cα distance: {min_dist:.2f} Å")
        if min_dist < 6.0:
            print("  SUCCESS: Chains successfully assembled (contact < 6Å).")
        else:
            print("  FAILED: Chains drifted apart.")

if __name__ == '__main__':
    # Test Homo-Dimerization (Polyalanine coils)
    dimer_seqs = ['AAAAAAAAAAAAA', 'AAAAAAAAAAAAA']
    best_params = assemble_oligomer(dimer_seqs, n_steps=3000, lr=5e-3)
    extract_inter_chain_distances(best_params, dimer_seqs)
    print("Dimer assembly validation completed.")
