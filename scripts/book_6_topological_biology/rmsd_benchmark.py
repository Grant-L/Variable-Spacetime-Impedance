#!/usr/bin/env python3
"""
B13: RMSD Benchmarking — AVE First-Principles vs PDB Experiment
================================================================
AVE PREDICTION CHAIN (zero empirical structural data):
  Axiom 1-2 → soliton_bond_solver → ramachandran_steric → Z_topo → 5-force engine → 3D coords
PDB data: comparison ONLY (never enters prediction)
"""
import sys, os, urllib.request, numpy as np, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mechanics'))
import amino_chain_pipeline as acp

def fetch_pdb_ca(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f'  Fetching {url}...', flush=True)
    resp = urllib.request.urlopen(url, timeout=15)
    lines = resp.read().decode().split('\n')
    ca_coords, sequence = [], []
    aa_map = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
              'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
              'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','AIB':'A','MSE':'M'}
    seen = set()
    for line in lines:
        if line.startswith('ENDMDL'): break
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            rn = int(line[22:26].strip())
            ch = line[21]
            key = (ch, rn)
            if key not in seen:
                seen.add(key)
                ca_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                sequence.append(aa_map.get(line[17:20].strip(), 'A'))
    return np.array(ca_coords), ''.join(sequence)

def kabsch_rmsd(P, Q):
    p_c, q_c = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(p_c.T @ q_c)
    d = np.linalg.det(Vt.T @ U.T)
    sm = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sm @ U.T
    aligned = (p_c @ R.T) + Q.mean(0)
    return np.sqrt(np.mean(np.sum((aligned - Q)**2, 1))), aligned

targets = [
    ('Trp-cage TC5b', '1L2Y'),
    ('Chignolin',     '5AWL'),
    ('Trpzip2',      '1LE1'),
    ('Villin HP35',   '1YRF'),
]

print('=' * 70, flush=True)
print('B13: RMSD BENCHMARKING  AVE vs PDB', flush=True)
print('=' * 70, flush=True)

results = []
for name, pdb_id in targets:
    print(f'\n--- {name} ({pdb_id}) ---', flush=True)
    try:
        pdb_coords, seq = fetch_pdb_ca(pdb_id)
        print(f'  Seq: {seq} (N={len(seq)})', flush=True)

        t0 = time.time()
        z = acp.compute_z_topo(seq)
        ave_coords, _ = acp.fold_chain_3d(seq, n_steps=15000, lr=0.01)
        dt = time.time() - t0

        # Scale AVE to match PDB mean Cα-Cα distance
        pdb_d = np.mean([np.linalg.norm(pdb_coords[i+1]-pdb_coords[i]) for i in range(len(pdb_coords)-1)])
        ave_d = np.mean([np.linalg.norm(ave_coords[i+1]-ave_coords[i]) for i in range(len(ave_coords)-1)])
        ave_s = ave_coords * (pdb_d / ave_d)

        n = min(len(pdb_coords), len(ave_s))
        rmsd, aligned = kabsch_rmsd(ave_s[:n], pdb_coords[:n])

        # Angle correlation
        a_ave, a_pdb = [], []
        for i in range(1, n-1):
            u1, u2 = ave_s[i]-ave_s[i-1], ave_s[i+1]-ave_s[i]
            a_ave.append(np.degrees(np.arccos(np.clip(np.dot(u1,u2)/(np.linalg.norm(u1)*np.linalg.norm(u2)+1e-10),-1,1))))
            v1, v2 = pdb_coords[i]-pdb_coords[i-1], pdb_coords[i+1]-pdb_coords[i]
            a_pdb.append(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-10),-1,1))))

        rg_a = np.sqrt(np.mean(np.sum((ave_s[:n]-ave_s[:n].mean(0))**2, 1)))
        rg_p = np.sqrt(np.mean(np.sum((pdb_coords[:n]-pdb_coords[:n].mean(0))**2, 1)))
        r_corr = np.corrcoef(a_ave, a_pdb)[0, 1] if len(a_ave) > 2 else 0

        print(f'  RMSD: {rmsd:.2f} A | angle_r={r_corr:.3f} | Rg={rg_a:.1f}/{rg_p:.1f} | t={dt:.1f}s', flush=True)
        print(f'  Z_topo: [{", ".join(f"{z:.1f}" for z in z[:8])}...]', flush=True)
        print(f'  Mean angle: AVE={np.mean(a_ave):.0f} PDB={np.mean(a_pdb):.0f}', flush=True)
        results.append((name, pdb_id, n, rmsd, r_corr, rg_a, rg_p, np.mean(a_ave), np.mean(a_pdb)))
    except Exception as e:
        import traceback
        print(f'  ERROR: {e}', flush=True)
        traceback.print_exc()

print(f'\n{"=" * 70}', flush=True)
print(f'SUMMARY', flush=True)
print(f'{"Name":25s} {"PDB":>5s} {"N":>3s} {"RMSD":>6s} {"r(∠)":>6s} {"Rg_A":>5s} {"Rg_P":>5s}', flush=True)
print('-' * 70, flush=True)
for r in results:
    print(f'{r[0]:25s} {r[1]:>5s} {r[2]:3d} {r[3]:6.2f} {r[4]:6.3f} {r[5]:5.1f} {r[6]:5.1f}', flush=True)
if results:
    print(f'\nMean RMSD: {np.mean([r[3] for r in results]):.2f} A', flush=True)
    print(f'Mean angle correlation: r = {np.mean([r[4] for r in results]):.3f}', flush=True)
print('DONE', flush=True)
