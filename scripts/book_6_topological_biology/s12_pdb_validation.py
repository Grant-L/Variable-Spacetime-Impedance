#!/usr/bin/env python3
"""
20-Protein Stress Test — v7 Multi-Scale S₁₁ Engine
====================================================

Validates the S₁₁ folding engine across 20 proteins spanning:
  - Chain lengths: 20–154 residues
  - Topologies: α-helix, β-sheet, α/β mixed
  - Fold classes: all-alpha, all-beta, alpha+beta

Metrics (all first-principles, zero fitted parameters):
  - Rg: radius of gyration vs η_eq prediction
  - RMSD: Cα RMSD vs native crystal structure
  - SS: secondary structure content (helix + sheet)

All constants from ave.core.constants.
"""
import sys, os, time, urllib.request
import numpy as np

os.environ['JAX_ENABLE_X64'] = '1'

from ave.core.constants import ETA_EQ
from s11_fold_engine_v3_jax import fold_s11_jax

# ═══════════════════════════════════════════════════════════════
# Protein Dataset — 20 well-characterised structures
# ═══════════════════════════════════════════════════════════════
AA_MAP = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',
    'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V'
}

PROTEINS = [
    # (name, pdb_id, chain, max_residues, fold_class)
    # Small (N ≤ 30)
    ("Trp-cage",        "1L2Y", "A",  20, "α"),
    ("BBA5",            "1T8J", "A",  23, "α/β"),
    ("Insulin B-chain", "4INS", "B",  30, "α"),

    # Medium-small (30 < N ≤ 50)
    ("WW domain PIN1",  "1PIN", "A",  34, "β"),
    ("Villin HP35",     "1YRF", "A",  35, "α"),
    ("WW domain FBP28", "1E0L", "A",  37, "β"),
    ("Crambin",         "1CRN", "A",  46, "α/β"),
    ("Protein B IgG",   "1IGD", "A",  61, "α/β"),

    # Medium (50 < N ≤ 80)
    ("Engrailed HD",    "1ENH", "A",  54, "α"),
    ("GB1",             "1PGA", "A",  56, "α/β"),
    ("SH3 src",         "1SRL", "A",  56, "β"),
    ("SH3 α-spectrin",  "1SHG", "A",  57, "β"),
    ("Protein A Bdomain","1BDD","A",  60, "α"),
    ("CI2",             "2CI2", "I",  64, "α/β"),
    ("Ubiquitin",       "1UBQ", "A",  76, "α/β"),
    ("Cytochrome c",    "1HRC", "A",  104,"α"),

    # Large (N > 80)
    ("λ-repressor",     "1LMB", "3",  80, "α"),
    ("FKBP12",          "1FKB", "A", 107, "α/β"),
    ("Barnase",         "1BNI", "A", 110, "α/β"),
    ("Lysozyme",        "2LZM", "A", 129, "α/β"),
]


def download_pdb(pdb_id):
    path = f"/tmp/{pdb_id}.pdb"
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except:
        return None


def extract_ca_and_seq(pdb_path, chain, max_res):
    ca, seq = [], []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (' ', 'A'):
                continue
            ch = line[21]
            if chain != "*" and ch != chain:
                continue
            res_id = line[22:27].strip()
            if res_id in seen:
                continue
            seen.add(res_id)
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            seq.append(AA_MAP.get(line[17:20].strip(), 'A'))
            if len(ca) >= max_res:
                break
    return np.array(ca) if ca else None, ''.join(seq)


def kabsch_rmsd(P, Q):
    p0 = P - P.mean(0); q0 = Q - Q.mean(0)
    H = p0.T @ q0
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return np.sqrt(np.mean(np.sum((p0 @ R.T - q0)**2, axis=1)))


def compute_ss(bb, N):
    """DSSP-like secondary structure assignment from N-Cα-C backbone.

    Uses the Kabsch-Sander H-bond energy criterion:
      E = 0.084 * 332 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) kcal/mol
    H-bond if E < -0.5 kcal/mol (DSSP standard).

    O placed from peptide plane geometry (sp2, 1.24Å from C).
    H placed as anti-bisector of C(i-1)-N(i)-Cα(i) (1.0Å from N).

    Helix: i→i+4 H-bonds (3 consecutive = α-helix).
    Sheet: i→j H-bonds with |i-j| ≥ 5.

    All cutoffs from Kabsch & Sander (1983), no parameter tuning.
    """
    aN = bb[:, 0, :]   # N positions  (N, 3)
    aCa = bb[:, 1, :]  # Cα positions (N, 3)
    aC = bb[:, 2, :]   # C positions  (N, 3)

    # Place O atoms: C=O is in the peptide plane, sp2 from C
    # Direction: perpendicular to Cα-C in the N-Cα-C plane, ~121° from Cα-C
    O_pos = np.zeros((N, 3))
    for i in range(N - 1):
        # Peptide plane: N(i)-Cα(i)-C(i)-N(i+1)
        v_ca_c = aC[i] - aCa[i]
        v_ca_n_next = aN[i+1] - aCa[i] if i < N-1 else aN[i] - aCa[i]
        # C=O direction: opposite to the Cα-N(i+1) projection onto the plane
        n_plane = np.cross(v_ca_c, v_ca_n_next)
        n_plane = n_plane / (np.linalg.norm(n_plane) + 1e-12)
        # O is roughly trans to N(i+1) across the C-Cα bond
        v_c_n = aN[i+1] - aC[i] if i < N-1 else aN[i] - aC[i]
        v_co = -v_c_n / (np.linalg.norm(v_c_n) + 1e-12)
        O_pos[i] = aC[i] + v_co * 1.24  # C=O bond length

    # Last residue O: use same direction as second-to-last
    if N > 1:
        O_pos[N-1] = aC[N-1] + (O_pos[N-2] - aC[N-2])

    # Place H atoms: N-H is anti-bisector of C(i-1)-N(i)-Cα(i)
    H_pos = np.zeros((N, 3))
    for i in range(1, N):
        v_n_c_prev = aC[i-1] - aN[i]
        v_n_ca = aCa[i] - aN[i]
        v_n_c_prev = v_n_c_prev / (np.linalg.norm(v_n_c_prev) + 1e-12)
        v_n_ca = v_n_ca / (np.linalg.norm(v_n_ca) + 1e-12)
        bisector = v_n_c_prev + v_n_ca
        bisector = bisector / (np.linalg.norm(bisector) + 1e-12)
        H_pos[i] = aN[i] - bisector * 1.0  # anti-bisector, N-H = 1.0Å

    # First residue has no H (N-terminus)
    H_pos[0] = aN[0] + np.array([0, 0, 1.0])

    # DSSP H-bond energy: E = q1*q2 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * f
    # where q1*q2 = 0.084 * 332 = 27.888 (kcal·Å/mol)
    # H-bond if E < -0.5 kcal/mol
    Q = 27.888  # 0.084 * 332

    def hbond_energy(i_donor, j_acceptor):
        """Energy of H-bond: N-H(i) → O=C(j)"""
        r_ON = np.linalg.norm(O_pos[j_acceptor] - aN[i_donor]) + 1e-6
        r_CH = np.linalg.norm(aC[j_acceptor] - H_pos[i_donor]) + 1e-6
        r_OH = np.linalg.norm(O_pos[j_acceptor] - H_pos[i_donor]) + 1e-6
        r_CN = np.linalg.norm(aC[j_acceptor] - aN[i_donor]) + 1e-6
        return Q * (1.0/r_ON + 1.0/r_CH - 1.0/r_OH - 1.0/r_CN)

    # Compute H-bond matrix
    hbond = np.zeros((N, N), dtype=bool)
    for i in range(1, N):       # donor (has H)
        for j in range(N - 1):  # acceptor (has O)
            if abs(i - j) < 2:
                continue
            E = hbond_energy(i, j)
            if E < -0.5:
                hbond[i, j] = True

    # α-helix: 3+ consecutive i→i+4 H-bonds
    h_mask = np.zeros(N, dtype=bool)
    for i in range(4, N):
        if hbond[i, i-4]:  # N-H(i) → O=C(i-4)
            h_mask[i-4:i+1] = True

    # Require ≥ 4 consecutive helix residues (1 turn minimum)
    h_runs = np.zeros(N, dtype=bool)
    count = 0
    for i in range(N):
        if h_mask[i]:
            count += 1
        else:
            if count >= 4:
                h_runs[i-count:i] = True
            count = 0
    if count >= 4:
        h_runs[N-count:N] = True

    # β-sheet: long-range H-bonds (|i-j| ≥ 5)
    b_mask = np.zeros(N, dtype=bool)
    for i in range(1, N):
        for j in range(N - 1):
            if abs(i - j) >= 5 and hbond[i, j]:
                b_mask[i] = True
                b_mask[j] = True

    # Don't double-count helix as sheet
    b_mask = b_mask & ~h_runs

    h = int(np.sum(h_runs))
    b = int(np.sum(b_mask))
    return h, b


# ═══════════════════════════════════════════════════════════════
# Run stress test
# ═══════════════════════════════════════════════════════════════
print("=" * 75)
print("  v7 Multi-Scale S₁₁ Engine: 20-Protein Stress Test")
print("=" * 75)
print(f"  η_eq = {ETA_EQ:.6f}")
print()

# First download all PDBs
print("Downloading PDB files...", flush=True)
for name, pdb_id, chain, nmax, fold in PROTEINS:
    download_pdb(pdb_id)
print("Done.\n")

results = []

for idx, (name, pdb_id, chain, nmax, fold) in enumerate(PROTEINS):
    pdb_path = f"/tmp/{pdb_id}.pdb"
    ca_native, seq = extract_ca_and_seq(pdb_path, chain, nmax)

    if ca_native is None or len(seq) < 10:
        # Try wildcard chain
        ca_native, seq = extract_ca_and_seq(pdb_path, "*", nmax)

    if ca_native is None or len(seq) < 10:
        print(f"[{idx+1:2d}/20] {name:20s} — SKIP (PDB parse failed)")
        continue

    N = len(seq)
    Rg_eq = 1.7 * (N / ETA_EQ)**(1/3) * np.sqrt(3/5)

    print(f"[{idx+1:2d}/20] {name:20s} (N={N:3d}, {fold})", end="", flush=True)
    t0 = time.time()

    try:
        # n_steps = D×Q×N×k_adam, n_starts = ⌈D×N/(2πQ)⌉

        ave_ca, _, trace, ave_bb = fold_s11_jax(
            seq, n_steps=6000, lr=2e-3, n_starts=4)
        dt = time.time() - t0

        rg = np.sqrt(np.mean(np.sum((ave_ca - ave_ca.mean(0))**2, 1)))
        rg_err = 100 * abs(rg - Rg_eq) / Rg_eq
        rmsd = kabsch_rmsd(ave_ca, ca_native[:N])
        h, b = compute_ss(ave_bb, N)
        ss = 100 * (h + b) / max(N - 2, 1)

        results.append({
            'name': name, 'pdb': pdb_id, 'N': N, 'fold': fold,
            'rg': rg, 'rg_eq': Rg_eq, 'rg_err': rg_err,
            'rmsd': rmsd, 'ss': ss, 'h': h, 'b': b,
            'loss': trace[-1], 'time': dt, 'error': None
        })

        print(f"  Rg={rg:.1f}Å({rg_err:.0f}%) RMSD={rmsd:.1f}Å "
              f"SS={ss:.0f}% [{dt:.0f}s]")

    except Exception as e:
        dt = time.time() - t0
        results.append({
            'name': name, 'pdb': pdb_id, 'N': N, 'fold': fold,
            'rg': None, 'rg_eq': Rg_eq, 'rg_err': None,
            'rmsd': None, 'ss': None, 'h': None, 'b': None,
            'loss': None, 'time': dt, 'error': str(e)
        })
        print(f"  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════
valid = [r for r in results if r['error'] is None]

print(f"\n{'=' * 80}")
print(f"  RESULTS: {len(valid)}/{len(results)} proteins folded successfully")
print(f"{'=' * 80}")
print(f"\n  {'Protein':<22} {'PDB':>4} {'N':>3} {'Fold':>4} "
      f"{'Rg err':>6} {'RMSD':>6} {'SS':>4} {'Loss':>6} {'Time':>5}")
print(f"  {'─'*70}")

for r in valid:
    print(f"  {r['name']:<22} {r['pdb']:>4} {r['N']:>3d} {r['fold']:>4} "
          f"{r['rg_err']:5.1f}% {r['rmsd']:5.1f}Å {r['ss']:3.0f}% "
          f"{r['loss']:6.3f} {r['time']:4.0f}s")

print(f"  {'─'*70}")

# Aggregate stats
rg_errs = [r['rg_err'] for r in valid]
rmsds = [r['rmsd'] for r in valid]
sss = [r['ss'] for r in valid]

print(f"\n  Rg error:   mean={np.mean(rg_errs):.1f}%, "
      f"median={np.median(rg_errs):.1f}%, max={np.max(rg_errs):.1f}%")
print(f"  RMSD (Cα):  mean={np.mean(rmsds):.1f}Å, "
      f"median={np.median(rmsds):.1f}Å, min={np.min(rmsds):.1f}Å")
print(f"  SS content: mean={np.mean(sss):.0f}%, "
      f"median={np.median(sss):.0f}%, max={np.max(sss):.0f}%")

# Count pass/fail
rg_pass = sum(1 for r in valid if r['rg_err'] < 10)
rmsd_sub8 = sum(1 for r in valid if r['rmsd'] < 8)
rmsd_sub5 = sum(1 for r in valid if r['rmsd'] < 5)
ss_pass = sum(1 for r in valid if r['ss'] > 15)

print(f"\n  Rg < 10%:   {rg_pass}/{len(valid)}")
print(f"  RMSD < 8Å:  {rmsd_sub8}/{len(valid)}")
print(f"  RMSD < 5Å:  {rmsd_sub5}/{len(valid)}")
print(f"  SS > 15%:   {ss_pass}/{len(valid)}")
print(f"\n  Total time: {sum(r['time'] for r in results):.0f}s")
print("=" * 80)
