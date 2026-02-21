import os, re
path = '/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/manuscript/chapters/00_derivations/11_continuum_fluidics.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

out, count = re.subn(r'\bCosserat\b', 'Chiral LC', text)
print(f"Matches for Cosserat: {count}")
out, count = re.subn(r'\bCosserat\b', 'Chiral LC', text, flags=re.IGNORECASE)
print(f"Matches for Cosserat (ignorecase): {count}")
