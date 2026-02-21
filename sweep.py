import os, re
from collections import Counter

dirs = ['/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/manuscript', '/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/future_work']

pattern = re.compile(r'\b(Cosserat|Bingham|fluidic|aerodynamic|viscosity|shear thinning|fluid)\b', re.IGNORECASE)

matches = Counter()

for d in dirs:
    for root, _, files in os.walk(d):
        for f_name in files:
            if f_name.endswith('.tex'):
                path = os.path.join(root, f_name)
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        for m in pattern.finditer(line):
                            matches[m.group(0).lower()] += 1

print("Counts of mechanical terms found:")
for term, count in matches.most_common():
    print(f"  {term}: {count}")

