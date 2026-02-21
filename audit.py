import os, re

dirs = [
    '/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/manuscript',
    '/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/future_work'
]

pattern = re.compile(r'\b(Cosserat|Bingham|fluidic|aerodynamic|viscosity|shear thinning|fluid)\b', re.IGNORECASE)

for d in dirs:
    for root, _, files in os.walk(d):
        if 'legacy_fluidics' in root:
            continue
        for f_name in files:
            if f_name.endswith('.tex'):
                path = os.path.join(root, f_name)
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                matches = pattern.findall(text)
                if matches:
                    print(f"{f_name}: {set(m.lower() for m in matches)}")
