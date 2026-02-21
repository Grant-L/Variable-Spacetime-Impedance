import os, re

replacements = [
    (r'\bgravitomagnetic drag\b', 'mutual inductive drag'),
    (r'\bCosserat [Ss]olid\b', 'Chiral LC Network'),
    (r'\bCosserat\b', 'Chiral LC'),
    (r'\bBingham [Pp]lastic\b', 'non-linear dielectric'),
    (r'\bBingham yield\b', 'Dielectric Saturation'),
    (r'\bBingham\b', 'Dielectric Saturation'),
    (r'\bfluidic drag\b', 'inductive drag'),
    (r'\bfluid drag\b', 'inductive drag'),
    (r'\bFluidic\b', 'Inductive'),
    (r'\bfluidic\b', 'inductive'),
    (r'\bsuperfluid slip\b', 'zero-impedance phase slip'),
    (r'\bSuperfluid\b', 'Zero-Impedance Phase'),
    (r'\bsuperfluid\b', 'zero-impedance phase'),
    (r'\baerodynamic wave drag\b', 'dielectric wave drag'),
    (r'\bAerodynamics\b', 'Electrodynamics'),
    (r'\baerodynamics\b', 'electrodynamics'),
    (r'\bAerodynamic\b', 'Electrodynamic'),
    (r'\baerodynamic\b', 'electrodynamic'),
    (r'\bkinematic viscosity\b', 'mutual inductance'),
    (r'\bmacroscopic viscosity\b', 'macroscopic reluctance'),
    (r'\bViscosity\b', 'Mutual Inductance'),
    (r'\bviscosity\b', 'mutual inductance'),
    (r'\bshear thinning\b', 'dielectric saturation'),
    (r'\bshear-thinning\b', 'saturating'),
    (r'\bViscous\b', 'Highly-Reluctant'),
    (r'\bviscous\b', 'highly-reluctant'),
    (r'\bFluid\b', 'Network'),
    (r'\bfluid\b', 'network')
]

dirs = [
    '/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/manuscript',
    '/Users/grantlindblom/Variable-Spacetime-Impedance/Variable-Spacetime-Impedance/future_work'
]
total_replacements = 0

for d in dirs:
    for root, _, files in os.walk(d):
        if 'legacy_fluidics' in root:
            continue
        for f_name in files:
            if f_name.endswith('.tex'):
                path = os.path.join(root, f_name)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content
                
                for pattern, replacement in replacements:
                    content, count = re.subn(pattern, replacement, content)
                    total_replacements += count
                    
                if content != original_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)

print(f"Total replacements made: {total_replacements}")
