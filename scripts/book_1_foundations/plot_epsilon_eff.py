"""
Plot the three regimes of the non-linear dielectric permittivity ε_eff(Δφ).

Regime 1: Linear (small field) — ε ≈ ε₀
Regime 2: E⁴ correction (intermediate) — Euler-Heisenberg Lagrangian
Regime 3: Saturation (large field at α boundary) — ε → ∞

Output: assets/sim_outputs/vacuum_dielectric_saturation.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from ave.core.constants import ALPHA

# Normalised field variable: Δφ / α  (0 to 0.999)
x = np.linspace(0, 0.999, 2000)

# AVE exact form: ε(Δφ) = ε₀ / sqrt(1 - (Δφ/α)²)
eps_exact = 1.0 / np.sqrt(1 - x**2)

# Taylor expansion to E⁴ order: ε ≈ ε₀ [1 + (1/2)(Δφ/α)²]
eps_taylor = 1.0 + 0.5 * x**2

# Linear regime: ε = ε₀
eps_linear = np.ones_like(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

# Plot all three
ax.plot(x, eps_exact, color='#58a6ff', linewidth=2.5, label=r'AVE exact: $\varepsilon_0 / \sqrt{1-(\Delta\phi/\alpha)^2}$')
ax.plot(x, eps_taylor, color='#f0883e', linewidth=2.0, linestyle='--', label=r'Euler-Heisenberg ($E^4$ Taylor): $\varepsilon_0[1 + \frac{1}{2}(\Delta\phi/\alpha)^2]$')
ax.plot(x, eps_linear, color='#8b949e', linewidth=1.5, linestyle=':', label=r'Linear: $\varepsilon = \varepsilon_0$')

# Shade the three regimes
ax.axvspan(0, 0.3, alpha=0.08, color='#238636', label='Regime I: Linear')
ax.axvspan(0.3, 0.8, alpha=0.08, color='#f0883e', label='Regime II: Euler-Heisenberg')
ax.axvspan(0.8, 1.0, alpha=0.12, color='#da3633', label='Regime III: Saturation')

# Saturation asymptote
ax.axvline(x=1.0, color='#da3633', linewidth=1.5, linestyle='-', alpha=0.5)
ax.annotate(r'$\Delta\phi = \alpha$ (Dielectric Rupture)', xy=(0.98, 5),
            fontsize=10, color='#da3633', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#da3633'))

ax.set_xlabel(r'Normalised Field Strain $\Delta\phi / \alpha$', fontsize=13, color='white')
ax.set_ylabel(r'Effective Permittivity $\varepsilon_{eff} / \varepsilon_0$', fontsize=13, color='white')
ax.set_title(r'Three Regimes of Vacuum Non-Linearity (Axiom 4)', fontsize=15, color='white', pad=15)
ax.set_ylim(0.8, 8)
ax.set_xlim(-0.02, 1.02)
ax.legend(fontsize=9, loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.15, color='#30363d')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs', 'vacuum_dielectric_saturation.png')
plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {output_path}")
