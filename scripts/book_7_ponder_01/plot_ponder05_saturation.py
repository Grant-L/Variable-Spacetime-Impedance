"""
PONDER-05 DC-Biased Quartz — Saturation Curves.

Shows the dual behavior of ε_eff (collapse) and C_eff (divergence) as
V → V_yield, plus the DC operating point at 30 kV and the thrust profile.

Output: assets/sim_outputs/ponder05_saturation_curves.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from ave.core.constants import V_YIELD

V_yield_kV = V_YIELD / 1e3  # ≈ 43.65 kV

V = np.linspace(0, V_yield_kV * 0.995, 500)  # kV
ratio = V / V_yield_kV

# Axiom 4 saturation
S = np.sqrt(1 - ratio**2)
eps_eff = S             # ε_eff / ε_0
C_eff = 1.0 / S        # C_eff / C_0

# DC operating points
V_dc_points = [10, 20, 30, 40, 43]
thrust_lin = [156, 313, 469, 626, 672]  # μN from the table

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0d1117')

for ax in (ax1, ax2):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#30363d')

# Left: ε_eff and C_eff
ax1.plot(V, eps_eff, color='#4FC3F7', linewidth=2.5,
         label=r'$\varepsilon_{eff}/\varepsilon_0 = S$ (collapse)')
ax1.plot(V, C_eff, color='#EF5350', linewidth=2.5,
         label=r'$C_{eff}/C_0 = 1/S$ (divergence)')

# Mark 30 kV operating point
S_30 = np.sqrt(1 - (30/V_yield_kV)**2)
ax1.scatter([30], [S_30], s=100, c='#4FC3F7', zorder=10, edgecolors='white', linewidths=1.5)
ax1.scatter([30], [1/S_30], s=100, c='#EF5350', zorder=10, edgecolors='white', linewidths=1.5)
ax1.axvline(30, color='#FFD600', alpha=0.3, linewidth=1.5, linestyle='--')
ax1.text(30.5, S_30 + 0.15, f'DC bias\n{S_30:.3f}', fontsize=9, color='#4FC3F7')
ax1.text(30.5, 1/S_30 + 0.15, f'{1/S_30:.3f}', fontsize=9, color='#EF5350')

# V_yield line
ax1.axvline(V_yield_kV, color='#FF9800', alpha=0.5, linewidth=1.5, linestyle='-.')
ax1.text(V_yield_kV - 0.5, 6, f'$V_{{yield}}$\n{V_yield_kV:.1f} kV',
         ha='right', fontsize=10, color='#FF9800')

ax1.set_xlabel('Applied Voltage (kV)', fontsize=13, color='white')
ax1.set_ylabel('Normalized Value', fontsize=13, color='white')
ax1.set_title(r'Axiom 4: $\varepsilon_{eff}$ Collapse vs $C_{eff}$ Divergence',
              fontsize=13, color='white', pad=10)
ax1.set_ylim(0, 7)
ax1.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# Right: Thrust profile
ax2.bar(V_dc_points, thrust_lin, width=2, color='#7C4DFF', alpha=0.7,
        edgecolor='#B388FF', linewidth=1.5)
ax2.plot(V_dc_points, thrust_lin, '-o', color='#B388FF', linewidth=2,
         markersize=8, zorder=5)

# Detection floor
ax2.axhline(1, color='#F44336', alpha=0.5, linewidth=1, linestyle='--')
ax2.text(12, 15, 'Torsion Balance Floor (1 μN)', fontsize=9, color='#F44336', alpha=0.7)

# Mark 30 kV
ax2.scatter([30], [469], s=150, c='#FFD600', zorder=10, edgecolors='white', linewidths=2)
ax2.text(31, 485, f'30 kV: 469 μN\n(120× SNR)', fontsize=10, color='#FFD600', fontweight='bold')

ax2.set_xlabel('DC Bias Voltage (kV)', fontsize=13, color='white')
ax2.set_ylabel('Cross-Term Thrust (μN)', fontsize=13, color='white')
ax2.set_title('PONDER-05: Linear Cross-Term Thrust vs DC Bias',
              fontsize=13, color='white', pad=10)

fig.suptitle('PONDER-05: DC-Biased Quartz Thruster Design Space',
             fontsize=16, color='white', y=1.02)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'sim_outputs', 'ponder05_saturation_curves.png')
plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

import shutil
dst = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'sim_outputs', 'ponder05_saturation_curves.png')
shutil.copy2(output_path, dst)
print(f"Copied to: {dst}")
