import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/14_active_metric_engineering/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_casimir_modulation():
    n_vac = np.linspace(1.0, 1.5, 500) 
    F_c = 1.0 / n_vac
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(n_vac, F_c, color='#00ffcc', lw=4, label=r'Modulated Casimir Force ($F_c \propto 1/n$)')
    ax.fill_between(n_vac, 0, F_c, color='#00ffcc', alpha=0.15)
    
    ax.set_title('Tabletop Signature: Metric Modulation of Casimir Force', color='white', fontsize=14, weight='bold')
    ax.set_xlabel('Local Vacuum Refractive Index ($n$)', color='white', weight='bold')
    ax.set_ylabel('Normalized Casimir Force ($F_c / F_{c0}$)', color='white', weight='bold')
    ax.set_ylim(0.5, 1.1); ax.set_xlim(1.0, 1.5)
    
    textstr = (
        r"$\mathbf{The~LC~Zero{-}Point~Test:}$" + "\n" +
        r"If the Casimir effect is the physical exclusion of discrete LC resonance modes," + "\n" +
        r"increasing the vacuum density ($n > 1$) lowers the local speed of light." + "\n" +
        r"This dynamically red-shifts the zero-point energy, generating a" + "\n" +
        r"measurable, falsifiable suppression of the macroscopic Casimir attraction."
    )
    ax.text(1.02, 0.6, textstr, color='white', fontsize=11, bbox=dict(facecolor='#111111', edgecolor='#ff3366', alpha=0.9, pad=10))

    ax.grid(True, ls=':', color='#333333'); ax.tick_params(colors='white')
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "casimir_modulation.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__": simulate_casimir_modulation()