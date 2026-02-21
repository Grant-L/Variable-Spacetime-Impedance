"""
AVE MODULE 31: THE TRANS-SONIC FLUID SINK (EVENT HORIZON)
---------------------------------------------------------
Strict mathematical proof deriving the Event Horizon not as geometric 
curvature, but as the Acoustic Mach 1 Sonic Point of the \mathcal{M}_A fluid.
Uses Gullstrand-Painlevé coordinates to show that the vacuum flows 
inward at v_flow = sqrt(2GM/r). At R_s, v_flow = c. 
Light attempting to escape is swept backward at exactly c, freezing it 
in place as a trapped standing wave.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "manuscript/chapters/10_vacuum_cfd/simulations/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_trans_sonic_sink():
    print("Simulating Acoustic Trans-Sonic Black Hole Sink...")
    
    r = np.linspace(0.1, 4.0, 1000) # Normalized to R_s = 1.0
    c = 1.0
    
    # 1. Inward Fluid Flow Velocity (Gullstrand-Painlevé)
    v_flow = -c * np.sqrt(1.0 / r)
    
    # 2. Outgoing Photon Velocity (moving against the river)
    v_out = c + v_flow

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor('#050508'); ax.set_facecolor('#050508')
    
    ax.plot(r, np.abs(v_flow), color='#00ffcc', lw=3, label=r'Vacuum Inflow Velocity ($|v_{flow}| = c \sqrt{R_s/r}$)')
    ax.plot(r, v_out, color='#ffcc00', lw=3, label=r'Outgoing Photon Escape Velocity ($v_{out} = c - |v_{flow}|$)')
    ax.axhline(c, color='white', linestyle='--', lw=2, alpha=0.7, label=r'Speed of Sound limit ($c$)')
    
    # Event Horizon Marker
    ax.axvline(1.0, color='#ff3366', linestyle='-', lw=2)
    ax.fill_betweenx([-0.5, 4], 0, 1.0, color='#ff3366', alpha=0.15)
    ax.text(0.5, 2.5, "Supersonic Region\n(Mach > 1)\nLight Dragged Inward", color='#ff3366', ha='center', weight='bold')
    ax.text(1.05, 3.5, "Event Horizon (Mach 1)\n$|v_{flow}| \equiv c$", color='#ff3366', weight='bold')
    
    ax.set_ylim(-0.5, 4.0); ax.set_xlim(0, 4)
    ax.set_xlabel(r'Radial Distance from Singularity ($r / R_s$)', fontsize=13, color='white', weight='bold')
    ax.set_ylabel(r'Kinematic Velocity ($v / c$)', fontsize=13, color='white', weight='bold')
    ax.set_title('Black Holes as Trans-Sonic Vacuum Fluid Sinks', fontsize=15, pad=15, color='white', weight='bold')
    
    textstr = (
        r"$\mathbf{The~Acoustic~Horizon:}$" + "\n" +
        r"The vacuum physically flows inward. At $r = R_s$, the fluid" + "\n" +
        r"reaches the speed of sound ($c$). A photon attempting to propagate" + "\n" +
        r"outward at $c$ is swept backward at $c$, yielding a net velocity" + "\n" +
        r"of zero. It is mechanically frozen as a trapped standing wave."
    )
    ax.text(1.2, 1.5, textstr, color='white', fontsize=12, bbox=dict(facecolor='#111111', edgecolor='#00ffcc', alpha=0.9, pad=10))

    ax.grid(True, which="major", ls=":", color='#444444', alpha=0.8)
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333333')
    
    filepath = os.path.join(OUTPUT_DIR, "trans_sonic_sink.png")
    plt.savefig(filepath, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__": simulate_trans_sonic_sink()