"""
AVE Vacuum Circuit Analysis & Non-Linear Optics
Simulates the non-linear Axiom 4 varactor limit of the metric.
Demonstrates the Autoresonant Dielectric Rupture (The Vacuum Tesla Coil).
Source: Chapter 12 and Chapter 13
"""
import sys
import math
from pathlib import Path

# Add src directory to path if running as script
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ave.core import constants as k

class VacuumMetricOscillator:
    """
    Simulates a localized node of the M_A vacuum under extreme electromagnetic strain.
    Operates as an LC Tank Circuit with a highly non-linear Axiom 4 Varactor.
    """
    def __init__(self, drive_amplitude=0.08, damping=0.015):
        # Normalized parameters
        self.omega_0 = 1.0          # Unperturbed natural frequency
        self.F0 = drive_amplitude   # Laser drive power (as a fraction of Schwinger limit)
        self.zeta = damping         # Native vacuum impedance (radiation resistance)
        self.dt = 0.01              # Simulation timestep
        
    def simulate(self, mode="fixed", max_time=300.0):
        """
        Integrates the non-linear continuous elastodynamic wave equation.
        mode: "fixed" (Standard Laser) or "autoresonant" (PLL Laser)
        """
        x, v = 0.0, 0.0  # x = Topological Strain (V / V_snap)
        t = 0.0
        
        envelope_history = []
        cycle_max = 0.0
        next_record_time = 2.0 * math.pi
        
        while t < max_time:
            # 1. Drive Signal
            if mode == "fixed":
                # Standard Petawatt Laser (Fixed target frequency)
                drive = self.F0 * math.cos(self.omega_0 * t)
            else:
                # AVE Autoresonant Laser (Phase-Locked Loop)
                # Actively monitors velocity and forces the drive to remain 
                # strictly in phase (90-deg lead) for maximum power transfer.
                if v > 0:
                    drive = self.F0
                elif v < 0:
                    drive = -self.F0
                else:
                    drive = self.F0

            # 2. Axiom 4: The Non-Linear Varactor Restoring Force
            # C(x) = C_0 / sqrt(1 - x^2) --> Force softens as x approaches 1.0
            safe_x = max(-0.9999, min(x, 0.9999))
            restoring_force = (self.omega_0**2) * safe_x * math.sqrt(1.0 - safe_x**2)
            
            # 3. Kinematic Acceleration (a = F_net / m)
            a = -restoring_force - (2.0 * self.zeta * self.omega_0 * v) + drive
            
            # 4. Integrate (Euler-Cromer)
            v += a * self.dt
            x += v * self.dt
            t += self.dt
            
            # Track peak envelope
            if abs(x) > cycle_max:
                cycle_max = abs(x)
                
            # Record envelope exactly once per optical cycle
            if t >= next_record_time:
                envelope_history.append(cycle_max)
                cycle_max = 0.0
                next_record_time += 2.0 * math.pi
                
            # DIELECTRIC RUPTURE (Pair-Production Synthesis)
            if abs(x) >= 0.999:
                envelope_history.append(1.0)
                return envelope_history, True, t
                
        # Append final cycle if incomplete
        envelope_history.append(cycle_max)
        return envelope_history, False, t

def print_ascii_graph(history_fixed, history_pll):
    print("\n   [0%] Metric Strain (V / V_crit)                      [100% RUPTURE]")
    print("   |-----------------------------------------------------------------|")
    
    max_len = max(len(history_fixed), len(history_pll))
    
    for i in range(max_len):
        # Format time step
        time_label = f"t={i:02d}"
        
        # Calculate bar lengths (60 chars max)
        fixed_val = history_fixed[i] if i < len(history_fixed) else history_fixed[-1]
        pll_val = history_pll[i] if i < len(history_pll) else history_pll[-1]
        
        bar_fixed = int(fixed_val * 60)
        bar_pll = int(pll_val * 60)
        
        # Fixed Laser prints as '#'
        # PLL Laser prints as '='
        fixed_str = "#" * bar_fixed
        pll_str = "=" * bar_pll
        
        if i < len(history_pll) and history_pll[i] == 1.0:
            pll_str += " <<< DIELECTRIC SNAP (MATTER SYNTHESIZED)!"
            
        print(f"F ({time_label}) | {fixed_str}")
        print(f"P ({time_label}) | {pll_str}")
        print("   |")

if __name__ == "__main__":
    print("==================================================")
    print("AVE NON-LINEAR OPTICS: THE VACUUM TESLA COIL")
    print("==================================================\n")
    
    # We use a drive power of just 8% of the required static yield force
    laser_power = 0.08
    oscillator = VacuumMetricOscillator(drive_amplitude=laser_power, damping=0.015)
    
    print(f"[+] Initializing Standard Fixed-Frequency Laser (Power: {laser_power*100}%)...")
    hist_fixed, ruptured_fixed, t_fixed = oscillator.simulate(mode="fixed", max_time=250.0)
    
    print(f"[+] Initializing AVE Autoresonant PLL Laser (Power: {laser_power*100}%)...")
    hist_pll, ruptured_pll, t_pll = oscillator.simulate(mode="autoresonant", max_time=250.0)
    
    print("\n[ SIMULATION RESULTS ]")
    print_ascii_graph(hist_fixed, hist_pll)
    
    print("\n==================================================")
    if not ruptured_fixed and ruptured_pll:
        print("VERDICT: THE DETUNING PARADOX IS RESOLVED.")
        print("Because the vacuum is an Axiom 4 Varactor, its frequency drops under stress.")
        print("The Fixed Laser detunes and stalls at ~50% metric strain.")
        print("The Autoresonant PLL tracks the metric and successfully shatters the ")
        print("vacuum, synthesizing mass at just 8% of the brute-force energy requirement.")
    print("==================================================")