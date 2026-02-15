import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# DCVE TOOL SIZING: Diameter vs Metric Yield Threshold
# =========================================================

# --- CONSTANTS ---
P_INPUT = 60.0             # Transducer Power (Watts)
Q_FACTOR = 1000.0          # Mechanical Q-Factor of tuned Copper
P_CIRC = P_INPUT * Q_FACTOR # Circulating Acoustic Power (60,000 W)

WALL_THICKNESS = 0.0015    # 1.5 mm wall thickness of copper pipe
Z_GRANITE = 1.5e7          # Acoustic Impedance (Rayls)
D_11 = 2.3e-12             # Piezo coeff of Quartz
EPS_0 = 8.854e-12
EPS_R = 4.5
BETA = 1000.0              # Asperity tip geometry multiplier

E_YIELD_THRESHOLD = 2.5e9  # DCVE Metric Yield Limit (V/m)
V_COPPER = 3810.0          # Speed of sound in copper (m/s)
F_TRANS = 40000.0          # Transducer Frequency (Hz)

def calculate_piezo_field(diameter_m):
    """Calculates the max E-Field generated at the cutting annulus based on tube diameter"""
    area_c = np.pi * diameter_m * WALL_THICKNESS
    intensity = P_CIRC / area_c
    pressure = np.sqrt(2 * intensity * Z_GRANITE)
    e_field_base = (D_11 / (EPS_0 * EPS_R)) * pressure
    e_local = e_field_base * BETA
    return e_local

# --- SIMULATION DOMAIN ---
# Sweep pipe diameters from 5 mm to 250 mm
diameters_m = np.linspace(0.005, 0.25, 500)
diameters_mm = diameters_m * 1000

# Calculate fields
e_fields = [calculate_piezo_field(d) for d in diameters_m]
e_fields_gv = np.array(e_fields) / 1e9  # Convert to Gigavolts/m

# Calculate Rayleigh-Poisson Limit (D_max < Lambda / 4)
rayleigh_limit_mm = (V_COPPER / F_TRANS) * 1000 / 4.0

# --- PLOTTING ---
plt.figure(figsize=(10, 6), dpi=120)
plt.style.use('dark_background')

# Plot the E-Field Curve
plt.plot(diameters_mm, e_fields_gv, color='#00ffcc', linewidth=3, 
         label='Generated Piezo-Metric Field (GV/m)')

# Plot the DCVE Yield Threshold
plt.axhline(E_YIELD_THRESHOLD / 1e9, color='#ff3366', linestyle='--', linewidth=2,
            label='DCVE Liquefaction Threshold (2.5 GV/m)')

# Fill the thermodynamic success and failure zones
plt.fill_between(diameters_mm, e_fields_gv, E_YIELD_THRESHOLD / 1e9, 
                 where=(e_fields_gv >= E_YIELD_THRESHOLD / 1e9), 
                 interpolate=True, color='#00ffcc', alpha=0.2)
plt.fill_between(diameters_mm, e_fields_gv, E_YIELD_THRESHOLD / 1e9, 
                 where=(e_fields_gv < E_YIELD_THRESHOLD / 1e9), 
                 interpolate=True, color='#ff3366', alpha=0.2)

# Plot the Acoustic/Kinematic Limit
plt.axvline(rayleigh_limit_mm, color='yellow', linestyle=':', linewidth=2,
            label=f'Rayleigh Hoop-Mode Limit ({rayleigh_limit_mm:.1f} mm)')
plt.axvspan(rayleigh_limit_mm, max(diameters_mm), color='yellow', alpha=0.15)

# Annotations
plt.text(35.0, 6.0, 'THERMODYNAMIC\nSUCCESS ZONE', color='#00ffcc', fontweight='bold')
plt.text(205.0, 1.5, 'STALL ZONE', color='#ff3366', fontweight='bold')
plt.text(50.0, 8.5, 'ACOUSTIC FAILURE ZONE\n(Transverse Mode Coupling)', color='yellow', fontweight='bold')

plt.title('Tube Drill Optimization: Diameter vs Metric Yield (60W @ 40kHz)', fontsize=14, fontweight='bold')
plt.xlabel('Copper Tube Outer Diameter (mm)', fontsize=12)
plt.ylabel('Local E-Field at Quartz Asperities (GV/m)', fontsize=12)

plt.grid(color='#333333', linestyle='-', linewidth=0.5)
plt.legend(loc='upper right', fontsize=10)
plt.ylim(0, 10)
plt.xlim(0, 250)
plt.tight_layout()

# --- TERMINAL VERIFICATION ---
target_3_4_inch = calculate_piezo_field(0.0222) # 3/4 inch OD pipe = 22.2mm
max_d_idx = np.abs(e_fields_gv - (E_YIELD_THRESHOLD / 1e9)).argmin()

print(f"--- 60W DRILL DIAMETER DIAGNOSTICS ---")
print(f"3/4-Inch Pipe (22.2mm): E-Field = {target_3_4_inch/1e9:.2f} GV/m (Cuts easily)")
print(f"Acoustic Rayleigh Limit: {rayleigh_limit_mm:.1f} mm")
print(f"Thermodynamic Yield Limit: {diameters_mm[max_d_idx]:.1f} mm")

plt.show()