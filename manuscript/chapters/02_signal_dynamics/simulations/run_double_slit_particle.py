import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "manuscript/chapters/02_signal_dynamics/simulations"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_pilot_wave_statistics():
    print("--- AVE Simulation: The Hydrodynamic Measurement Effect ---")
    
    # 1. SETUP VACUUM TANK
    NX, NY = 200, 100
    dx = 1.0
    
    # Barrier Geometry
    wall_x = 50
    slit_sep = 12
    slit_width = 4
    mid = NY // 2
    
    # Slit positions (Y-indices)
    slit_1 = range(mid - slit_sep - slit_width, mid - slit_sep)
    slit_2 = range(mid + slit_sep, mid + slit_sep + slit_width)
    
    # 2. DEFINE VACUUM WAVE FIELD (The Lattice Memory)
    # We pre-calculate the interference field for speed
    X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
    
    # Wave sources at the slits
    k = 0.5 # Wave number
    r1 = np.sqrt((X - wall_x)**2 + (Y - (mid - slit_sep))**2)
    r2 = np.sqrt((X - wall_x)**2 + (Y - (mid + slit_sep))**2)
    
    # 3. RUN TWO EXPERIMENTS
    modes = ['Coherent (No Detector)', 'Measured (Detector at Slit 2)']
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#111111')
    
    for mode_idx, mode in enumerate(modes):
        print(f"Running Experiment: {mode}...")
        
        # --- DEFINE THE FIELD ---
        if mode_idx == 0:
            # COHERENT: Waves from both slits interfere
            # Amplitude = cos(k*r1) + cos(k*r2)
            # This creates the "Tracks" the particle surfs
            field = np.cos(k*r1) + np.cos(k*r2)
            field[:, :wall_x] = 0 # Block behind wall
        else:
            # MEASURED: Detector at Slit 2 acts as Impedance Load (Resistor)
            # It dampens the wave from Slit 2 significantly
            damping = 0.1 # Detector absorbs 90% of local wave energy
            field = np.cos(k*r1) + (damping * np.cos(k*r2)) 
            field[:, :wall_x] = 0
            
        # --- FIRE PARTICLES ---
        n_particles = 1500
        screen_hits = []
        trajectories = [] # Keep a few for plotting
        
        for p in range(n_particles):
            # Particle starts at Slit 1 (top) or Slit 2 (bottom) randomly
            start_slit = 1 if np.random.rand() > 0.5 else 2
            py = (mid - slit_sep) if start_slit == 1 else (mid + slit_sep)
            px = wall_x + 2
            
            vx, vy = 1.0, 0.0 # Initial momentum vector
            
            path_x, path_y = [px], [py]
            
            # Time Integration (The Surfing)
            for t in range(180): # Time to reach screen
                # 1. Calculate Gradient (The Pilot Force)
                # Particle accelerates downhill on the pressure gradient
                # F = -grad(P^2) (Ponderomotive Force approximation)
                
                ix, iy = int(px), int(py)
                if ix < NX-1 and iy < NY-1 and iy > 1:
                    # Simple central difference
                    dp_dy = (field[iy+1, ix]**2 - field[iy-1, ix]**2) / 2.0
                    
                    # Force Coupling
                    # In Coherent mode, dp_dy oscillates (fringes) -> Particle wiggles
                    # In Measured mode, fringes are weak -> Particle flies straight
                    force_y = -dp_dy * 0.15 
                    
                    vy += force_y
                
                px += vx
                py += vy
                
                if p < 20: # Save first 20 paths for visual
                    path_x.append(px)
                    path_y.append(py)
                
                if px >= NX-5: # Hit Screen
                    screen_hits.append(py)
                    break
        
        # --- VISUALIZATION ---
        # Left Panel: Trajectories on Field
        ax_field = axes[mode_idx, 0]
        # Plot Vacuum Pressure Field
        ax_field.imshow(field**2, extent=[0, NX, 0, NY], origin='lower', cmap='inferno', vmin=0, vmax=4)
        
        # Plot Barrier
        ax_field.axvline(wall_x, color='gray', linewidth=3)
        
        # Plot Particle Paths
        for px_trace, py_trace in trajectories:
            ax_field.plot(px_trace, py_trace, color='cyan', linewidth=0.8, alpha=0.6)
            
        ax_field.set_title(f"Vacuum Geometry: {mode}", color='white', fontsize=12)
        ax_field.axis('off')
        
        if mode_idx == 1:
            # Draw "Detector"
            ax_field.add_patch(plt.Circle((wall_x, mid+slit_sep), 3, color='red', label='Detector Load'))
            ax_field.text(wall_x+5, mid+slit_sep, "Impedance Load\n(Damps Wave)", color='red', fontsize=9)

        # Right Panel: Screen Intensity (Histogram)
        ax_hist = axes[mode_idx, 1]
        
        # Bin the hits
        counts, bins = np.histogram(screen_hits, bins=50, range=(0, NY))
        centers = (bins[:-1] + bins[1:]) / 2
        
        color = 'cyan' if mode_idx == 0 else 'orange'
        ax_hist.plot(counts, centers, color=color, linewidth=2)
        ax_hist.fill_betweenx(centers, 0, counts, color=color, alpha=0.3)
        
        ax_hist.set_ylim(0, NY)
        ax_hist.set_xlim(0, max(counts)*1.2)
        ax_hist.set_facecolor('#111111')
        ax_hist.tick_params(colors='white')
        
        if mode_idx == 0:
            ax_hist.set_title("Result: Interference Fringes (Wave-like)", color='white')
        else:
            ax_hist.set_title("Result: Two Lumps (Particle-like)", color='white')
            
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "double_slit_particle.png")
    plt.savefig(output_path, dpi=150, facecolor='#111111')
    print(f"Simulation saved to {output_path}")

if __name__ == "__main__":
    simulate_pilot_wave_statistics()