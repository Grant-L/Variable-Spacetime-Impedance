import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    os.makedirs('assets/sim_outputs', exist_ok=True)
    
    fig = plt.figure(figsize=(14, 8), facecolor='black')
    fig.patch.set_facecolor('black')
    
    # -------------------------------------------------------------------------
    # Left Pane: W Boson Mode (Torsional Acoustic Twist)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_facecolor('black')
    ax1.axis('off')
    
    # Torsional properties
    z1 = np.linspace(-3, 3, 60)
    theta1 = np.linspace(0, 2*np.pi, 40)
    Z1, Theta1 = np.meshgrid(z1, theta1)
    R = 1.0
    
    # Apply continuous twisting twist_rate * Z
    twist_rate = 0.5 
    Theta_twist = Theta1 + twist_rate * Z1
    X1 = R * np.cos(Theta_twist)
    Y1 = R * np.sin(Theta_twist)
    
    # Plot Torsional Cylinder
    ax1.plot_surface(X1, Y1, Z1, color='cyan', alpha=0.5, edgecolor='none')
    
    # Add wireframe grid (twisted) to clearly visualize the torsional strain
    z_lines = np.linspace(-3, 3, 10)
    for zl in z_lines:
        t_l = np.linspace(0, 2*np.pi, 50)
        x_l = R * np.cos(t_l + twist_rate*zl)
        y_l = R * np.sin(t_l + twist_rate*zl)
        ax1.plot(x_l, y_l, zl*np.ones_like(t_l), color='white', alpha=0.2)
        
    for th in np.linspace(0, 2*np.pi, 12, endpoint=False):
        t_line = th + twist_rate*z1
        x_l = R * np.cos(t_line)
        y_l = R * np.sin(t_line)
        ax1.plot(x_l, y_l, z1, color='lime', linewidth=1.5, alpha=0.8)
        
    ax1.set_box_aspect([1,1,2])
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-3, 3])
    ax1.view_init(elev=15, azim=45)
    
    title_1 = (r"$W^{\pm}$ Boson Mode: Pure Torsion" + "\n" +
               r"Polar Moment of Inertia ($J = \frac{\pi}{2}r^4$)")
    ax1.set_title(title_1, color='white', pad=15, fontsize=15, fontweight='bold')
    ax1.text2D(0.5, 0.05, r"Fundamental $W^{\pm}$ acoustic rotational shear", 
               color='cyan', fontsize=12, ha='center', transform=ax1.transAxes)

    # -------------------------------------------------------------------------
    # Right Pane: Z Boson Mode (Transverse Bending)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor('black')
    ax2.axis('off')
    
    # Bending properties (standing wave)
    z2 = np.linspace(-3, 3, 60)
    theta2 = np.linspace(0, 2*np.pi, 40)
    Z2, Theta2 = np.meshgrid(z2, theta2)
    
    bend_amp = 0.8
    # Centerline bend follows a cosine profile
    bend_profile = bend_amp * np.cos(np.pi * Z2 / 6)
    
    # Calculate geometric normals for the bending tube to keep the cross-section circular
    # slope = derivative of bend_profile
    slope = -bend_amp * (np.pi/6) * np.sin(np.pi * Z2 / 6)
    angle = np.arctan(slope)
    
    # Transverse projection avoiding self-intersection
    X2 = R * np.cos(Theta2) * np.cos(angle) + bend_profile
    Y2 = R * np.sin(Theta2)
    Z2_actual = Z2 - R * np.cos(Theta2) * np.sin(angle)
    
    # Plot Bending Cylinder
    ax2.plot_surface(X2, Y2, Z2_actual, color='magenta', alpha=0.5, edgecolor='none')
    
    # Add wireframe grid
    # Horizontal slices
    for zl in np.linspace(-3, 3, 10):
        t_l = np.linspace(0, 2*np.pi, 50)
        b_p = bend_amp * np.cos(np.pi * zl / 6)
        s_p = -bend_amp * (np.pi/6) * np.sin(np.pi * zl / 6)
        ang = np.arctan(s_p)
        x_l = R * np.cos(t_l) * np.cos(ang) + b_p
        y_l = R * np.sin(t_l)
        z_l = zl - R * np.cos(t_l) * np.sin(ang)
        ax2.plot(x_l, y_l, z_l, color='white', alpha=0.2)
        
    # Longitudinal fibers showing compression/tension
    for th in np.linspace(0, 2*np.pi, 12, endpoint=False):
        b_p = bend_amp * np.cos(np.pi * z2 / 6)
        s_p = -bend_amp * (np.pi/6) * np.sin(np.pi * z2 / 6)
        ang = np.arctan(s_p)
        x_l = R * np.cos(th) * np.cos(ang) + b_p
        y_l = R * np.sin(th) * np.ones_like(z2)
        z_l = z2 - R * np.cos(th) * np.sin(ang)
        if np.cos(th) > 0.5 or np.cos(th) < -0.5:
            # Highlight max tension/compression fibers
            ax2.plot(x_l, y_l, z_l, color='yellow', linewidth=1.5, alpha=0.8)
        else:
            ax2.plot(x_l, y_l, z_l, color='white', linewidth=1.0, alpha=0.3)
            
    ax2.set_box_aspect([1,1,2])
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-3, 3])
    ax2.view_init(elev=15, azim=45)
    
    title_2 = (r"$Z^{0}$ Boson Mode: Transverse Bending" + "\n" +
               r"Area Moment of Inertia ($I = \frac{\pi}{4}r^4$)")
    ax2.set_title(title_2, color='white', pad=15, fontsize=15, fontweight='bold')
    ax2.text2D(0.5, 0.05, r"Fundamental $Z^{0}$ acoustic elastic bending" + "\n" + r"$J = 2I \Rightarrow \sin^2\theta_W \equiv 0.25$", 
               color='magenta', fontsize=12, ha='center', transform=ax2.transAxes)
    
    # Super-title linking them mathematically
    fig.suptitle(r"The Weak Mixing Angle ($\sin^2\theta_W$): The Geometric $J=2I$ Equivalence", 
                 color='white', fontsize=18, fontweight='bold', y=0.98)
                 
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    output_path = 'assets/sim_outputs/electroweak_acoustic_modes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Successfully saved W/Z Boson acoustic visualization to {output_path}")

if __name__ == '__main__':
    main()
