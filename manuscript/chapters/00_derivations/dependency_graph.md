graph TD
    %% ----------------------------------------------------
    %% AXIOMS & EMPIRICAL CALIBRATION
    %% ----------------------------------------------------
    substyle AXIOMS fill:#f9f2f4,stroke:#333,stroke-width:2px;
    A1["Axiom 1: Topo-Kinematic Isomorphism<br>[Charge Q ≡ Length L]"]:::axiom
    CAL["Empirical Calibration<br>(Electron Mass Saturation)"]:::axiom
    COS["Axiom 2: Cosserat Elasticity<br>(Trace-Free TT Gauge)"]:::axiom

    %% ----------------------------------------------------
    %% KINEMATICS & GEOMETRY
    %% ----------------------------------------------------
    A1 --> XT["Topological Conversion<br>ξ_topo = e / l_node"]
    CAL --> LN["Hardware Pitch<br>l_node = ħ / m_e c"]
    LN --> AL["Porosity Ratio (Duty Cycle)<br>α = r_core / l_node ≈ 1/137"]
    AL --> KV["Volumetric Packing Fraction<br>κ_V = 8π α ≈ 0.183"]
    
    %% ----------------------------------------------------
    %% IMPEDANCE & INERTIA
    %% ----------------------------------------------------
    XT --> OHM["Electrical Impedance<br>1 Ω ≡ ξ_topo² (kg/s)"]
    XT --> IN["Inertia (B-EMF)<br>p = ξ_topo⁻¹ Φ_Z"]

    %% ----------------------------------------------------
    %% ELASTODYNAMICS & WEAK FORCE
    %% ----------------------------------------------------
    COS --> K2G["Trace-Reversed Moduli<br>K_vac = 2 G_vac"]
    K2G --> NU["Vacuum Poisson's Ratio<br>ν_vac = 2/7"]
    NU --> WZ["Weak Mixing Angle<br>m_W / m_Z ≈ 0.8819"]

    %% ----------------------------------------------------
    %% GRAVITY & COSMOLOGY
    %% ----------------------------------------------------
    LN --> TEM["1D QED Tension Limit<br>T_EM = m_e c² / l_node"]
    KV --> TEM
    AL --> XI["Hierarchy Coupling<br>ξ = 4π (c/H_0) / (l_node α²)"]
    LN --> XI
    XI --> TMAX["3D Gravimetric Tension<br>T_max = ξ · T_EM"]
    TEM --> TMAX
    TMAX --> GLAP["Laplacian Boundary<br>G_calc = c⁴ / T_max"]
    NU --> PROJ["7-DOF Tensor Trace Projection<br>Factor = 1/7"]
    GLAP --> GACT["Newton's G<br>G = G_calc / 7"]
    PROJ --> GACT
    GACT --> H0["Hubble Tension Resolution<br>H_0 = 69.31 km/s/Mpc"]

    %% ----------------------------------------------------
    %% TOPOLOGICAL MASS HIERARCHY
    %% ----------------------------------------------------
    COS --> FUNC["Cosserat Energy Functional<br>(Kinetic + Skyrme)"]
    LN --> FUNC
    FUNC --> SOLV["Non-Linear Saturation Solver<br>+ 3D Tensor Isospin"]
    SOLV --> MASS["Mass Hierarchy Ratios<br>Muon ≈ 208, Proton ≈ 1832"]

    %% ----------------------------------------------------
    %% THERMODYNAMICS & DARK ENERGY
    %% ----------------------------------------------------
    KV --> RHOV["Invariant Vacuum Density<br>ρ_vac"]
    RHOV --> BAL["Open System 1st Law<br>dU = dQ_latent - P dV"]
    BAL --> W["Stable Phantom Dark Energy<br>w_vac ≈ -1.0001"]
    BAL --> CMB["CMB Thermal Attractor<br>T_CMB = 2.7 K"]

    classDef axiom fill:#d1e7dd,stroke:#c0392b,stroke-width:2px,font-weight:bold;