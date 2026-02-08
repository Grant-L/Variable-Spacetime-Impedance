# Variable Spacetime Impedance

A theoretical physics research project exploring novel concepts in spacetime dynamics using computational methods.

## 📁 Project Structure

```
Variable-Spacetime-Impedance/
├── README.md               # Project overview, setup instructions, notebook list
├── requirements.txt         # Dependencies (numpy, matplotlib, scipy, astropy, etc.)
├── .gitignore               # Ignores checkpoints, caches, large data
├── notebooks/               # Jupyter notebooks for step-by-step work
│   ├── 00_template.ipynb
│   ├── 01_Relativistic_Limit.ipynb
│   ├── 02_CMB_BAO_Fitting.ipynb
│   ├── 03_Bullet_Cluster_Sim.ipynb
│   ├── 04_Vacuum_Energy.ipynb
│   ├── 05_Lepton_Asymmetry.ipynb
│   ├── 06_Superconductor_Vortex.ipynb
│   ├── 07_Quantum_Hall.ipynb
│   ├── 08_Gravitational_Waves.ipynb
│   ├── 09_Atomic_Spectra.ipynb
│   ├── 10_Cosmic_Inflation.ipynb
│   └── (Add more as needed, e.g., 11_Theory_Synthesis.ipynb)
├── docs/                    # Paper drafts, abstracts, figures
│   └── (Empty for now—add Markdown or LaTeX files)
├── src/                     # Reusable Python modules/scripts
│   ├── __init__.py
│   └── constants.py         # Physical constants (optional utilities)
├── data/                    # Datasets (e.g., SPARC galaxy data, Planck CMB)
│   └── (Empty—git ignore large files; use Git LFS if needed)
└── simulations/             # Raw outputs, logs, plots from runs
    └── (Empty—store .npy, .csv, .png here)
```

## 🚀 Quick Start

### 1. Initial Setup

Run the setup script to create a virtual environment and install dependencies:

```bash
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Create Jupyter kernel
python -m ipykernel install --user --name=variable-spacetime-impedance
```

### 2. Start Jupyter Lab

```bash
# Activate virtual environment first
source venv/bin/activate

# Start Jupyter Lab
jupyter lab
```

Jupyter Lab will open in your browser at `http://localhost:8888`

### 3. Work with Notebooks

- Use `00_template.ipynb` as a starting point for new notebooks
- Number notebooks sequentially (01_, 02_, etc.) for chronological order
- Select the kernel: **Variable Spacetime Impedance**

## 📓 Notebook List

1. **01_Relativistic_Limit.ipynb** - Relativistic limit calculations
2. **02_CMB_BAO_Fitting.ipynb** - Cosmic Microwave Background and Baryon Acoustic Oscillation fitting
3. **03_Bullet_Cluster_Sim.ipynb** - Bullet Cluster simulations
4. **04_Vacuum_Energy.ipynb** - Vacuum energy calculations
5. **05_Lepton_Asymmetry.ipynb** - Lepton asymmetry analysis
6. **06_Superconductor_Vortex.ipynb** - Superconductor vortex dynamics
7. **07_Quantum_Hall.ipynb** - Quantum Hall effect
8. **08_Gravitational_Waves.ipynb** - Gravitational wave analysis
9. **09_Atomic_Spectra.ipynb** - Atomic spectra calculations
10. **10_Cosmic_Inflation.ipynb** - Cosmic inflation models

## 🔄 Workflow

### Daily Workflow

1. **Start your session:**
   ```bash
   source venv/bin/activate
   jupyter lab
   ```

2. **Work in notebooks:**
   - Use numbered notebooks for specific explorations
   - Keep notebooks focused on specific topics
   - Document your thought process with markdown cells

3. **Save outputs:**
   - Save simulation outputs to `simulations/` directory
   - Store data files in `data/` (large files gitignored)

4. **Commit changes:**
   ```bash
   git add notebooks/your_notebook.ipynb
   git commit -m "Add exploration of [topic]"
   git push
   ```

### Best Practices

- **Notebook naming:** Use numbered prefixes (01_, 02_, etc.) for chronological order
- **Code organization:** Put reusable functions in `src/` modules (e.g., `lattice_utils.py`)
- **Documentation:** Write clear markdown explanations in notebooks
- **Version control:** Commit frequently with descriptive messages
- **Data management:** Large data files should go in `data/` (gitignored by default)
- **Simulations:** Save raw outputs, logs, and plots to `simulations/` directory

## 🛠️ Development Tools

### Python Environment

- **Python 3.11+** recommended
- Virtual environment managed via `venv`
- Dependencies listed in `requirements.txt`

### Key Libraries

- **NumPy & SciPy:** Numerical computations
- **SymPy:** Symbolic mathematics
- **Matplotlib & Seaborn:** Visualization
- **Astropy:** Astronomy/physics utilities
- **Jupyter Lab:** Interactive development environment

### Cursor Integration

This project is optimized for use with Cursor IDE:
- AI-assisted code generation and refactoring
- Intelligent autocomplete for physics calculations
- Integrated terminal for running Jupyter commands
- See `.cursorrules` for project-specific guidelines

## 📚 Documentation

- Paper drafts, abstracts, and figures go in `docs/`
- Notebooks serve as both code and documentation
- Use markdown cells extensively for explanations

## 🤝 Contributing

1. Create a new branch for your work
2. Use descriptive commit messages
3. Keep notebooks clean and well-documented
4. Push changes and create pull requests

## 📝 License

[Add your license here]

## 🔬 Research Notes

[Add your research notes, references, and ideas here]
