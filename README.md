# Variable Spacetime Impedance

A theoretical physics research project exploring novel concepts in spacetime dynamics using computational methods.

## 🎯 Project Overview

This repository is structured for collaborative theoretical physics research using:
- **Jupyter Lab** for interactive exploration and calculations
- **GitHub** for version control and collaboration
- **Cursor** for code editing and AI-assisted development

## 📁 Project Structure

```
Variable-Spacetime-Impedance/
├── notebooks/          # Jupyter notebooks for exploration
│   └── 00_template.ipynb  # Template for new notebooks
├── src/                # Reusable Python modules
│   ├── __init__.py
│   └── constants.py    # Physical constants
├── docs/               # Documentation and notes
├── data/               # Data files (gitignored if large)
├── results/            # Generated results and outputs
├── .jupyter/           # Jupyter configuration
├── requirements.txt    # Python dependencies
├── setup.sh           # Setup script
└── README.md          # This file
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

### 3. Create a New Notebook

1. Copy `notebooks/00_template.ipynb` to create a new notebook
2. Name it with a descriptive prefix (e.g., `01_spacetime_metric.ipynb`)
3. Select the kernel: **Variable Spacetime Impedance**

## 🔄 Workflow

### Daily Workflow

1. **Start your session:**
   ```bash
   source venv/bin/activate
   jupyter lab
   ```

2. **Work in notebooks:**
   - Use the template notebook as a starting point
   - Keep notebooks focused on specific explorations
   - Document your thought process with markdown cells

3. **Save and commit:**
   ```bash
   git add notebooks/your_notebook.ipynb
   git commit -m "Add exploration of [topic]"
   git push
   ```

### Best Practices

- **Notebook naming:** Use numbered prefixes (00_, 01_, 02_) for chronological order
- **Code organization:** Put reusable functions in `src/` modules
- **Documentation:** Write clear markdown explanations in notebooks
- **Version control:** Commit frequently with descriptive messages
- **Data management:** Large data files should go in `data/` (may be gitignored)

## 🛠️ Development Tools

### Python Environment

- **Python 3.11+** recommended
- Virtual environment managed via `venv`
- Dependencies listed in `requirements.txt`

### Key Libraries

- **NumPy & SciPy:** Numerical computations
- **SymPy:** Symbolic mathematics
- **Matplotlib & Seaborn:** Visualization
- **Jupyter Lab:** Interactive development environment

### Cursor Integration

This project is optimized for use with Cursor IDE:
- AI-assisted code generation and refactoring
- Intelligent autocomplete for physics calculations
- Integrated terminal for running Jupyter commands

## 📚 Documentation

- See `docs/README.md` for documentation structure
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
