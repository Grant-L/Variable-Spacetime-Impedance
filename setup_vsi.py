import os
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(".")  # Runs in current directory

# Structure Definition based on the "Atomic Chapter" protocol
STRUCTURE = {
    "assets/figures": ["01_hardware", "02_signal", "03_quantum", "04_topology", 
                       "05_weak", "06_cosmic", "07_engineering", "08_engineering", "08_falsifiability"],
    "assets/references": [],
    "build": [],
    "manuscript/structure": ["preamble.tex", "commands.tex", "titlepage.tex"],
    "manuscript/frontmatter": ["00_preface.tex", "01_glossary.tex"],
    "manuscript/backmatter": ["appendix_math.tex", "appendix_code.tex"],
    "simulations": ["01_hardware", "02_signal", "03_quantum", "04_topology", 
                    "05_weak", "06_cosmic", "07_engineering", "08_engineering", "08_falsifiability"],
    "src/vsi": ["__init__.py", "constants.py", "lattice.py", "solver.py", "metric.py"],
    "tests": [],
    "notebooks": []
}

# Chapter breakdown
CHAPTERS = {
    "01_hardware": "The Hardware Layer: Vacuum Constitutive Properties",
    "02_signal": "The Signal Layer: Variable Impedance and Mass Emergence",
    "03_quantum": "The Quantum Layer: Defects and Chiral Exclusion",
    "04_topology": "The Topological Layer: Matter as Defects",
    "05_weak": "The Weak Interaction: Chiral Clamping",
    "06_cosmic": "Cosmic Evolution: The Quench",
    "07_engineering": "Engineering Layer: Rotation and Impedance",
    "08_engineering": "The Engineering Layer: Metric Refraction",
    "08_falsifiability": "Falsifiability: The Universal Means Test"
}

# --- Templates ---

CONSTANTS_PY_CONTENT = """
# VSI Hardware Constants (The Rosetta Stone)
class HardwareConstants:
    L_NODE = 1.26e-6  # H/m
    C_NODE = 8.85e-12 # F/m
    l_P = 1.62e-35    # m

    @property
    def Z_0(self):
        return (self.L_NODE / self.C_NODE) ** 0.5

    @property
    def c(self):
        return 1.0 / (self.L_NODE * self.C_NODE) ** 0.5
"""

# Default Preface Content (This was missing and causing the LaTeX error)
PREFACE_CONTENT = r"""\chapter*{Preface}
\addcontentsline{toc}{chapter}{Preface}

Theoretical physics has reached a juncture where the mathematical complexity of our models has outpaced our mechanical understanding. This text proposes a return to hardware: treating the vacuum not as a geometric abstraction, but as a \textbf{Discrete Amorphous Manifold ($M_A$)}.

\section*{The Shift from Geometry to Hardware}
The central thesis of this work is that the vacuum is a physical substrate governed by finite inductive and capacitive densities. By redefining the fundamental constants of nature as the bulk engineering properties of this substrate, we move from a descriptive physics to an operational one.
"""

LATEX_CHAPTER_MANIFEST = r"""\chapter{{{title}}}
\label{{ch:{slug}}}

\input{{chapters/{folder}/0{i}_00_intro}}
\input{{chapters/{folder}/0{i}_01_theory}}
\input{{chapters/{folder}/0{i}_02_simulation}}
\input{{chapters/{folder}/0{i}_03_implications}}
\input{{chapters/{folder}/0{i}_04_exercises}}
"""

# --- Execution ---

def create_file(path, content=""):
    """Creates a file safely. Does NOT overwrite if content exists, unless empty."""
    if path.exists() and path.stat().st_size > 0:
        print(f"[SKIP] Exists: {path}")
        return
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[CREATE] {path}")
    except Exception as e:
        print(f"[ERROR] Could not create {path}: {e}")

def main():
    print("--- Initializing VSI Project Structure (Safe Mode) ---")

    # 1. Create Base Directories
    for folder, subfolders in STRUCTURE.items():
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
        for sub in subfolders:
            path = PROJECT_ROOT / folder / sub
            if "." in sub: 
                # Populating specific files
                content = ""
                if sub == "constants.py": content = CONSTANTS_PY_CONTENT
                if sub == "00_preface.tex": content = PREFACE_CONTENT
                create_file(path, content)
            else:
                path.mkdir(parents=True, exist_ok=True)

    # 2. Generate Chapters
    chapters_dir = PROJECT_ROOT / "manuscript/chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    for i, (folder, title) in enumerate(CHAPTERS.items(), start=1):
        chapter_path = chapters_dir / folder
        chapter_path.mkdir(exist_ok=True)
        
        # Manifest
        manifest_content = LATEX_CHAPTER_MANIFEST.format(
            title=title, slug=folder, folder=folder, i=i
        )
        create_file(chapter_path / "_manifest.tex", manifest_content)

        # Sections
        sections = ["00_intro", "01_theory", "02_simulation", "03_implications", "04_exercises"]
        for sec in sections:
            filename = f"0{i}_{sec}.tex"
            create_file(chapter_path / filename, f"% Section: {sec.replace('_', ' ').title()}\n")

    # 3. Create Root Files
    create_file(PROJECT_ROOT / ".gitignore", "*.pyc\n__pycache__\nbuild/\n.DS_Store\n")
    create_file(PROJECT_ROOT / "Makefile", "all:\n\t@echo 'Run make pdf or make sims'\n")

    print("\n--- Setup Complete. ---")

if __name__ == "__main__":
    main()