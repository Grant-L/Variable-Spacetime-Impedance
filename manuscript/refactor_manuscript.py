import os
import shutil

books = {
    'book_1_foundations': [
        '00_intro.tex',
        '01_fundamental_axioms.tex',
        '02_macroscopic_moduli.tex',
        '11_continuum_electrodynamics.tex',
        '03_quantum_and_signal_dynamics.tex',
        '13_universal_spatial_tension.tex'
    ],
    'book_2_topological_matter': [
        '05_topological_matter.tex',
        '06_baryon_sector.tex',
        '07_neutrino_sector.tex',
        '08_electroweak_gauge_theory.tex',
        '15_electroweak_and_higgs.tex',
        '16_quantum_mechanics_and_orbitals.tex',
        '14_planck_and_string_theory.tex'
    ],
    'book_3_macroscopic_continuity': [
        '09_macroscopic_relativity.tex',
        '04_gravity_and_yield.tex',
        '19_general_relativity_and_gravity.tex',
        '18_thermodynamics_and_entropy.tex',
        '10_generative_cosmology.tex',
        '17_condensed_matter_superconductivity.tex',
        '20_ideal_gas_law_and_fluid_pressure.tex'
    ],
    'book_4_applied_engineering': [
        '14_applied_fusion.tex',
        '15_antimatter_annihilation.tex'
    ]
}

def create_main_tex(book_dir, book_title):
    main_content = r"""\documentclass[11pt, letterpaper, openright]{book}

% =========================================
% PREAMBLE: THE ENGINEERING AESTHETIC
% =========================================

% --- Typography & Encoding ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern} 
\usepackage{microtype} 

% --- Layout & Geometry ---
\usepackage[margin=1.2in, headheight=14pt]{geometry}
\usepackage{fancyhdr} 
\usepackage{emptypage} 

% --- Mathematics & Science ---
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{siunitx} 
\usepackage{bm} 

% --- Graphics & Floats ---
\usepackage{graphicx}
\graphicspath{{../figures/}{../../assets/}{../../assets/figures/}{../../assets/sim_outputs/}{../../assets/derivations/}{../../assets/archive/}{../../notebooks/periodic/helium/}{../../notebooks/cosmology/}{chapters/}}
\usepackage{float}
\usepackage{booktabs} 
\usepackage{tabularx}
\usepackage{caption}
\usepackage[table,xcdraw]{xcolor}

% --- Navigation & Linking ---
\usepackage[hidelinks]{hyperref} 
\usepackage{tocloft}

% Custom abstract environment for book class
\newenvironment{abstract}
  {\small
   \begin{center}
   \bfseries Abstract\vspace{-.5em}\vspace{0pt}
   \end{center}
   \quotation}
  {\endquotation} 

% --- Code & Verbatim ---
\usepackage{listings}
\usepackage{tcolorbox}
\tcbuselibrary{skins, breakable}

% =========================================
% CUSTOM COMMANDS & DEFINITIONS
% =========================================
\input{../structure/commands.tex}

% Define the "Vacuum Engineering" constants
\newcommand{\vacuum}{\ensuremath{M_A}}
\newcommand{\slew}{\ensuremath{c}}
\newcommand{\planck}{\ensuremath{\hbar}}
\newcommand{\permeability}{\ensuremath{\mu_0}}
\newcommand{\permittivity}{\ensuremath{\epsilon_0}}
\newcommand{\impedance}{\ensuremath{Z_0}}

% Header/Footer Configuration
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\itshape Applied Vacuum Engineering}
\fancyhead[LO]{\itshape\nouppercase{\leftmark}}

% =========================================
% DOCUMENT STRUCTURE
% =========================================

\begin{document}

% --- FRONT MATTER ---
\frontmatter

\input{../frontmatter/00_title.tex}

\tableofcontents

% --- MAIN MATTER ---
\mainmatter

% Reset chapter counter to start at 1
\setcounter{chapter}{0}

\input{chapters/_manifest.tex}

% --- APPENDICES ---
\appendix
\renewcommand{\thechapter}{\Alph{chapter}}
\input{../backmatter/01_appendices.tex}
\input{../backmatter/12_mathematical_closure.tex}

% --- BIBLIOGRAPHY ---
\backmatter
\bibliographystyle{unsrt}
\bibliography{../references}

\end{document}
"""
    with open(f"{book_dir}/main.tex", 'w') as f:
        f.write(main_content)

os.makedirs("chapters", exist_ok=True)

# Create missing source folders if needed
if not os.path.exists('chapters/00_derivations'):
    os.makedirs('chapters/00_derivations')

for book, chapter_list in books.items():
    book_chapter_dir = f"{book}/chapters"
    os.makedirs(book_chapter_dir, exist_ok=True)
    
    manifest_content = ""
    for ch in chapter_list:
        # Move file if it exists in derivations or chapters
        src1 = f"chapters/00_derivations/{ch}"
        src2 = f"chapters/{ch}"
        if os.path.exists(src1):
            shutil.move(src1, f"{book_chapter_dir}/{ch}")
        elif os.path.exists(src2):
            shutil.move(src2, f"{book_chapter_dir}/{ch}")
        else:
            print(f"File not found: {ch}")
        
        manifest_content += f"\\input{{chapters/{ch}}}\n"
    
    if book == 'book_4_applied_engineering':
        # Add the falsifiable predictions manifest
        manifest_content += "\\input{../chapters/13_falsifiable_predictions/_manifest.tex}\n"
        manifest_content += "\\input{../chapters/12_vacuum_circuit_analysis/_manifest.tex}\n"

    with open(f"{book_chapter_dir}/_manifest.tex", 'w') as f:
        f.write(manifest_content)

    create_main_tex(book, book)

print("Distribution Complete.")
