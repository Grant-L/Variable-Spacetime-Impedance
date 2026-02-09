#!/bin/bash

# --- LCT Master Build Script ---
# Ensuring the Hardware, Signal, and Topological layers are synchronized.

# 1. Clean up old build artifacts
echo "Cleaning old artifacts..."
rm -f *.aux *.log *.out *.toc *.blg *.bbl chapters/*/*.aux
rm -rf build/*

# 2. Create build directory if it doesn't exist
mkdir -p build

# 3. First LaTeX Pass (Generate .aux files for BibTeX)
echo "Running First Pass (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex

# 4. BibTeX Pass (Link the consolidated bibliography.bib)
echo "Running Bibliography Pass (bibtex)..."
bibtex main

# 5. Second LaTeX Pass (Link citations)
echo "Running Second Pass (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex

# 6. Final LaTeX Pass (Finalize Cross-References & Table of Contents)
echo "Running Final Pass (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex

# 7. Move Final PDF to Build folder
mv main.pdf build/Variable_Spacetime_Impedance.pdf

echo "------------------------------------------------"
echo "BUILD COMPLETE: build/Variable_Spacetime_Impedance.pdf"
echo "------------------------------------------------"