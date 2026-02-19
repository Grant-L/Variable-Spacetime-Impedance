#!/bin/bash

# Build script for Experimental Protocols LaTeX document

# 1. Clean up old build artifacts
echo "Cleaning old artifacts..."
rm -f *.aux *.log *.out *.toc *.blg *.bbl chapters/*/*.aux
rm -rf main/*

# 2. Create build directory if it doesn't exist
mkdir -p main

# 3. First LaTeX Pass (Generate .aux files for BibTeX)
echo "Running First Pass (pdflatex)..."
pdflatex -interaction=nonstopmode -output-directory=main main.tex

# 4. BibTeX Pass (Link the bibliography)
echo "Running Bibliography Pass (bibtex)..."
cd main
bibtex main || true
cd ..

# 5. Second LaTeX Pass (Link citations)
echo "Running Second Pass (pdflatex)..."
pdflatex -interaction=nonstopmode -output-directory=main main.tex

# 6. Final LaTeX Pass (Finalize Cross-References & Table of Contents)
echo "Running Final Pass (pdflatex)..."
pdflatex -interaction=nonstopmode -output-directory=main main.tex

echo "------------------------------------------------"
echo "BUILD COMPLETE: main/main.pdf"
echo "------------------------------------------------"
