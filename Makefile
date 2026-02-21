# Applied Vacuum Engineering - Master Build System

PYTHON = python3
LATEX = pdflatex
BIBTEX = bibtex

# Directory Configuration
OUT_DIR = ../build
SRC_DIR = manuscript
SCRIPT_DIR = scripts
SOURCE_DIR = src

# Export PYTHONPATH so scripts can import 'ave' modules from src/
export PYTHONPATH := $(shell pwd)/$(SOURCE_DIR)

.PHONY: all clean verify sims test pdf experiments knots help

help:
	@echo "Applied Vacuum Engineering (AVE) Build System"
	@echo "---------------------------------------------"
	@echo "  make verify     : Run physics verification protocols (The Kernel Check)"
	@echo "  make sims       : Run dynamic simulations (Transmission Lines, etc.)"
	@echo "  make test       : Run unit tests (pytest)"
	@echo "  make pdf        : Compile BOTH the rigorous textbook and future work"
	@echo "  make experiments: Compile the Experimental Protocols LaTeX document"
	@echo "  make knots      : Compile the Periodic Table of Knots LaTeX document"
	@echo "  make clean      : Remove build artifacts"
	@echo "  make all        : Run verify, sims, and pdf"

all: verify sims pdf

# -----------------------------------------------------------------------------
# 1. Physics Verification (The "Simulate to Verify" Protocol)
# -----------------------------------------------------------------------------
verify:
	@echo "[Verify] Running DAG Anti-Cheat Scan..."
	$(PYTHON) $(SCRIPT_DIR)/verify_universe.py
	@echo "\n[Verify] Running FDTD LC Network solvers..."
	$(PYTHON) $(SCRIPT_DIR)/visualize_impedance_rupture.py
	@echo "\n[Verify] Running Macroscopic Mutual Inductance bounds..."
	$(PYTHON) $(SCRIPT_DIR)/simulate_mutual_inductance.py
	@echo "\n[Verify] Running Topological Borromean geometric limits..."
	$(PYTHON) $(SCRIPT_DIR)/visualize_topological_bounds.py
	@echo "\n=================================================="
	@echo "[Verify] ALL PHYSICS PROTOCOLS PASSED."
	@echo "=================================================="
	
# -----------------------------------------------------------------------------
# 2. Dynamic Simulations (Visual Assets & Time-Domain Solvers)
# -----------------------------------------------------------------------------
sims:
	@echo "[Sims] Spacetime Circuit Analysis is now unified under verify routines."
	@echo "[Sims] Simulation suite complete."

# -----------------------------------------------------------------------------
# 3. Unit Testing
# -----------------------------------------------------------------------------
test:
	@echo "[Test] Running Unit Tests..."
	# Requires pytest to be installed
	pytest tests/

# -----------------------------------------------------------------------------
# 4. Manuscript Compilation
# -----------------------------------------------------------------------------
pdf: pdf_manuscript pdf_future_work

pdf_manuscript:
	@echo "[Build] Setting up build directories for manuscript..."
	@mkdir -p build
	# Mirror the manuscript directory structure into build/
	@cd $(SRC_DIR) && find . -type d -exec mkdir -p ../build/{} \;
	
	@echo "[Build] Compiling Main LaTeX Manuscript..."
	# Clean potentially corrupted bookmark files
	rm -f build/main.out build/main.aux build/main.toc
	
	# 1. First Pass (Generate AUX)
	cd $(SRC_DIR) && $(LATEX) -output-directory=$(OUT_DIR) main.tex
	
	# 2. BibTeX (Run only if .bib file is present and citations exist)
	@if [ -f $(SRC_DIR)/bibliography.bib ]; then \
		echo "[Build] Processing Bibliography..."; \
		cp $(SRC_DIR)/bibliography.bib build/; \
		cd build && $(BIBTEX) main || true; \
	fi

	# 3. Second Pass (Link References)
	cd $(SRC_DIR) && $(LATEX) -output-directory=$(OUT_DIR) main.tex
	
	# 4. Third Pass (Resolve Citations & Layout)
	cd $(SRC_DIR) && $(LATEX) -output-directory=$(OUT_DIR) main.tex
	@echo "[Build] Academic PDF generated at build/main.pdf"

pdf_future_work:
	@echo "[Build] Setting up build directories for future work..."
	@mkdir -p build_future
	@cd future_work && find . -type d -exec mkdir -p ../build_future/{} \;
	
	@echo "[Build] Compiling Future Work LaTeX Manuscript..."
	rm -f build_future/main.out build_future/main.aux build_future/main.toc
	cd future_work && $(LATEX) -output-directory=../build_future main.tex
	@if [ -f future_work/bibliography.bib ]; then \
		echo "[Build] Processing Bibliography..."; \
		cp future_work/bibliography.bib build_future/; \
		cd build_future && $(BIBTEX) main || true; \
	fi
	cd future_work && $(LATEX) -output-directory=../build_future main.tex
	cd future_work && $(LATEX) -output-directory=../build_future main.tex
	@echo "[Build] Future Work PDF generated at build_future/main.pdf"

# -----------------------------------------------------------------------------
# 5. Experimental Protocols Compilation
# -----------------------------------------------------------------------------
experiments:
	@echo "[Build] Compiling Experimental Protocols..."
	@cd experiments && $(MAKE) pdf
	@echo "[Build] Experimental Protocols PDF generated at experiments/main/main.pdf"

# -----------------------------------------------------------------------------
# 6. Periodic Table of Knots Compilation
# -----------------------------------------------------------------------------
knots:
	@echo "[Build] Compiling Periodic Table of Knots..."
	@cd periodic_table_of_knots && $(MAKE) pdf
	@echo "[Build] Periodic Table of Knots PDF generated at periodic_table_of_knots/build/main.pdf"

clean:
	@echo "[Clean] Removing build artifacts..."
	rm -rf build/*
	rm -rf build_future/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +