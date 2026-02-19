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

.PHONY: all clean verify sims test pdf help

help:
	@echo "Applied Vacuum Engineering (AVE) Build System"
	@echo "---------------------------------------------"
	@echo "  make verify : Run physics verification protocols (The Kernel Check)"
	@echo "  make sims   : Run dynamic simulations (Transmission Lines, etc.)"
	@echo "  make test   : Run unit tests (pytest)"
	@echo "  make pdf    : Compile the LaTeX textbook"
	@echo "  make clean  : Remove build artifacts"
	@echo "  make all    : Run verify, sims, and pdf"

all: verify sims pdf

# -----------------------------------------------------------------------------
# 1. Physics Verification (The "Simulate to Verify" Protocol)
# -----------------------------------------------------------------------------
verify:
	@echo "[Verify] Running Single-Parameter Kernel Check..."
	$(PYTHON) $(SCRIPT_DIR)/verify_universe.py
	@echo "[Verify] Checking Fundamental Forces..."
	$(PYTHON) $(SCRIPT_DIR)/verify_fundamental_forces.py
	@echo "[Verify] Checking Matter Sector (Leptons/Baryons/Neutrinos)..."
	$(PYTHON) $(SCRIPT_DIR)/verify_matter_sector.py
	@echo "[Verify] Checking Cosmological Dynamics..."
	$(PYTHON) $(SCRIPT_DIR)/verify_cosmology.py
	@echo "[Verify] All physics protocols PASSED."

# -----------------------------------------------------------------------------
# 2. Dynamic Simulations (Visual Assets & Time-Domain Solvers)
# -----------------------------------------------------------------------------
sims:
	@echo "[Sims] Running Spacetime Circuit Analysis..."
	# Note: Generates Figure 12.3 (Transmission Line Velocity)
	$(PYTHON) $(SCRIPT_DIR)/simulate_chapter_12.py
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
pdf:
	@echo "[Build] Setting up build directories..."
	@mkdir -p build
	# Mirror the manuscript directory structure into build/
	@cd $(SRC_DIR) && find . -type d -exec mkdir -p ../build/{} \;
	
	@echo "[Build] Compiling LaTeX Manuscript..."
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
	@echo "[Build] PDF generated at build/main.pdf"

clean:
	@echo "[Clean] Removing build artifacts..."
	rm -rf build/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +