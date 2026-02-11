# Variable Spacetime Impedance - Master Build System

PYTHON = python3
LATEX = pdflatex
BIBTEX = bibtex
OUT_DIR = build
SRC_DIR = manuscript
SIM_DIR = simulations

.PHONY: all clean sims pdf test help

help:
	@echo "VSI Framework Build System"
	@echo "--------------------------"
	@echo "  make sims   : Run all Python simulations and generate figures"
	@echo "  make pdf    : Compile the LaTeX textbook"
	@echo "  make all    : Run tests, simulations, and build PDF"
	@echo "  make clean  : Remove build artifacts"

all: test sims pdf

# 1. Run Simulations
sims:
	@echo "[Build] Running Hardware Simulations..."
	$(PYTHON) $(SIM_DIR)/01_hardware/run_lattice_gen.py
	@echo "[Build] Running Signal Simulations..."
	$(PYTHON) $(SIM_DIR)/02_signal/run_refraction.py
	@echo "[Build] Running Quantum Simulations..."
	$(PYTHON) $(SIM_DIR)/03_quantum/run_pilot_wave.py
	@echo "[Build] Running Topology Simulations..."
	$(PYTHON) $(SIM_DIR)/04_topology/run_proton_knot.py
	@echo "[Build] Running Weak Simulations..."
	$(PYTHON) $(SIM_DIR)/05_weak/run_weak_clamping.py
	@echo "[Build] Running Cosmic Simulations..."
	$(PYTHON) $(SIM_DIR)/06_cosmic/run_cosmic_quench.py
	@echo "[Build] Running Macroscale Simulations..."
	$(PYTHON) $(SIM_DIR)/07_macroscale/run_galactic_rotation.py
	@echo "[Build] Running Engineering Simulations..."
	$(PYTHON) $(SIM_DIR)/08_engineering/run_warp_bubble.py
	@echo "[Build] Running Falsifiability Simulations..."
	$(PYTHON) $(SIM_DIR)/09_falsifiability/run_falsification.py
	@echo "[Build] All simulations complete."

# 2. Build PDF
pdf:
	@echo "[Build] Compiling LaTeX Manuscript..."
	mkdir -p $(OUT_DIR)
	$(LATEX) -output-directory=$(OUT_DIR) $(SRC_DIR)/main.tex
	# Run Bibtex if bibliography exists (commented out until bib file is populated)
	# $(BIBTEX) $(OUT_DIR)/main
	$(LATEX) -output-directory=$(OUT_DIR) $(SRC_DIR)/main.tex
	@echo "[Build] PDF generated at $(OUT_DIR)/main.pdf"

# 3. Unit Tests (Placeholder)
test:
	@echo "[Test] Running Unit Tests..."
	# $(PYTHON) -m unittest discover tests

clean:
	@echo "[Clean] Removing build artifacts..."
	rm -rf $(OUT_DIR)/*
	rm -f assets/sim_outputs/*.png