# Applied Vacuum Engineering - Master Build System

PYTHON = python3
LATEX = pdflatex
BIBTEX = bibtex
# Note: OUT_DIR is relative to where we run the command. 
# Since we change directory into 'manuscript' to run latex, we use ../build
OUT_DIR = ../build
SRC_DIR = manuscript
SCRIPT_DIR = scripts

.PHONY: all clean sims pdf test help

help:
	@echo "Applied Vacuum Engineering (AVE) Build System"
	@echo "---------------------------------------------"
	@echo "  make sims   : Run simulation suite"
	@echo "  make pdf    : Compile the LaTeX textbook"
	@echo "  make clean  : Remove build artifacts"

all: sims pdf

sims:
	@echo "[Build] Running Simulation Suite..."
	$(PYTHON) $(SCRIPT_DIR)/generate_simulations.py
	@echo "[Build] Asset generation complete."

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
	
	# 2. BibTeX (Run only if .bib file is present)
	@if [ -f $(SRC_DIR)/bibliography.bib ]; then \
		echo "[Build] Processing Bibliography..."; \
		cp $(SRC_DIR)/bibliography.bib build/; \
		cd build && $(BIBTEX) main; \
	fi

	# 3. Second Pass (Link References)
	cd $(SRC_DIR) && $(LATEX) -output-directory=$(OUT_DIR) main.tex
	
	# 4. Third Pass (Resolve Citations & Layout)
	cd $(SRC_DIR) && $(LATEX) -output-directory=$(OUT_DIR) main.tex
	@echo "[Build] PDF generated at build/main.pdf"

clean:
	@echo "[Clean] Removing build artifacts..."
	rm -rf build/*