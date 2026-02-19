# Experimental Protocols - Build System

LATEX = pdflatex
BIBTEX = bibtex

# Directory Configuration
OUT_DIR = main
SRC_DIR = .

.PHONY: all clean experiments pdf help

help:
	@echo "Experimental Protocols Build System"
	@echo "-----------------------------------"
	@echo "  make experiments : Compile LaTeX PDF (default)"
	@echo "  make pdf        : Compile LaTeX PDF"
	@echo "  make clean      : Remove build artifacts"
	@echo "  make all        : Same as make experiments"

all: experiments

# -----------------------------------------------------------------------------
# LaTeX Compilation
# -----------------------------------------------------------------------------
experiments: pdf

pdf:
	@echo "[Build] Setting up build directories..."
	@mkdir -p $(OUT_DIR)
	# Mirror the directory structure into main/
	@cd $(SRC_DIR) && find . -type d -not -path "./main*" -not -path "./.git*" -exec mkdir -p $(OUT_DIR)/{} \;
	
	@echo "[Build] Compiling LaTeX Document..."
	# Clean potentially corrupted bookmark files
	@rm -f $(OUT_DIR)/main.out $(OUT_DIR)/main.aux $(OUT_DIR)/main.toc
	
	# 1. First Pass (Generate AUX)
	@echo "[Build] First Pass (pdflatex)..."
	cd $(SRC_DIR) && $(LATEX) -interaction=nonstopmode -output-directory=$(OUT_DIR) main.tex
	
	# 2. BibTeX (Run only if .bib file is present and citations exist)
	@if [ -f $(SRC_DIR)/bibliography.bib ]; then \
		echo "[Build] Processing Bibliography..."; \
		cp $(SRC_DIR)/bibliography.bib $(OUT_DIR)/; \
		cd $(OUT_DIR) && $(BIBTEX) main || true; \
	fi
	
	# 3. Second Pass (Link References)
	@echo "[Build] Second Pass (pdflatex)..."
	cd $(SRC_DIR) && $(LATEX) -interaction=nonstopmode -output-directory=$(OUT_DIR) main.tex
	
	# 4. Third Pass (Resolve Citations & Layout)
	@echo "[Build] Third Pass (pdflatex)..."
	cd $(SRC_DIR) && $(LATEX) -interaction=nonstopmode -output-directory=$(OUT_DIR) main.tex
	@echo ""
	@echo "=================================================="
	@echo "[Build] PDF generated at $(OUT_DIR)/main.pdf"
	@echo "=================================================="

clean:
	@echo "[Clean] Removing build artifacts..."
	@rm -rf $(OUT_DIR)/*
	@rm -f *.aux *.log *.out *.toc *.blg *.bbl
	@find . -name "*.aux" -not -path "./main/*" -delete
	@echo "[Clean] Build artifacts removed."
