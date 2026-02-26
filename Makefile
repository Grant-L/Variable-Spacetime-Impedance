# Applied Vacuum Engineering - Master Build System

PYTHON = python3
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

# Directory Configuration
OUT_DIR = build
SRC_DIR = manuscript
SCRIPT_DIR = scripts
SOURCE_DIR = src

# Export PYTHONPATH so scripts can import 'ave' modules from src/
export PYTHONPATH := $(shell pwd)/$(SOURCE_DIR)

.PHONY: all clean verify sims test pdf pdf_manuscript pdf_future_work pdf_spice knots help

help:
	@echo "Applied Vacuum Engineering (AVE) Build System"
	@echo "---------------------------------------------"
	@echo "  make verify     : Run physics verification protocols (The Kernel Check)"
	@echo "  make sims       : Run dynamic simulations (Transmission Lines, etc.)"
	@echo "  make test       : Run unit tests (pytest)"
	@echo "  make pdf        : Compile BOTH the rigorous textbook and future work"
	@echo "  make knots      : Compile the Periodic Table LaTeX document"
	@echo "  make clean      : Remove build artifacts"
	@echo "  make all        : Run verify, sims, and pdf"

all: verify sims pdf

# -----------------------------------------------------------------------------
# 1. Physics Verification (The "Simulate to Verify" Protocol)
# -----------------------------------------------------------------------------
verify:
	@echo "[Verify] Running DAG Anti-Cheat Scan..."
	$(PYTHON) $(SCRIPT_DIR)/book_1_foundations/verify_universe.py
	@echo "\n[Verify] Running FDTD LC Network solvers..."
	$(PYTHON) $(SCRIPT_DIR)/book_4_applied_engineering/visualize_impedance_rupture.py
	@echo "\n[Verify] Running Macroscopic Mutual Inductance bounds..."
	$(PYTHON) $(SCRIPT_DIR)/book_4_applied_engineering/simulate_mutual_inductance.py
	@echo "\n[Verify] Running Topological Borromean geometric limits..."
	$(PYTHON) $(SCRIPT_DIR)/book_1_foundations/visualize_topological_bounds.py
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
	pytest tests/

# -----------------------------------------------------------------------------
# 4. Manuscript Compilation
# -----------------------------------------------------------------------------
pdf: pdf_manuscript pdf_future_work pdf_spice knots

pdf_manuscript:
	@echo "[Build] Setting up build directories for manuscript..."
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@echo "[Build] Compiling Books 1 to 5..."
	@for dir in book_1_foundations book_2_topological_matter book_3_macroscopic_continuity book_4_applied_engineering book_5_topological_biology book_6_ponder_01 book_7_test_article; do \
		echo "[Build] Compiling $$dir..."; \
		rm -f $(OUT_DIR)/aux/$$dir.out $(OUT_DIR)/aux/$$dir.aux $(OUT_DIR)/aux/$$dir.toc; \
		(cd $(SRC_DIR)/$$dir && $(LATEX) -jobname=$$dir -output-directory=../../$(OUT_DIR)/aux main.tex); \
		if [ -f $(SRC_DIR)/bibliography.bib ]; then \
			cp $(SRC_DIR)/bibliography.bib $(OUT_DIR)/aux/; \
			(cd $(OUT_DIR)/aux && $(BIBTEX) $$dir || true); \
			(cd $(SRC_DIR)/$$dir && $(LATEX) -jobname=$$dir -output-directory=../../$(OUT_DIR)/aux main.tex); \
		fi; \
		(cd $(SRC_DIR)/$$dir && $(LATEX) -jobname=$$dir -output-directory=../../$(OUT_DIR)/aux main.tex); \
		mv $(OUT_DIR)/aux/$$dir.pdf $(OUT_DIR)/ 2>/dev/null || true; \
	done
	@echo "[Build] Manuscript PDFs generated in $(OUT_DIR)/ directory."

pdf_future_work:
	@echo "[Build] Setting up build directories for future work..."
	@mkdir -p future_work/build/aux
	@echo "[Build] Compiling Future Work LaTeX Manuscript..."
	@rm -f future_work/build/aux/future_work.out future_work/build/aux/future_work.aux future_work/build/aux/future_work.toc
	@if [ -f future_work/bibliography.bib ]; then \
		cp future_work/bibliography.bib future_work/build/aux/; \
	fi
	@(cd future_work && $(LATEX) -jobname=future_work -output-directory=build/aux main.tex)
	@if [ -f future_work/bibliography.bib ]; then \
		echo "[Build] Processing Bibliography..."; \
		(cd future_work/build/aux && $(BIBTEX) future_work || true); \
		(cd future_work && $(LATEX) -jobname=future_work -output-directory=build/aux main.tex); \
	fi
	@(cd future_work && $(LATEX) -jobname=future_work -output-directory=build/aux main.tex)
	@mv future_work/build/aux/future_work.pdf future_work/build/ 2>/dev/null || true
	@echo "[Build] Future Work PDF generated at future_work/build/future_work.pdf"

pdf_spice:
	@echo "[Build] Setting up build directories for SPICE Manual..."
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@echo "[Build] Compiling SPICE Manual LaTeX Document..."
	@rm -f $(OUT_DIR)/aux/spice_manual.out $(OUT_DIR)/aux/spice_manual.aux $(OUT_DIR)/aux/spice_manual.toc
	@(cd spice_manual && $(LATEX) -jobname=spice_manual -output-directory=../$(OUT_DIR)/aux main.tex)
	@(cd spice_manual && $(LATEX) -jobname=spice_manual -output-directory=../$(OUT_DIR)/aux main.tex)
	@mv $(OUT_DIR)/aux/spice_manual.pdf $(OUT_DIR)/ 2>/dev/null || true
	@echo "[Build] SPICE Manual PDF generated at $(OUT_DIR)/spice_manual.pdf"

# -----------------------------------------------------------------------------
# 5. Periodic Table Compilation
# -----------------------------------------------------------------------------
knots:
	@echo "[Build] Compiling Periodic Table..."
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@rm -f $(OUT_DIR)/aux/periodic_table.out $(OUT_DIR)/aux/periodic_table.aux $(OUT_DIR)/aux/periodic_table.toc
	@(cd periodic_table && $(LATEX) -jobname=periodic_table -output-directory=../$(OUT_DIR)/aux main.tex)
	@(cd periodic_table && $(LATEX) -jobname=periodic_table -output-directory=../$(OUT_DIR)/aux main.tex)
	@mv $(OUT_DIR)/aux/periodic_table.pdf $(OUT_DIR)/ 2>/dev/null || true
	@echo "[Build] Periodic Table PDF generated at $(OUT_DIR)/periodic_table.pdf"

clean:
	@echo "[Clean] Removing auxiliary build artifacts (preserving PDFs)..."
	rm -rf $(OUT_DIR)/aux/*
	rm -rf future_work/build/aux/*
	rm -rf spice_manual/build/*
	rm -rf periodic_table/main.pdf
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

distclean: clean
	@echo "[DistClean] Removing ALL build artifacts including PDFs..."
	rm -rf $(OUT_DIR)/*
	rm -rf future_work/build/*