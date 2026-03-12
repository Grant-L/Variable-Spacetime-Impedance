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

.PHONY: all clean distclean verify test pdf pdf_manuscript pdf_future_work pdf_spice periodic_table figures help vol1 vol2 vol3 vol4 vol5

help:
	@echo "Applied Vacuum Engineering (AVE) Build System"
	@echo "---------------------------------------------"
	@echo "  make all             : Run verify, then compile all PDFs"
	@echo "  make verify          : Run physics verification protocols (The Kernel Check)"
	@echo "  make test            : Run unit tests (pytest)"
	@echo "  make pdf             : Compile all documents (manuscript, future work, SPICE, periodic table)"
	@echo "  make pdf_manuscript  : Compile manuscript Volumes I-V only"
	@echo "  make vol1            : Compile Vol I: Foundations only"
	@echo "  make vol2            : Compile Vol II: Subatomic only"
	@echo "  make vol3            : Compile Vol III: Macroscopic only"
	@echo "  make vol4            : Compile Vol IV: Engineering only"
	@echo "  make vol5            : Compile Vol V: Biology only"
	@echo "  make pdf_future_work : Compile future work document only"
	@echo "  make pdf_spice       : Compile SPICE manual only"
	@echo "  make periodic_table  : Compile the Periodic Table document"
	@echo "  make figures         : Generate particle topology figure suite"
	@echo "  make clean           : Remove auxiliary build artifacts (preserves PDFs)"
	@echo "  make distclean       : Remove ALL build artifacts including PDFs"

all: verify pdf

# -----------------------------------------------------------------------------
# 1. Physics Verification (The "Simulate to Verify" Protocol)
# -----------------------------------------------------------------------------
verify:
	@echo "[Verify] Running DAG Anti-Cheat Scan..."
	$(PYTHON) $(SCRIPT_DIR)/vol_1_foundations/verify_universe.py
	@echo "\n[Verify] Running FDTD LC Network solvers..."
	$(PYTHON) $(SCRIPT_DIR)/vol_4_engineering/visualize_impedance_rupture.py
	@echo "\n[Verify] Running Macroscopic Mutual Inductance bounds..."
	$(PYTHON) $(SCRIPT_DIR)/vol_4_engineering/simulate_mutual_inductance.py
	@echo "\n[Verify] Running Topological Borromean geometric limits..."
	$(PYTHON) $(SCRIPT_DIR)/vol_1_foundations/visualize_topological_bounds.py
	@echo "\n=================================================="
	@echo "[Verify] ALL PHYSICS PROTOCOLS PASSED."
	@echo "=================================================="

# -----------------------------------------------------------------------------
# 2. Unit Testing
# -----------------------------------------------------------------------------
test:
	@echo "[Test] Running Unit Tests..."
	$(PYTHON) -m pytest tests/

# -----------------------------------------------------------------------------
# 3. Manuscript Compilation
# -----------------------------------------------------------------------------
pdf: pdf_manuscript pdf_future_work pdf_spice periodic_table

pdf_manuscript:
	@echo "[Build] Setting up build directories for manuscript..."
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@echo "[Build] Compiling Volumes I to V..."
	@for dir in vol_1_foundations vol_2_subatomic vol_3_macroscopic vol_4_engineering vol_5_biology; do \
		echo "[Build] Compiling $$dir..."; \
		rm -f $(OUT_DIR)/aux/$$dir.out $(OUT_DIR)/aux/$$dir.aux $(OUT_DIR)/aux/$$dir.toc; \
		(cd $(SRC_DIR)/$$dir && $(LATEX) -jobname=$$dir -output-directory=../../$(OUT_DIR)/aux main.tex); \
		if [ -f $(SRC_DIR)/bibliography.bib ]; then \
			cp $(SRC_DIR)/bibliography.bib $(OUT_DIR)/aux/; \
			cp $(SRC_DIR)/bibliography.bib $(OUT_DIR)/bibliography.bib; \
			(cd $(OUT_DIR)/aux && $(BIBTEX) $$dir || true); \
			(cd $(SRC_DIR)/$$dir && $(LATEX) -jobname=$$dir -output-directory=../../$(OUT_DIR)/aux main.tex); \
		fi; \
		(cd $(SRC_DIR)/$$dir && $(LATEX) -jobname=$$dir -output-directory=../../$(OUT_DIR)/aux main.tex); \
		mv $(OUT_DIR)/aux/$$dir.pdf $(OUT_DIR)/ 2>/dev/null || true; \
	done
	@echo "[Build] Manuscript PDFs generated in $(OUT_DIR)/ directory."

# Individual volume targets
define COMPILE_VOL
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@echo "[Build] Compiling $(1)..."
	@rm -f $(OUT_DIR)/aux/$(1).out $(OUT_DIR)/aux/$(1).aux $(OUT_DIR)/aux/$(1).toc
	@(cd $(SRC_DIR)/$(1) && $(LATEX) -jobname=$(1) -output-directory=../../$(OUT_DIR)/aux main.tex)
	@if [ -f $(SRC_DIR)/bibliography.bib ]; then \
		cp $(SRC_DIR)/bibliography.bib $(OUT_DIR)/aux/; \
		(cd $(OUT_DIR)/aux && $(BIBTEX) $(1) || true); \
		(cd $(SRC_DIR)/$(1) && $(LATEX) -jobname=$(1) -output-directory=../../$(OUT_DIR)/aux main.tex); \
	fi
	@(cd $(SRC_DIR)/$(1) && $(LATEX) -jobname=$(1) -output-directory=../../$(OUT_DIR)/aux main.tex)
	@mv $(OUT_DIR)/aux/$(1).pdf $(OUT_DIR)/ 2>/dev/null || true
	@echo "[Build] $(1).pdf generated in $(OUT_DIR)/"
endef

vol1:
	$(call COMPILE_VOL,vol_1_foundations)

vol2:
	$(call COMPILE_VOL,vol_2_subatomic)

vol3:
	$(call COMPILE_VOL,vol_3_macroscopic)

vol4:
	$(call COMPILE_VOL,vol_4_engineering)

vol5:
	$(call COMPILE_VOL,vol_5_biology)

pdf_future_work:
	@echo "[Build] Setting up build directories for future work..."
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@echo "[Build] Compiling Future Work LaTeX Manuscript..."
	@rm -f $(OUT_DIR)/aux/future_work.out $(OUT_DIR)/aux/future_work.aux $(OUT_DIR)/aux/future_work.toc
	@if [ -f future_work/bibliography.bib ]; then \
		cp future_work/bibliography.bib $(OUT_DIR)/aux/; \
	fi
	@(cd future_work && $(LATEX) -jobname=future_work -output-directory=../$(OUT_DIR)/aux main.tex)
	@if [ -f future_work/bibliography.bib ]; then \
		echo "[Build] Processing Bibliography..."; \
		(cd $(OUT_DIR)/aux && $(BIBTEX) future_work || true); \
		(cd future_work && $(LATEX) -jobname=future_work -output-directory=../$(OUT_DIR)/aux main.tex); \
	fi
	@(cd future_work && $(LATEX) -jobname=future_work -output-directory=../$(OUT_DIR)/aux main.tex)
	@mv $(OUT_DIR)/aux/future_work.pdf $(OUT_DIR)/ 2>/dev/null || true
	@echo "[Build] Future Work PDF generated at $(OUT_DIR)/future_work.pdf"

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
# 4. Periodic Table Compilation
# -----------------------------------------------------------------------------
periodic_table:
	@echo "[Build] Compiling Periodic Table..."
	@mkdir -p $(OUT_DIR)/aux/chapters $(OUT_DIR)/aux/frontmatter $(OUT_DIR)/aux/backmatter
	@rm -f $(OUT_DIR)/aux/periodic_table.out $(OUT_DIR)/aux/periodic_table.aux $(OUT_DIR)/aux/periodic_table.toc
	@(cd periodic_table && $(LATEX) -jobname=periodic_table -output-directory=../$(OUT_DIR)/aux main.tex)
	@(cd periodic_table && $(LATEX) -jobname=periodic_table -output-directory=../$(OUT_DIR)/aux main.tex)
	@mv $(OUT_DIR)/aux/periodic_table.pdf $(OUT_DIR)/ 2>/dev/null || true
	@echo "[Build] Periodic Table PDF generated at $(OUT_DIR)/periodic_table.pdf"

# -----------------------------------------------------------------------------
# 5. Figure Generation
# -----------------------------------------------------------------------------
figures:
	@echo "[Figures] Generating particle topology suite..."
	$(PYTHON) $(SCRIPT_DIR)/vol_2_subatomic/generate_particle_topology_suite.py
	@echo "[Figures] Regenerating electron topology figure..."
	$(PYTHON) $(SCRIPT_DIR)/vol_2_subatomic/simulate_electron_topology.py
	@echo "[Figures] All figures generated."

clean:
	@echo "[Clean] Removing auxiliary build artifacts (preserving build/ PDFs)..."
	rm -rf $(OUT_DIR)/aux/*
	rm -rf future_work/build/
	rm -rf spice_manual/build/
	rm -rf periodic_table/main.pdf
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "[Clean] Removing in-tree LaTeX artifacts from volume directories..."
	rm -f $(SRC_DIR)/vol_*/main.pdf
	find $(SRC_DIR)/vol_* -maxdepth 1 \( -name "*.aux" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.out" -o -name "*.log" -o -name "*.synctex.gz" \) -delete 2>/dev/null || true

distclean: clean
	@echo "[DistClean] Removing ALL build artifacts including PDFs..."
	rm -rf $(OUT_DIR)/*