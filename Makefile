# Applied Vacuum Engineering - Master Build System

PYTHON = python3
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

# Directory Configuration
OUT_DIR = build
SRC_DIR = manuscript
SCRIPT_DIR = scripts
SOURCE_DIR = src

# Volume list (single source of truth)
VOLUMES = vol_1_foundations vol_2_subatomic vol_3_macroscopic vol_4_engineering vol_5_biology

# Export PYTHONPATH so scripts can import 'ave' modules from src/
export PYTHONPATH := $(shell pwd)/$(SOURCE_DIR)

.PHONY: all clean distclean verify test pdf pdf_manuscript pdf_future_work pdf_spice periodic_table figures help vol1 vol2 vol3 vol4 vol5

help:
	@echo "Applied Vacuum Engineering (AVE) Build System"
	@echo "---------------------------------------------"
	@echo "  make all             : Run verify, then compile all PDFs"
	@echo "  make verify          : Run physics verification protocols (The Kernel Check)"
	@echo "  make test            : Run unit tests (pytest)"
	@echo "  make pdf             : Compile all documents (5 volumes + future work + SPICE + periodic table)"
	@echo "  make pdf_manuscript  : Compile manuscript Volumes I-V"
	@echo "  make vol1            : Vol I:  Foundations & Universal Operators"
	@echo "  make vol2            : Vol II: The Subatomic Lattice"
	@echo "  make vol3            : Vol III: The Macroscopic Continuum"
	@echo "  make vol4            : Vol IV: Applied Impedance Engineering"
	@echo "  make vol5            : Vol V:  Topological Biology"
	@echo "  make pdf_future_work : Compile speculative future work document"
	@echo "  make pdf_spice       : Compile SPICE manual"
	@echo "  make periodic_table  : Compile the Periodic Table document"
	@echo "  make figures         : Generate particle topology figure suite"
	@echo "  make clean           : Remove auxiliary build artifacts (preserves PDFs)"
	@echo "  make distclean       : Remove ALL build artifacts including PDFs"

all: verify pdf

# =============================================================================
# 1. Physics Verification (The "Simulate to Verify" Protocol)
# =============================================================================
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

# =============================================================================
# 2. Unit Testing
# =============================================================================
test:
	@echo "[Test] Running Unit Tests..."
	$(PYTHON) -m pytest tests/

# =============================================================================
# 3. Manuscript Compilation
# =============================================================================

# --- Single volume compilation macro ---
# Usage: $(call COMPILE_VOL,vol_name)
# Runs: pdflatex → bibtex → pdflatex → pdflatex (standard triple-pass)
define COMPILE_VOL
	@mkdir -p $(OUT_DIR)/aux
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
	@echo "[Build] $(1).pdf → $(OUT_DIR)/"
endef

# --- Compile all 5 volumes ---
pdf: pdf_manuscript pdf_future_work pdf_spice periodic_table

pdf_manuscript:
	@echo "[Build] Compiling Volumes I–V..."
	@for dir in $(VOLUMES); do \
		$(MAKE) --no-print-directory _compile_vol VOL=$$dir; \
	done
	@echo "[Build] All 5 volume PDFs generated in $(OUT_DIR)/"

# Internal target used by the loop
_compile_vol:
	$(call COMPILE_VOL,$(VOL))

# --- Individual volume targets ---
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

# --- Standalone document compilation macro ---
# Usage: $(call COMPILE_DOC,dir_name,job_name)
define COMPILE_DOC
	@mkdir -p $(OUT_DIR)/aux
	@echo "[Build] Compiling $(2)..."
	@rm -f $(OUT_DIR)/aux/$(2).out $(OUT_DIR)/aux/$(2).aux $(OUT_DIR)/aux/$(2).toc
	@(cd $(1) && $(LATEX) -jobname=$(2) -output-directory=../$(OUT_DIR)/aux main.tex)
	@if [ -f $(1)/bibliography.bib ]; then \
		cp $(1)/bibliography.bib $(OUT_DIR)/aux/; \
		(cd $(OUT_DIR)/aux && $(BIBTEX) $(2) || true); \
		(cd $(1) && $(LATEX) -jobname=$(2) -output-directory=../$(OUT_DIR)/aux main.tex); \
	fi
	@(cd $(1) && $(LATEX) -jobname=$(2) -output-directory=../$(OUT_DIR)/aux main.tex)
	@mv $(OUT_DIR)/aux/$(2).pdf $(OUT_DIR)/ 2>/dev/null || true
	@echo "[Build] $(2).pdf → $(OUT_DIR)/"
endef

pdf_future_work:
	$(call COMPILE_DOC,future_work,future_work)

pdf_spice:
	$(call COMPILE_DOC,spice_manual,spice_manual)

periodic_table:
	$(call COMPILE_DOC,periodic_table,periodic_table)

# =============================================================================
# 4. Figure Generation
# =============================================================================
figures:
	@echo "[Figures] Generating particle topology suite..."
	$(PYTHON) $(SCRIPT_DIR)/vol_2_subatomic/generate_particle_topology_suite.py
	@echo "[Figures] Regenerating electron topology figure..."
	$(PYTHON) $(SCRIPT_DIR)/vol_2_subatomic/simulate_electron_topology.py
	@echo "[Figures] All figures generated."

# =============================================================================
# 5. Cleanup
# =============================================================================
clean:
	@echo "[Clean] Removing auxiliary build artifacts (preserving PDFs)..."
	rm -rf $(OUT_DIR)/aux
	@echo "[Clean] Removing in-tree LaTeX artifacts..."
	@find $(SRC_DIR) future_work spice_manual periodic_table \
		\( -name "*.aux" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \
		   -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.out" \
		   -o -name "*.log" -o -name "*.synctex.gz" -o -name "*.bbl" \
		   -o -name "*.blg" \) -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "[Clean] Done."

distclean: clean
	@echo "[DistClean] Removing ALL build artifacts including PDFs..."
	rm -rf $(OUT_DIR)
	@echo "[DistClean] Done."