import os
import re

# --- CONFIGURATION ---
# Path to your manuscript chapters
CHAPTERS_DIR = "manuscript/chapters"

# Keyword Mapping: If the text contains 'Keyword', use 'Citation Key'
# Based on the bibliography.bib we created
CITATION_MAP = {
    # Physics & Relativity
    r"(?i)(general relativity|einstein|field equations|curvature)": "einstein1916",
    r"(?i)(misner|thorne|wheeler|gravitation book)": "misner1973",
    
    # Information Theory & Quantum Limits
    r"(?i)(shannon|information theory|bandwidth|signal processing)": "shannon1949",
    r"(?i)(nyquist|sampling|thermal noise|johnson noise)": "nyquist1928",
    r"(?i)(feynman|qed|path integral)": "feynman1964",
    
    # Space & Anomalies
    r"(?i)(warp|alcubierre|negative energy|bubble)": "alcubierre1994",
    r"(?i)(flyby|anderson|pioneer|anomaly|spacecraft)": "flyby2008",
    
    # Interferometry & Aether
    r"(?i)(michelson|cahill|interferometer|absolute motion)": "cahill2005"
}

def restore_citations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into paragraphs to ensure context is local
    # (This prevents a keyword in paragraph 1 from citing paragraph 10)
    # But for simplicity, we will replace specific \cite{einstein1916} instances
    # by looking at the 500 characters preceding them.
    
    def replacement_logic(match):
        # Look at the text immediately preceding the citation
        start_index = max(0, match.start() - 500)
        preceding_text = content[start_index:match.start()]
        
        # Check for keywords in the preceding text
        for pattern, key in CITATION_MAP.items():
            if re.search(pattern, preceding_text):
                print(f"  -> Found context for {key}")
                return f"\\cite{{{key}}}"
        
        # Default fallback if no specific keyword is found
        return "\\cite{einstein1916}"

    # Regex to find the placeholder citation we added earlier
    # It looks for \cite{einstein1916} or standard [?] markers
    new_content = re.sub(r"\\cite\{einstein1916\}", replacement_logic, content)
    
    # Also catch any missed [?] or [cite_] artifacts
    new_content = re.sub(r"\[cite_.*?\]", replacement_logic, new_content)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {file_path}")

def main():
    print("--- Restoring Smart Citations ---")
    for root, _, files in os.walk(CHAPTERS_DIR):
        for file in files:
            if file.endswith(".tex"):
                restore_citations(os.path.join(root, file))
    print("--- Restoration Complete ---")

if __name__ == "__main__":
    main()