cat << 'EOF' > clean_latex.py
import os
import re

# Define the directory to clean relative to where the script is run
TARGET_DIR = "manuscript"

# Regex patterns to remove
# 1. Matches 
# [cite_start]2. Matches [cite: 123] [cite_start]or [cite: 1, 2]
PATTERNS = [
    (r"\[cite_start\]", ""), 
    (r"\", "")
]

def clean_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in PATTERNS:
            content = re.sub(pattern, replacement, content)
            
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[CLEANED] {filepath}")
            
    except Exception as e:
        print(f"[ERROR] Could not process {filepath}: {e}")

def main():
    print("--- Starting LaTeX Cleanup ---")
    if not os.path.exists(TARGET_DIR):
        print(f"Directory '{TARGET_DIR}' not found. Make sure you are running this from the project root.")
        return

    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".tex"):
                clean_file(os.path.join(root, file))
    
    print("--- Cleanup Complete ---")

if __name__ == "__main__":
    main()
EOF