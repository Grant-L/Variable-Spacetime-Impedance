import os
import re

# Target the manuscript directory relative to project root
TARGET_DIR = "manuscript"

# Regex patterns to remove:
# 1. [cite_start]tags
# 2. [cite: 123] citations
PATTERNS = [(r"\[cite_start\]", ""), (r"\[cite: \d+\]", "")]


def clean_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply all cleaning patterns
        for pattern, replacement in PATTERNS:
            content = re.sub(pattern, replacement, content)

        # Only write if changes were made
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[CLEANED] {filepath}")

    except Exception as e:
        print(f"[ERROR] Could not process {filepath}: {e}")


def main():
    print("--- Starting LaTeX Cleanup ---")
    if not os.path.exists(TARGET_DIR):
        print(f"Directory '{TARGET_DIR}' not found. Ensure you are running from the project root.")
        return

    # Walk through all files in the manuscript directory
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".tex"):
                clean_file(os.path.join(root, file))

    print("--- Cleanup Complete ---")


if __name__ == "__main__":
    main()
