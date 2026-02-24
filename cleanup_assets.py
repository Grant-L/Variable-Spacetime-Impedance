import os
import re

def find_files(root_dir, exts):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if '.git' in dirpath or 'build' in dirpath or '__pycache__' in dirpath or '.venv' in dirpath or 'tests/' in dirpath or '.gemini' in dirpath:
            continue
        for f in filenames:
            if any(f.endswith(ext) for ext in exts):
                files.append(os.path.join(dirpath, f))
    return files

def cleanup():
    latex_files = find_files('.', ['.tex'])
    image_files = find_files('.', ['.png', '.jpg', '.pdf', '.gif'])
    
    # 1. Extract all \includegraphics
    inc_pattern = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
    referenced_basenames = []
    
    for lf in latex_files:
        with open(lf, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = inc_pattern.findall(content)
            for m in matches:
                referenced_basenames.append(os.path.basename(m))
                # Also add standard extensions just in case
                for ext in ['.png', '.pdf', '.jpg', '.gif']:
                    referenced_basenames.append(os.path.basename(m) + ext)
    
    referenced_basenames = set(referenced_basenames)
    
    # 2. Find internal project PDFs that are output binaries from latex, we shouldn't delete them
    keep_list = ['book_1_foundations.pdf', 'book_2_topological_matter.pdf', 'book_3_macroscopic_continuity.pdf', 'book_4_applied_engineering.pdf', 'book_5_topological_biology.pdf', 'future_work.pdf', 'main.pdf', 'spice_manual.pdf', 'periodic_table.pdf']
    
    deleted_count = 0
    for img in image_files:
        base = os.path.basename(img)
        
        if base in keep_list or base.startswith('circuit_') and base.endswith('.pdf'):
            continue
            
        if base not in referenced_basenames:
            print(f"Deleting unused/obsolete asset: {img}")
            os.remove(img)
            deleted_count += 1
            
    print(f"\nCleanup complete. Deleted {deleted_count} unused/obsolete assets.")

if __name__ == '__main__':
    cleanup()
