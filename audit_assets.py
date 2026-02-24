import os
import re

def find_files(root_dir, exts):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if '.git' in dirpath or 'build' in dirpath or '__pycache__' in dirpath:
            continue
        for f in filenames:
            if any(f.endswith(ext) for ext in exts):
                files.append(os.path.join(dirpath, f))
    return files

def audit():
    latex_files = find_files('.', ['.tex'])
    image_files = find_files('.', ['.png', '.jpg', '.pdf', '.gif'])
    script_files = find_files('.', ['.py'])
    
    # Exclude build dir images
    image_files = [img for img in image_files if '/build/' not in img and '/.gemini/' not in img]
    image_abs_paths = {os.path.abspath(img) for img in image_files}
    
    # 1. Extract all \includegraphics
    inc_pattern = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
    missing_images = []
    
    for lf in latex_files:
        with open(lf, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = inc_pattern.findall(content)
            for m in matches:
                # Resolve relative path from the dir of the latex file
                # Latex files might use relative paths or just filenames if a graphicspath is set.
                lf_dir = os.path.dirname(os.path.abspath(lf))
                target_path = os.path.abspath(os.path.join(lf_dir, m))
                
                # Check if target_path exists in image_abs_paths
                if target_path not in image_abs_paths:
                    # It might be in another dir, or missing extension if latex omits .png
                    # Let's check with standard extensions
                    found = False
                    for ext in ['', '.png', '.pdf', '.jpg']:
                        if os.path.isfile(target_path + ext):
                            found = True
                            break
                    if not found:
                        missing_images.append((m, lf, target_path))

    
    if list(missing_images):
        print(f"\n[!] MISSING OR BADLY LINKED IMAGES ({len(missing_images)}):")
        for m, lf, target in missing_images:
            print(f"  - Ref: {m} in {lf}")
            print(f"    Target absolute path checked: {target}")
    else:
        print("\n[+] All LaTeX relative image paths perfectly resolve to actual files.")
        
if __name__ == '__main__':
    audit()
