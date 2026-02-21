import os
import re

chapters_dir = 'future_work/chapters'
tex_files = []

for root, _, files in os.walk(chapters_dir):
    for f in files:
        if f.endswith('.tex'):
            tex_files.append(os.path.join(root, f))

for tex_file in tex_files:
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Regex to find \includegraphics and extract just the filename
    # Matches \includegraphics[...]{path/to/image.png} or \includegraphics{path/to/image.png}
    def replace_image(match):
        pre = match.group(1) # \includegraphics[width=...]
        full_path = match.group(2) # path/to/image.png
        filename = os.path.basename(full_path)
        return f"{pre}{{{filename}}}"

    new_content = re.sub(r'(\\includegraphics(?:\[.*?\])?)\{([^}]+)\}', replace_image, content)
    
    if new_content != content:
        with open(tex_file, 'w') as f:
            f.write(new_content)
        print(f"Updated {tex_file}")

print("Done stripping paths.")
