import os
import re

def find_files(root_dir, exts):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if '.git' in dirpath or 'build' in dirpath or 'tests/' in dirpath or '.venv' in dirpath or '.gemini' in dirpath:
            continue
        for f in filenames:
            if any(f.endswith(ext) for ext in exts):
                files.append(os.path.join(dirpath, f))
    return files

def extract_content(latex_files):
    report_lines = ["# AVE Framework - Manuscript Logic & Math Audit Report\n"]
    
    eq_pattern = re.compile(r'\\begin\{equation\}(.*?)\\end\{equation\}', re.DOTALL)
    align_pattern = re.compile(r'\\begin\{align\}(.*?)\\end\{align\}', re.DOTALL)
    section_pattern = re.compile(r'\\(?:chapter|section|subsection)\{([^}]+)\}')
    caption_pattern = re.compile(r'\\caption\{([^}]*)\}')
    
    for lf in latex_files:
        with open(lf, 'r', encoding='utf-8') as f:
            content = f.read()
            
            equations = eq_pattern.findall(content)
            aligns = align_pattern.findall(content)
            sections = section_pattern.findall(content)
            captions = caption_pattern.findall(content)
            
            # Use basic heuristics to check if file actually has mathematical claims
            if not (equations or aligns or sections or captions):
                continue
                
            report_lines.append(f"## {os.path.basename(lf)}")
            report_lines.append(f"**Path**: `{lf}`\n")
            
            if sections:
                report_lines.append("### Sections")
                for sec in sections:
                    report_lines.append(f"- {sec.strip()}")
                report_lines.append("")
                
            if captions:
                report_lines.append("### Figure Captions")
                for cap in captions:
                    report_lines.append(f"- {cap.strip()}")
                report_lines.append("")
                
            if equations or aligns:
                report_lines.append("### Equations")
                for eq in equations:
                    # Clean up random spaces/newlines for the report
                    clean_eq = " ".join(eq.strip().split())
                    report_lines.append(f"$$ {clean_eq} $$")
                for align in aligns:
                    clean_align = " ".join(align.strip().split())
                    report_lines.append(f"$$ {clean_align} $$")
                report_lines.append("")
                
    with open("analytical_audit.md", "w", encoding="utf-8") as out:
        out.write("\n".join(report_lines))
        
    print("Analytical audit report generated: analytical_audit.md")

if __name__ == '__main__':
    latex_files = find_files('manuscript', ['.tex'])
    extract_content(sorted(latex_files))
