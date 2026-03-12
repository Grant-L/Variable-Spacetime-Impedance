#!/usr/bin/env python3
import os
import subprocess
import random

APP_DIR = "/root/projects/ave_alpha_search"

def generate_workunit(wu_idx, seed, chunk_size=500000, box_size=100.0):
    wu_name = f"alpha_chunk_{wu_idx:05d}"
    input_filename = f"input_{wu_idx:05d}.txt"
    
    payload = f"SEED={seed}\nCHUNK_SIZE={chunk_size}\nBOUNDING_BOX={box_size}\n"
    
    try:
        res = subprocess.run([os.path.join(APP_DIR, "bin", "dir_hier_path"), input_filename], 
                             capture_output=True, text=True, check=True, cwd=APP_DIR)
        input_path = res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting dir_hier_path: {e}")
        return False
        
    try:
        with open(input_path, 'w') as f:
            f.write(payload)
    except IOError as e:
        print(f"Error writing to {input_path}: {e}")
        return False
        
    try:
        subprocess.run([
            os.path.join(APP_DIR, "bin", "create_work"),
            "-appname", "ave_alpha",
            "-wu_name", wu_name,
            "-wu_template", "templates/ave_alpha_in.xml",
            "-result_template", "templates/ave_alpha_out.xml",
            input_filename
        ], check=True, cwd=APP_DIR, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating workunit {wu_name}: {e.stderr}")
        return False
        
    return True

def main():
    print("==> Initializing Monte Carlo Alpha Generation...")
    os.chdir(APP_DIR)
    
    num_chunks = 1000
    print(f"==> Generating {num_chunks} Workunits (This bridges 500,000,000 spatial topologies)...")
    
    success_count = 0
    for i in range(1, num_chunks + 1):
        # Generate a random 32-bit integer seed for C++ mt19937
        seed = random.randint(0, 4294967295)
        if generate_workunit(i, seed):
            success_count += 1
            if success_count % 100 == 0:
                print(f"    -> Queued {success_count}/{num_chunks} tasks...")
                
    print(f"==> Successfully enqueued {success_count} independent topological matrices into the BOINC array.")

if __name__ == "__main__":
    main()
