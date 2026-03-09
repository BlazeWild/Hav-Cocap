import json
import os

BLACKLIST_FILE = "Hav-Cocap/corrupt_files.json"
DATASET_ROOT_REL = "Hav-Cocap" # Prefix to add if missing

def fix_paths():
    if not os.path.exists(BLACKLIST_FILE):
        print(f"{BLACKLIST_FILE} not found.")
        return

    with open(BLACKLIST_FILE, 'r') as f:
        data = json.load(f)

    fixed_data = []
    print(f"Processing {len(data)} entries...")
    
    current_cwd = os.getcwd() # c:\hav_video_captioning\Hav-Cocap_avcaps
    
    for path in data:
        # Normalize slashes
        p = os.path.normpath(path)
        
        # Check if it's already absolute
        if os.path.isabs(p):
            fixed_data.append(p)
            continue
            
        # If it starts with dataset\, it's the old relative path
        if p.startswith("dataset"):
            # The actual files are in Hav-Cocap/dataset/...
            new_rel = os.path.join("Hav-Cocap", p)
            abs_p = os.path.abspath(new_rel)
            fixed_data.append(abs_p)
        elif p.startswith("Hav-Cocap"):
             # Already has prefix, make absolute
             abs_p = os.path.abspath(p)
             fixed_data.append(abs_p)
        else:
             # Unknown, just make absolute
             fixed_data.append(os.path.abspath(p))

    # Remove duplicates
    fixed_data = sorted(list(set(fixed_data)))
    
    with open(BLACKLIST_FILE, 'w') as f:
        json.dump(fixed_data, f, indent=4)
        
    print(f"Fixed {len(fixed_data)} entries. Saved to {BLACKLIST_FILE}")

if __name__ == "__main__":
    fix_paths()
