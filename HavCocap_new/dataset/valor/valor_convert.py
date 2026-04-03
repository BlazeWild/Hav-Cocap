import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- DYNAMIC PATH RESOLUTION ---
# This finds the absolute path of the directory where this script lives
SCRIPT_DIR = Path(__file__).parent.absolute()
# Assuming 'dataset' is in the project root (one level up from 'tools')
PROJECT_ROOT = SCRIPT_DIR.parent

# --- CONFIGURATION ---
INPUT_DIR = PROJECT_ROOT / "dataset" / "valor" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "valor_mp4_240p_30fps_keyint=30"

# Optimized for H100/A100. You can push this to 32 if your CPU/Disk I/O is strong.
NUM_WORKERS = 16 

def process_video(video_path):
    video_path = Path(video_path)
    # Maintain the internal folder structure if the raw videos are in subfolders
    relative_path = video_path.relative_to(INPUT_DIR)
    target_path = OUTPUT_DIR / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # --- THE RESIZE LOGIC ---
    # Short-edge 240p filter: Preserves aspect ratio.
    # If landscape: Height=240, Width=Auto (even). If portrait: Width=240, Height=Auto.
    scale_filter = "scale='if(gt(a,1),-2,240):if(gt(a,1),240,-2)'"
    
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",             # GPU Decoding
        "-hwaccel_output_format", "cuda",
        "-i", str(video_path),
        "-r", "30",                     # 30 FPS
        "-vf", scale_filter,            # 240p Short-edge Resize
        "-c:v", "h264_nvenc",           # NVIDIA GPU Encoding
        "-g", "30",                     # GOP Size (keyint) = 30
        "-bf", "0",                     # No B-frames for clean Motion Vectors
        "-forced-idr", "1",             # Every I-frame is an IDR frame
        "-sc_threshold", "0",           # Strict GOP boundary (no early I-frames)
        "-c:a", "copy",                 # Leave audio untouched for 1:1 sync
        "-preset", "p4",                # Balanced speed/quality for H100
        "-tune", "hq",
        str(target_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # If a video fails, log it so you can inspect it later
        with open("encoding_errors.log", "a") as f:
            f.write(f"Error on {video_path}: {e.stderr.decode()}\n")

def main():
    # Create the output directory relative to the project root
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    print(f"🔥 HPC Mode Initialized (H100/A100)")
    print(f"📂 Input:  {INPUT_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    # Supported video formats
    video_extensions = [".mp4", ".mkv", ".avi", ".webm"]
    all_videos = [
        p for p in INPUT_DIR.rglob("*") 
        if p.suffix.lower() in video_extensions
    ]
    
    if not all_videos:
        print(f"❌ No videos found in {INPUT_DIR}. Check your paths!")
        return

    print(f"✅ Found {len(all_videos)} videos. Launching {NUM_WORKERS} parallel encoders...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_video, all_videos), total=len(all_videos)))

    print("\n🎉 Preprocessing Complete. Your dataset is now HavCocap ready!")

if __name__ == "__main__":
    main()