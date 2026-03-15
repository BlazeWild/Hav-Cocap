import os
import subprocess
from pathlib import Path
from tqdm import tqdm

dataset_dir = Path(r"c:\hav_video_captioning\Hav-Cocap_avcaps\CoCap\dataset\MSVD_yt_videos\MSVD_yt_videos")
in_dirs = [dataset_dir / "videos", dataset_dir / "videos_240_h264_keyint_60"]
out_dir = dataset_dir / "videos_mp4"
out_dir.mkdir(exist_ok=True, parents=True)

avi_files = []
for d in in_dirs:
    if d.exists():
        avi_files.extend(list(d.glob("*.avi")))

# deduplicate by stem
avi_files_dict = {f.stem: f for f in avi_files}
avi_files = list(avi_files_dict.values())

print(f"Found {len(avi_files)} unique avi files.")

import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

for avi in tqdm(avi_files, desc="Converting to MP4"):
    out_file = out_dir / (avi.stem + ".mp4")
    if out_file.exists():
        continue
    
    # Run ffmpeg to convert to h264
    cmd = [
        ffmpeg_path,
        "-i", str(avi),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "faster",
        "-an", # No audio needed for CoCap MSVD
        "-y",
        str(out_file)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Done converting videos.")
