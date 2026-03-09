import os
from huggingface_hub import snapshot_download

repo_id = "Blazewild/MSVD_yt_videos"
local_dir = r"c:\hav_video_captioning\Hav-Cocap_avcaps\CoCap\dataset"

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    resume_download=True,
    max_workers=8
)
print("Download complete.")
