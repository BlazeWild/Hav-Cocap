import os
from havcocap_new.data.datasets.compressed_video.video_readers import read_frames_compressed_domain

video_path = "./dataset/Charades/Charades_filtered_240/8WJIR.mp4"
print(f"Reading video: {video_path}")
video, mask = read_frames_compressed_domain(
    video_path,
    resample_num_gop=16,
    resample_num_mv=59,
    resample_num_res=59,
    with_residual=True,
    pre_extract=False,
    sample="rand"
)
print("Keys in video:", video.keys() if isinstance(video, dict) else type(video))
for k, v in video.items() if isinstance(video, dict) else enumerate([]):
    print(f"Key {k}, shape: {v.shape}")
