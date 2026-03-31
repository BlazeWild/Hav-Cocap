import os
from havcocap_new.data.datasets.compressed_video.video_readers import read_frames_compressed_domain

video_path = "./dataset/Charades/Charades_filtered_240/127G2.mp4"
if not os.path.exists(video_path):
    # Find any mp4 to test
    for file in os.listdir("./dataset/Charades/Charades_filtered_240"):
        if file.endswith(".mp4"):
            video_path = os.path.join("./dataset/Charades/Charades_filtered_240", file)
            break

print(f"Reading video: {video_path}")
try:
    video, mask = read_frames_compressed_domain(
        video_path,
        resample_num_gop=16,
        resample_num_mv=59,
        resample_num_res=59,
        with_residual=True,
        pre_extract=False,
        sample="rand"
    )
    print("Success reading video!")
    print(f"I-frames shape: {video[0].shape}, dtype: {video[0].dtype}")
except Exception as e:
    print(f"Error: {e}")
