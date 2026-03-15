import sys
import os

sys.path.append(r"c:\hav_video_captioning\Hav-Cocap_avcaps\CoCap")
from cocap.data.datasets.compressed_video.video_readers import read_frames_compressed_domain

vid = r"c:\hav_video_captioning\Hav-Cocap_avcaps\CoCap\dataset\msvd\videos_240_h264_keyint_60\-4wsuPCjDBc_5_15.avi"
print(f"Testing reading from {vid}")
res, success = read_frames_compressed_domain(
    video_path=vid,
    resample_num_gop=8,
    resample_num_mv=59,
    resample_num_res=59,
    with_residual=True,
    with_bp_rgb=False,
    pre_extract=False,
    sample="rand"
)
print("SUCCESS:", success)
if success:
    print("I-Frame shape:", res["iframe"].shape)
