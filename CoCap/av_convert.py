import av
import os
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

for avi in tqdm(avi_files, desc="Converting to MP4"):
    out_file = out_dir / (avi.stem + ".mp4")
    if out_file.exists():
        continue
    
    try:
        # Open input container
        input_container = av.open(str(avi))
        in_video = input_container.streams.video[0]
        
        # Open output container
        output_container = av.open(str(out_file), mode='w')
        
        # Setup output stream matching input frame rate
        fps = in_video.average_rate
        if not fps or fps <= 0:
            fps = 30
            
        out_stream = output_container.add_stream('h264', rate=int(fps))
        out_stream.width = in_video.width
        out_stream.height = in_video.height
        out_stream.pix_fmt = 'yuv420p'
        
        # Decode and encode frames
        for frame in input_container.decode(in_video):
            for packet in out_stream.encode(frame):
                output_container.mux(packet)
                
        # Flush the stream
        for packet in out_stream.encode():
            output_container.mux(packet)
            
        input_container.close()
        output_container.close()
        
    except Exception as e:
        print(f"\nError converting {avi.name}: {e}")

print("Done converting videos.")
