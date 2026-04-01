import numpy as np
import decord
decord.bridge.set_bridge("torch")

def read_video(video_path):
    reader = decord.VideoReader(video_path, num_threads=1)
    num_frames = len(reader)
    
    # We create a dummy GOP structure
    pict_types = ["I"] + ["P"] * (num_frames - 1)
    
    first_frame = reader[0]
    H, W = first_frame.shape[0], first_frame.shape[1]
    
    reader_ret = []
    for i in range(num_frames):
        frame_data = {"pict_type": pict_types[i], "frame_idx": i}
        
        if pict_types[i] == "I":
            frame_data["rgb"] = first_frame.numpy()
        else:
            frame_data["motion_vector"] = np.zeros((H, W, 4), dtype=np.float32)
            frame_data["residual"] = np.zeros((H, W, 3), dtype=np.float32)
            frame_data["rgb"] = np.zeros((H, W, 3), dtype=np.uint8)
            
        reader_ret.append(frame_data)
        
    return reader_ret
