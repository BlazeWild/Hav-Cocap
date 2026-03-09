
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from dataloader import GOPDataloader

def test_dataloader(video_path):
    print(f"Testing GOPDataloader with {video_path}")
    loader = GOPDataloader(video_path)
    
    gops = []
    
    for gop in loader.process():
        print(f"GOP {gop['gop_index']}:")
        print(f"  Time Range: {gop['start_time']:.4f}s - {gop['end_time']:.4f}s (Duration: {gop['end_time'] - gop['start_time']:.4f}s)")
        print(f"  Video Frames: {len(gop['video_frames'])}")
        print(f"  Audio Samples Shape: {gop['audio_samples'].shape} @ {gop['sample_rate']}Hz")
        
        # Verify audio duration roughly matches video GOP duration
        audio_duration = gop['audio_samples'].shape[-1] / gop['sample_rate'] # Assuming last dim is samples
        print(f"  Audio Duration: {audio_duration:.4f}s")
        
        gops.append(gop)
        
        # Save first 3 GOPs for visual/audio verification
        if gop['gop_index'] < 3:
            save_gop_debug(gop, "debug_output")
            
    print(f"\nTotal GOPs found: {len(gops)}")

def save_gop_debug(gop, output_dir):
    import os
    import wave
    import av
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    idx = gop['gop_index']
    
    # 1. Save Audio (WAV)
    # Audio samples are (channels, samples) or (samples, channels). 
    # My code in dataloader.py yielded whatever came out of concatenation.
    # Let's inspect shape in the test output: (2, 44144) -> This is (channels, samples).
    # wave module expects bytes. We need to convert numpy to int16 bytes usually.
    # PyAV usually gives float32 planar.
    
    audio_data = gop['audio_samples']
    sample_rate = gop['sample_rate']
    
    audio_path = os.path.join(output_dir, f"gop_{idx}_audio.wav")
    
    # Transpose to (samples, channels) for typical processing if needed, 
    # but let's check what we have. (2, N)
    if audio_data.ndim > 1 and audio_data.shape[0] < 10:
        audio_data = audio_data.T # details: (2, N) -> (N, 2)
    
    # Normalize and convert to int16 if float
    if audio_data.dtype.kind == 'f':
        audio_data = (audio_data * 32767).astype(np.int16)
        
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(audio_data.shape[1] if audio_data.ndim > 1 else 1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
        
    print(f"  Saved Audio: {audio_path}")
    
    # 2. Save Video (MP4) - strictly the frames we collected
    # We will use PyAV to write these frames to a new container
    video_path = os.path.join(output_dir, f"gop_{idx}_video.mp4")
    
    try:
        # We need to create a container
        output = av.open(video_path, 'w')
        video_stream = output.add_stream('libx264', rate=30) # Assuming 30fps, ideally we get this from source frames/stream
        # We can guess width/height from first frame
        if gop['video_frames']:
            first_frame = gop['video_frames'][0]
            video_stream.width = first_frame.width
            video_stream.height = first_frame.height
            video_stream.pix_fmt = 'yuv420p'
            
            for frame in gop['video_frames']:
                # We need to re-encode the frames.
                # frame is an av.VideoFrame.
                packet = video_stream.encode(frame)
                output.mux(packet)
            
            # Flush
            packet = video_stream.encode(None)
            output.mux(packet)
            
        output.close()
        print(f"  Saved Video: {video_path}")
    except Exception as e:
        print(f"  Failed to save video: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_dataloader.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    test_dataloader(video_path)
