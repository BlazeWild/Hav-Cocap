# GOP Dataloader

This project implements a video dataloader that segments videos into **Group of Pictures (GOP)** aligned chunks. It extracts video frames and the corresponding audio segment for each GOP.

## Setup
1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install av numpy
   ```

## Usage

### Using the Dataloader in Python
```python
from dataloader import GOPDataloader

loader = GOPDataloader("path/to/video.mp4")

for gop in loader.process():
    print(f"GOP {gop['gop_index']} [{gop['start_time']}s - {gop['end_time']}s]")
    # Access data:
    # gop['video_frames'] (List of av.VideoFrame)
    # gop['audio_samples'] (numpy array)
```

### Running the Verification Script
To test the dataloader and verify GOP boundaries on a video file, run the `test_dataloader.py` script from the terminal.

**Command:**
```bash
python test_dataloader.py <path_to_video>
```

**Example:**
```bash
python test_dataloader.py val_videos/4963357278.mp4
```

This script will:
1. Print GOP information (timestamps, frame counts, audio duration).
2. **Save debug files** for the first 3 GOPs in the `debug_output/` directory:
   - `gop_0_video.mp4`, `gop_0_audio.wav`
   - `gop_1_video.mp4`, `gop_1_audio.wav`
   - ...
