# Video Length Management in CoCap DataLoader

This document explains how variable video lengths (e.g., a 1-second video vs. a 20-second video) are handled within the CoCap video dataloader.

## 1. Selected Number of GOPs / Frames is Taken (Subsampling)

Regardless of whether a video is 1 second or 20 seconds long, the model does **not** process every single frame or GOP. Instead, it scales by the video's total duration and **takes a selected number of GOPs/frames**.

This behavior is controlled by the `sample_frames` function. The process is as follows:
1. The total duration (number of frames or GOPs) of the video is calculated as `vlen`.
2. The entire video timeline is divided into exactly `num_frames` (or `num_gop` from the config, usually 8) equal intervals.
3. From each interval, **one frame (or GOP index)** is selected either randomly (during training) or uniformly from the center (during testing).

This means a 1-second video (e.g., 30 frames) might have intervals `[0-3, 3-7, 7-10...]` and extracts 8 frames close to each other. A 20-second video (e.g., 600 frames) has wide intervals `[0-75, 75-150...]` and also extracts exactly 8 frames sparsely distributed across the full length.

### *Where this is done:*
**File:** `CoCap/cocap/data/datasets/compressed_video/video_readers.py`
**Function:** `sample_frames(num_frames, vlen, sample='rand', fix_start=None)`

```python
def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    # Divides the total video length into `num_frames` chunks
    intervals = np.linspace(start=0, stop=vlen, num=num_frames + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))
        
    # Picks 1 frame per chunk (either randomly or uniformly)
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) if x[0] != x[1] else x[0] for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    # ...
    return frame_idxs
```

## 2. Padding (Handling Extremely Short Videos)

Padding is mostly used as a fallback. Since the `sample_frames` method inherently limits the output length to exactly `num_frames`, dynamic truncation isn't necessary. However, padding takes place when a video is so short or corrupted that the decoding loop extracts *fewer* frames than the configured `resample_num_gop`. 

If the loaded tensors fall short of the required dimension (e.g., required 8 GOPs, but only loaded 6), the dataloader uses zero-padding on the missing GOPs and creates masks that tell the Transformer to ignore the padded regions.

### *Where this is done:*
**File:** `CoCap/cocap/data/datasets/compressed_video/video_readers.py`
**Function:** `read_frames_compressed_domain`

```python
        # The number of extracted frames/GOPs is compared against the target size
        if iframe.size(0) < resample_num_gop:
             # Tensor is zero-padded up to `resample_num_gop`
             iframe = pad_tensor(iframe, target_size=resample_num_gop, dim=0)

        # Padding masks are created for the model to ignore the empty space
        # 0 means actual data, 1 means padded/masked out
        input_mask_gop = torch.tensor([0] * iframe.size(0) + [1] * (resample_num_gop - iframe.size(0)), dtype=torch.bool)
```

## Summary
- **Is it padded or just a selected number of GOPs?** Primarily, a **selected, fixed number of GOPs** is sampled uniformly from the video timeline regardless of how long the video is. This ensures constant model input size.
- **When does padding occur?** Padding only occurs when the video is extremely short (fewer frames than the required number of GOPs/frames) or fails to provide enough data during PyAV/CV extraction.