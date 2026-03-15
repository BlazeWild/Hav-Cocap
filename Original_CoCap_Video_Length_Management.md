# Video Length Management in Original CoCap

This document explains how variable video lengths (e.g., a 1-second video vs. a 20-second video) are handled within the *original* CoCap dataloader, specifically through the C++ `cv_reader` bindings and PyTorch sampling steps.

## 1. Subsampling via `sample_frames` (Scaling to duration)

Regardless of whether a video is 1 second or 20 seconds long, `cv_reader` initially returns the entire structure of the video separated into I-frames (GOPs) and their subsequent B/P-frames. To handle varying video lengths, the mechanism scales by the length of the video and dynamically samples frames and GOPs down to fixed numbers configured by `resample_num_gop`, `resample_num_mv`, and `resample_num_res`.

This is handled by the `sample_frames` function (used for both `"rand"` during training and `"uniform"` during testing):
1. The entire video timeline is divided into exactly `N` intervals (using `np.linspace`).
2. A single frame/GOP index is randomly or uniformly chosen from each interval.

Because of this, a 1-second video pulls frames much closer together, while a 20-second video selects sparsely across the video. The number of returning tensors is kept exactly constant.

### Two-Level Sampling

**Level 1: Within each GOP (Motion Vectors and Residuals)**
For every GOP, it extracts a specific number of dependent B/P-frames natively:
```python
# coCap/cocap/data/datasets/compressed_video/video_readers.py
idxs = sample_frames(num_frames=resample_num_mv, vlen=len(full_frame_gop[gop_idx]) - 1, sample="rand")
mv_frame_gop.append([full_frame_gop[gop_idx][i + 1] for i in idxs])
```

**Level 2: The entire Video Timeline (GOP Subsampling)**
Then, it runs the subsampling algorithm again over all available GOPs to enforce bounded dimensions for the whole video:
```python
idxs = sample_frames(num_frames=resample_num_gop, vlen=len(mv_frame_gop), sample=sample)
i_frame_gop = [i_frame_gop[i] for i in idxs]
mv_frame_gop = [mv_frame_gop[i] for i in idxs]
```

## 2. Padding ("pad" Strategy and Handling Edge Cases)

Padding happens in two cases:
1. When the user sets the sample strategy explicitly to `"pad"`.
2. When the original video clip is so short that the extracted components are fewer than the minimum requested limits configuration (`resample_num_gop` or `resample_num_mv`).

### How Padding is Applied
If a video has fewer components than requested, the existing dimensions are scaled to the target limit using a `pad_tensor` function which pads 0s (or 128 for unnormalized uint8 residuals).

```python
if sample == "pad" and iframe.size(0) < resample_num_gop:
    iframe = pad_tensor(iframe, target_size=resample_num_gop, dim=0)

# Similarly padded across motion vector sets inside a GOP:
if sample == "pad" and gop_mv.size(0) < resample_num_mv:
    gop_mv = pad_tensor(gop_mv, target_size=resample_num_mv, dim=0)
```

At the same time, binary masking (`input_mask_gop` and `input_mask_mv`) tracks which temporal steps contain real video data and which steps were injected via padding. In Python slices, `0` corresponds to valid input elements, whilst `1` indicates patched zero-values so that the self-attention matrices in the Transformer correctly ignore them:

```python
input_mask_gop = torch.tensor([0] * iframe.size(0) + [1] * (resample_num_gop - iframe.size(0)), dtype=torch.bool)
# ...
input_mask_mv.append(torch.tensor([0] * gop_mv.size(0) + [1] * (resample_num_mv - gop_mv.size(0)), dtype=torch.bool))
```

## Summary
- **Handling video duration**: `CoCap` relies on mathematically spacing out fixed amounts of frame queries across the entire temporal capacity using the helper function `sample_frames`. Short videos take denser samples; long videos take sparse samples.
- **Is it padded?** By default ("rand" / "uniform"), no padding is applied unless the total number of frames is fewer than the requested configuration dimensions, at which point valid data masks are generated so the model knows what to ignore.