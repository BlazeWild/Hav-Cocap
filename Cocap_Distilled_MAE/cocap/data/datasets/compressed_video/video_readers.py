# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 14:56
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : video_readers.py

# based on https://github.com/m-bain/frozen-in-time/blob/main/base/base_dataset.py


import logging
import pickle
import random
import subprocess
import traceback
from typing import Dict

import cv_reader
import decord
import lz4.frame
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
from fvcore.common.registry import Registry

from cocap.utils.profile import Timer
from .compressed_video_utils import deserialize

logger = logging.getLogger(__name__)

VIDEO_READER_REGISTRY = Registry("VIDEO_READER")


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    # acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=num_frames + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) if x[0] != x[1] else x[0] for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


@VIDEO_READER_REGISTRY.register()
def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    import cv2

    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs


@VIDEO_READER_REGISTRY.register()
def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    import av

    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


@VIDEO_READER_REGISTRY.register()
def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_video_size(video_path):
    import cv2

    vcap = cv2.VideoCapture(video_path)  # 0=camera
    if vcap.isOpened():
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        return int(width), int(height)
    else:
        raise RuntimeError(f"VideoCapture cannot open video file: {video_path}")


def get_frame_type(video_path):
    command = '/usr/bin/ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_path]).decode()
    frame_types = out.replace('pict_type=', '').split()
    return frame_types


def pad_tensor(tensor: torch.Tensor, target_size: int, dim: int, pad_value=0):
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_size - pad_shape[dim]
    return torch.concat([tensor, torch.full(pad_shape, device=tensor.device, dtype=tensor.dtype, fill_value=pad_value)],
                        dim=dim)


@VIDEO_READER_REGISTRY.register()
def read_frames_compressed_domain(
        video_path: str,
        resample_num_gop: int, resample_num_mv: int, resample_num_res: int,
        with_residual: bool = False,
        with_bp_rgb: bool = False,
        pre_extract: bool = False,
        sample: str = "pad"
) -> Dict[str, torch.Tensor]:
    """
    Sub-GOP Slicing Architecture optimized for L4 GPU processing.
    Masking Convention: 0 = Valid Data, 1 = Padding/Ignore.
    """
    decord.bridge.set_bridge("torch")
    assert sample in {"rand", "uniform", "pad"}
    try:
        timer = Timer()
        reader = decord.VideoReader(video_path, num_threads=1)
        timer("check_video_length")
        
        # ==========================================================
        # 1. LOAD DATA (Raw or Pre-extracted)
        # ==========================================================
        if not pre_extract:
            reader_ret = cv_reader.read_video(video_path)
            timer("cv_reader")
        else:
            data = {}
            read_type = ["pict_type", "rgb_gop", "motion_vector"]
            for t in read_type:
                if t == 'motion_vector':
                    with lz4.frame.open(f"{video_path}.{t}", "rb") as f:
                        data.update(pickle.load(f))
                else:
                    with open(f"{video_path}.{t}", "rb") as f:
                        data.update(pickle.load(f))
            timer("read")
            data = deserialize(data)
            timer("deserialize")
            reader_ret = [{} for _ in range(len(data["pict_type"]))]
            for k, v_list in data.items():
                if k == "rgb_gop":
                    idx_iframe = 0
                    for i, t in enumerate(data["pict_type"]):
                        if t == "I":
                            reader_ret[i]["rgb"] = v_list[idx_iframe]
                            idx_iframe += 1
                    assert idx_iframe == len(v_list)
                else:
                    for i, v in enumerate(v_list):
                        reader_ret[i][k] = v
            timer("format")

        # ==========================================================
        # 2. SUB-GOP SLICER (1 Anchor + up to `resample_num_mv` P-frames)
        # ==========================================================
        # First, group into natural continuous GOPs (I -> P P P...)
        natural_gops = []
        for f in reader_ret:
            if f["pict_type"] == "I":
                natural_gops.append([f])
            elif f["pict_type"] == "P" and len(natural_gops) > 0:
                natural_gops[-1].append(f)
                
        natural_gops = [g for g in natural_gops if len(g) > 2]

        # Slicer: Break long GOPs into max length chunks
        all_sub_gops = []
        for gop in natural_gops:
            total_frames = len(gop)
            for start_idx in range(0, total_frames, resample_num_mv + 1):
                chunk = gop[start_idx : start_idx + resample_num_mv + 1]
                if len(chunk) < 2: continue # Skip fragments with no motion
                all_sub_gops.append(chunk)

        # Safety Fallback
        if len(all_sub_gops) == 0:
            raise RuntimeError("No valid motion found in video.")

        # ==========================================================
        # 3. UNIFORM SAMPLING (np.linspace) & GOP MASKING
        # ==========================================================
        total_valid_chunks = len(all_sub_gops)
        sampled_chunks = []
        
        # input_mask_gop: 0 = Valid, 1 = Padding
        input_mask_gop = torch.ones(resample_num_gop, dtype=torch.bool)
        
        if total_valid_chunks >= resample_num_gop:
            gop_idxs = np.linspace(0, total_valid_chunks - 1, resample_num_gop).astype(int)
            sampled_chunks = [all_sub_gops[i] for i in gop_idxs]
            input_mask_gop[:] = 0 # All 8 are valid
        else:
            sampled_chunks = all_sub_gops.copy()
            input_mask_gop[:total_valid_chunks] = 0 # Mark valid ones as 0
            # Pad the rest with the last valid chunk just to maintain tensor shapes
            while len(sampled_chunks) < resample_num_gop:
                sampled_chunks.append(sampled_chunks[-1])
                
        timer("sample")

        # ==========================================================
        # 4. EXTRACT RGB ANCHORS (iframe)
        # ==========================================================
        if pre_extract:
            iframe = [chunk[0]["rgb"] for chunk in sampled_chunks]
            iframe = torch.stack([torch.from_numpy(f) for f in iframe]).permute(0, 3, 1, 2).float() / 255.0
        else:
            iframe_idx = [chunk[0]["frame_idx"] for chunk in sampled_chunks]
            iframe = reader.get_batch(iframe_idx).permute(0, 3, 1, 2).float() / 255.0
            
        timer("stack_iframe")

        # ==========================================================
        # 5. BUILD MOTION TENSORS & FRAME MASKS
        # ==========================================================
        motion_vector_list = []
        input_mask_mv_list = []
        type_ids_mv_list = []

        # Find a valid MV to get spatial dimensions (usually 4xHxW)
        sample_mv_shape = None
        for chunk in sampled_chunks:
            if len(chunk) > 1 and "motion_vector" in chunk[1]:
                sample_mv_shape = chunk[1]["motion_vector"].shape
                break

        for i, chunk in enumerate(sampled_chunks):
            mvs = chunk[1:]
            is_ghost_gop = input_mask_gop[i].item() # True(1) if padding GOP
            
            # mask: 0 = valid, 1 = padding
            mask_mv = torch.ones(resample_num_mv, dtype=torch.bool)
            type_ids = torch.full((resample_num_mv,), 2, dtype=torch.long)
            
            if is_ghost_gop or len(mvs) == 0:
                mv_tensor = torch.zeros((resample_num_mv, sample_mv_shape[2], sample_mv_shape[0], sample_mv_shape[1]))
            else:
                num_actual = len(mvs)
                # Convert list of arrays [H, W, 4] to tensor [MV, 4, H, W]
                raw_mvs = [f["motion_vector"].transpose((2, 0, 1)).astype(np.float32) for f in mvs]
                mv_tensor_valid = torch.from_numpy(np.stack(raw_mvs))
                
                mv_tensor = pad_tensor(mv_tensor_valid, target_size=resample_num_mv, dim=0)
                mask_mv[:num_actual] = 0 # Mark actual MVs as valid (0)
                type_ids[:num_actual] = 0 # 0 = P-frame
                
            motion_vector_list.append(mv_tensor)
            input_mask_mv_list.append(mask_mv)
            type_ids_mv_list.append(type_ids)

        # motion_vector shape: [GOP, MV, 4, H, W]
        motion_vector = torch.stack(motion_vector_list)
        input_mask_mv = torch.stack(input_mask_mv_list)
        type_ids_mv = torch.stack(type_ids_mv_list)
        timer("stack_motion")

        # ==========================================================
        # 6. 4-TO-2 CHANNEL CONVERSION & PHYSICS MASKING
        # ==========================================================
        dx = motion_vector[:, :, 2, :, :] - motion_vector[:, :, 0, :, :]
        dy = motion_vector[:, :, 3, :, :] - motion_vector[:, :, 1, :, :]
        motion_vector_2d = torch.stack([dx, dy], dim=2) # [8, 29, 2, H, W]
        
        # Filter crazy spikes (scene cuts within a sub-GOP) or frozen frames
        mag = motion_vector_2d.abs().mean(dim=(2, 3, 4), keepdim=True)
        valid_physics_mask = ((mag > 0.01) & (mag < 40.0)).float()
        motion_vector_2d = motion_vector_2d * valid_physics_mask

        # ==========================================================
        # 7. DISTILLED MAE TARGET: LAST P-FRAME EXTRACTION
        # ==========================================================
        last_p_frames = []
        for i, chunk in enumerate(sampled_chunks):
            if input_mask_gop[i].item() == 1:
                # Ghost GOP, just use the anchor as a dummy target
                last_p_frames.append(iframe[i])
                continue
                
            mvs = chunk[1:]
            num_actual = len(mvs)
            gop_mv = motion_vector_2d[i, :num_actual]
            
            mag_per_frame = gop_mv.abs().mean(dim=(1, 2, 3))
            nonzero_mask = mag_per_frame > 0.01
            
            if nonzero_mask.sum() > 0:
                mean_mag = mag_per_frame[nonzero_mask].mean().item()
                std_mag  = mag_per_frame[nonzero_mask].std().item() if nonzero_mask.sum() > 1 else 0.0
            else:
                mean_mag, std_mag = 0.0, 0.0
                
            dynamic_threshold = max(mean_mag + (3 * std_mag), 10.0)
            
            # Step backward to find the final frame with valid motion
            last_valid_idx = None
            for j in range(num_actual - 1, -1, -1):
                mag_val = mag_per_frame[j].item()
                if mag_val > dynamic_threshold: continue
                if mag_val < (mean_mag * 0.05) and j > 3: continue
                last_valid_idx = j
                break
                
            if last_valid_idx is not None:
                p_frame_data = mvs[last_valid_idx]
                if "frame_idx" in p_frame_data:
                    frame_tensor = reader.get_batch([p_frame_data["frame_idx"]])
                    last_p_rgb = frame_tensor[0].permute(2, 0, 1).float() / 255.0
                elif "rgb" in p_frame_data:
                    last_p_rgb = torch.from_numpy(p_frame_data["rgb"]).permute(2, 0, 1).float() / 255.0
                else:
                    last_p_rgb = iframe[i]
            else:
                last_p_rgb = iframe[i]
                
            last_p_frames.append(last_p_rgb)
            
        last_p_frames = torch.stack(last_p_frames)

        # ==========================================================
        # 8. BUILD FINAL RETURN DICTIONARY (ret)
        # ==========================================================
        ret = {
            "iframe": iframe,                               # [8, 3, 224, 224]
            "motion_vector": motion_vector_2d,              # [8, 29, 2, 56, 56]
            "last_p_frame": last_p_frames,                  # [8, 3, 224, 224] target for Distilled MAE
            "input_mask_gop": input_mask_gop,               # [8] (0=Valid, 1=Padding)
            "input_mask_mv": input_mask_mv,                 # [8, 29] (0=Valid, 1=Padding)
            "type_ids_mv": type_ids_mv                      # [8, 29] (0=P-frame, 2=Padding)
        }

        # Optional Returns
        if with_residual:
            residual_list = []
            input_mask_res_list = []
            for i, chunk in enumerate(sampled_chunks):
                mvs = chunk[1:]
                mask_res = torch.ones(resample_num_res, dtype=torch.bool)
                if input_mask_gop[i].item() == 1 or len(mvs) == 0:
                    res_tensor = torch.zeros((resample_num_res, 3, sample_mv_shape[0], sample_mv_shape[1]))
                else:
                    num_actual = len(mvs)
                    raw_res = [f["residual"].transpose((2, 0, 1)) for f in mvs]
                    res_tensor_valid = torch.from_numpy(np.stack(raw_res))
                    res_tensor = pad_tensor(res_tensor_valid, target_size=resample_num_res, dim=0)
                    mask_res[:num_actual] = 0
                residual_list.append(res_tensor)
                input_mask_res_list.append(mask_res)
                
            ret["residual"] = torch.stack(residual_list)
            ret["input_mask_res"] = torch.stack(input_mask_res_list)

        if with_bp_rgb:
            bp_rgb_list = []
            for i, chunk in enumerate(sampled_chunks):
                mvs = chunk[1:]
                if input_mask_gop[i].item() == 1 or len(mvs) == 0:
                    gop_rgb = torch.zeros((resample_num_mv, 3, 224, 224)) # Adjust dims if needed
                else:
                    gop_rgb_frames = []
                    for f in mvs:
                        if "rgb" in f:
                            gop_rgb_frames.append(torch.from_numpy(f["rgb"]).permute(2, 0, 1).float() / 255.0)
                        else:
                            frame_tensor = reader.get_batch([f["frame_idx"]])[0]
                            gop_rgb_frames.append(frame_tensor.permute(2, 0, 1).float() / 255.0)
                    gop_rgb_valid = torch.stack(gop_rgb_frames)
                    gop_rgb = pad_tensor(gop_rgb_valid, target_size=resample_num_mv, dim=0, pad_value=0)
                bp_rgb_list.append(gop_rgb)
            ret["bp_rgb"] = torch.stack(bp_rgb_list)

        logger.debug(timer.get_info(averaged=False))
        return ret, True

    except Exception as e:
        logger.exception(f"video read failed for {video_path}: {type(e).__name__}: {e}")
        raise