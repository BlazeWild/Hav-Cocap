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

# import cv_reader  # Bypassed for Windows native compatibility without C++ compiler
import decord
import lz4.frame
import numpy as np
import torch
import torch.nn.functional
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
        with_residual: bool = False, with_bp_rgb: bool = False, pre_extract: bool = False,
        sample: str = "rand"
) -> Dict[str, np.ndarray]:
    import av
    # Bypass C++ cv_reader block entirely! We use native PyAV for seamless Windows compatibility.
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        frames = []
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                frames.append(frame.to_rgb().to_ndarray())
        
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")

        vlen = len(frames)
        frame_idxs = sample_frames(resample_num_gop, vlen, sample=sample)
        
        iframe = torch.stack([torch.from_numpy(frames[idx]).permute(2, 0, 1) for idx in frame_idxs]).float() / 255.0
        
        if iframe.size(0) < resample_num_gop:
             iframe = pad_tensor(iframe, target_size=resample_num_gop, dim=0)

        # Generates PyTorch Tensors for CoCap model inputs, padding Motion and Residuals with 0s as cv_reader is missing.
        motion_vector = torch.zeros((resample_num_gop, resample_num_mv, 4, 56, 56), dtype=torch.float32)
        input_mask_gop = torch.tensor([0] * iframe.size(0) + [1] * (resample_num_gop - iframe.size(0)), dtype=torch.bool)
        input_mask_mv = torch.ones((resample_num_gop, resample_num_mv), dtype=torch.bool)
        type_ids_mv = torch.zeros((resample_num_gop, resample_num_mv), dtype=torch.long)
        
        ret = {
            "iframe": iframe,
            "motion_vector": motion_vector,
            "input_mask_gop": input_mask_gop,
            "input_mask_mv": input_mask_mv,
            "type_ids_mv": type_ids_mv
        }

        if with_residual:
            # Use uint8 128 (== 0.0 after normalization) to save memory
            residual = torch.full((resample_num_gop, resample_num_res, 3, 224, 224), 128, dtype=torch.uint8)
            input_mask_res = torch.ones((resample_num_gop, resample_num_res), dtype=torch.bool)
            ret["residual"] = residual
            ret["input_mask_res"] = input_mask_res
        
        if with_bp_rgb:
            # We don't have bp_rgb in bypass
            bp_rgb = torch.zeros((resample_num_gop, resample_num_mv, 3, 224, 224), dtype=torch.float32)
            ret["bp_rgb"] = bp_rgb
            
        return ret, True
    except Exception:
        print(f"video load error: {video_path}")
        traceback.print_exc()
        traceback.print_exc(file=open("video_reader_error.log", "a"))
        # create a dummy return data
        ret = {
            "iframe": torch.zeros((resample_num_gop, 3, 224, 224), dtype=torch.float),
            "motion_vector": torch.zeros((resample_num_gop, resample_num_mv, 4, 56, 56), dtype=torch.float),
            "input_mask_gop": torch.ones((resample_num_gop,), dtype=torch.bool),
            "input_mask_mv": torch.ones((resample_num_gop, resample_num_mv), dtype=torch.bool),
            "input_mask_res": torch.ones((resample_num_gop, resample_num_mv), dtype=torch.bool),
            "type_ids_mv": torch.zeros((resample_num_gop, resample_num_mv), dtype=torch.long)
        }
        if with_residual:
            ret["residual"] = torch.full((resample_num_gop, resample_num_res, 3, 224, 224), 128, dtype=torch.uint8)
        if with_bp_rgb:
            ret["bp_rgb"] = torch.zeros((resample_num_gop, resample_num_mv, 3, 224, 224), dtype=torch.float)
        return ret, False


def get_video_len(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen
