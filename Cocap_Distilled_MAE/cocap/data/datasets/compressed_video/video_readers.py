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
        pre_extract: bool = False,
        sample: str = "rand"
) -> Dict[str, np.ndarray]:
    """
    This function process the output of `cv_reader` to obtain the inputs for training
    :param video_path: path to the video
    :param resample_num_gop: number of GOP sampled from the video
    :param resample_num_mv: number of motion vectors sampled from each GOP
    :param resample_num_res: number of residuals sampled from each GOP
    :param with_residual: whether to return residual
    :param with_bp_rgb: also return the decoded RGB frames of the video
    :param pre_extract: use pre-extracted data
    :param sample: sample method
    :return:
    """
    decord.bridge.set_bridge("torch")
    assert sample in {"rand", "uniform", "pad"}
    try:
        timer = Timer()
        reader = decord.VideoReader(video_path, num_threads=1)
        timer("check_video_length")
        # load data from video file/pre-extracted feature
        # load data from video file/pre-extracted feature
        if not pre_extract:
            reader_ret = cv_reader.read_video(video_path)
            timer("cv_reader")
        else:
            data = {}
            # We strictly only need the I-frame RGB ("rgb_gop") and the Motion Vectors
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
        full_frame_gop = []
        for f in reader_ret:
            if f["pict_type"] == "I":
                full_frame_gop.append([f, ])
            elif f["pict_type"] == "P":  # Only keep P frames, drop B frames
                if len(full_frame_gop) > 0:
                    full_frame_gop[-1].append(f)
        full_frame_gop = [g for g in full_frame_gop if len(g) > 2]  # remove gop which do not contain any P-frame

        # Ensure we only take the consecutive P frames immediately after the I frame
        for i in range(len(full_frame_gop)):
            # Index 0 is I-frame, followed by P-frames. Keep up to 'resample_num_mv' P-frames + 1 I-frame
            full_frame_gop[i] = full_frame_gop[i][:1 + resample_num_mv]

        valid_i_frame_gop = []
        valid_mv_frame_gop = []
        valid_res_frame_gop = []

        for gop_idx in range(len(full_frame_gop)):
            # Safely extract up to resample_num_mv P-frames
            raw_mvs = [f["motion_vector"] for f in full_frame_gop[gop_idx][1 : 1 + resample_num_mv]]
            
            if len(raw_mvs) > 0:
                # Fast magnitude check (Ignore static/dead GOPs like title cards)
                gop_mv_tensor = torch.from_numpy(np.stack(raw_mvs).astype(np.float32))
                magnitude = gop_mv_tensor.abs().mean()
                
                # Only keep the GOP if it has actual motion (> 1e-3)
                if magnitude > 1e-3:
                    valid_i_frame_gop.append(full_frame_gop[gop_idx][0])
                    valid_mv_frame_gop.append(full_frame_gop[gop_idx][1 : 1 + resample_num_mv])
                    valid_res_frame_gop.append(full_frame_gop[gop_idx][1 : 1 + resample_num_res])

        # SAFETY NET: If the ENTIRE video is static, fallback to raw GOPs
        if len(valid_mv_frame_gop) == 0:
            valid_i_frame_gop = [g[0] for g in full_frame_gop]
            valid_mv_frame_gop = [g[1 : 1 + resample_num_mv] for g in full_frame_gop]
            valid_res_frame_gop = [g[1 : 1 + resample_num_res] for g in full_frame_gop]

        # ==========================================================
        # 2. ENFORCE NP.LINSPACE FOR GOP SAMPLING (On Valid GOPs Only)
        # ==========================================================
        total_valid_gops = len(valid_mv_frame_gop)
        
        if total_valid_gops >= resample_num_gop:
            # Uniformly sample across the active parts of the video
            gop_idxs = np.linspace(0, total_valid_gops - 1, resample_num_gop).astype(int)
            i_frame_gop = [valid_i_frame_gop[i] for i in gop_idxs]
            mv_frame_gop = [valid_mv_frame_gop[i] for i in gop_idxs]
            res_frame_gop = [valid_res_frame_gop[i] for i in gop_idxs]
        else:
            # Fewer valid GOPs than requested: take them all, rely on sample="pad" later
            i_frame_gop = valid_i_frame_gop
            mv_frame_gop = valid_mv_frame_gop
            res_frame_gop = valid_res_frame_gop
            
        timer("sample")

        # stack iframe
        if with_bp_rgb or pre_extract:
            iframe = [cur_gop["rgb"] for cur_gop in i_frame_gop]
            iframe = torch.stack([torch.from_numpy(f) for f in iframe]).permute(0, 3, 1, 2) / 255
        else:
            iframe_idx = [cur_gop["frame_idx"] for cur_gop in i_frame_gop]
            iframe = reader.get_batch(iframe_idx).permute(0, 3, 1, 2) / 255
        input_mask_gop = torch.tensor([0] * iframe.size(0) + [1] * (resample_num_gop - iframe.size(0)),
                                      dtype=torch.bool)
        if sample == "pad" and iframe.size(0) < resample_num_gop:
            iframe = pad_tensor(iframe, target_size=resample_num_gop, dim=0)
        assert iframe.size(0) == resample_num_gop
        timer("stack_iframe")
        # encode motion
        assert mv_frame_gop[0][0]["motion_vector"].shape[-1] == 4, \
            "format is avc, but motion vector has {} !=4 dims".format(mv_frame_gop[0][0]["motion_vector"].shape[-1])
        # h264 motion vector
        for g in mv_frame_gop:
            for f in g:
                if "encoded" in f and f["encoded"]:
                    continue
                else:
                    f["motion_vector"] = torch.from_numpy( 
                        f["motion_vector"].transpose((2, 0, 1)).astype(np.float32)
                    )
                    f["encoded"] = True
        timer("encode_motion")
        # stack mv
        motion_vector = []
        type_ids_mv = []
        input_mask_mv = []
        for gop_idx in range(len(mv_frame_gop)):
            gop_mv = torch.stack([f["motion_vector"] for f in mv_frame_gop[gop_idx]])
            input_mask_mv.append(torch.tensor([0] * gop_mv.size(0) + [1] * (resample_num_mv - gop_mv.size(0)),
                                              dtype=torch.bool))
            type_ids_mv.append(torch.tensor([0 if f["pict_type"] == "P" else 1 for f in mv_frame_gop[gop_idx]] +
                                            [2] * (resample_num_mv - gop_mv.size(0)), dtype=torch.long))
            if sample == "pad" and gop_mv.size(0) < resample_num_mv:  # pad for mv in each gop
                gop_mv = pad_tensor(gop_mv, target_size=resample_num_mv, dim=0)
            assert gop_mv.size(0) == resample_num_mv
            motion_vector.append(gop_mv)
        motion_vector = torch.stack(motion_vector)
        type_ids_mv = torch.stack(type_ids_mv)
        input_mask_mv = torch.stack(input_mask_mv)


        # pad for gop
        # pad for gop
        if sample == "pad" and motion_vector.size(0) < resample_num_gop:  # pad for gop number
            motion_vector = pad_tensor(motion_vector, target_size=resample_num_gop, dim=0)
            type_ids_mv = pad_tensor(type_ids_mv, target_size=resample_num_gop, dim=0, pad_value=2)
            input_mask_mv = pad_tensor(input_mask_mv, target_size=resample_num_gop, dim=0, pad_value=1)
        
        # ==========================================================
        # 3. DIMENSION REDUCTION & NOISE MASKING
        # ==========================================================
        # A: Convert 4-Channel (src_x, src_y, dst_x, dst_y) to 2-Channel (dx, dy)
        dx = motion_vector[:, :, 2, :, :] - motion_vector[:, :, 0, :, :]
        dy = motion_vector[:, :, 3, :, :] - motion_vector[:, :, 1, :, :]
        motion_vector_2d = torch.stack([dx, dy], dim=2) # Shape: [GOP, MV, 2, 56, 56]
        
        # B: Magnitude Masking (Filter FPS duplicates & Scene Cuts)
        mag = motion_vector_2d.abs().mean(dim=(2, 3, 4), keepdim=True)
        
        # > 0.01 ignores the FFmpeg FPS duplication zeros
        # < 40.0 ignores the exploding scene-cut frames
        valid_physics_mask = ((mag > 0.01) & (mag < 40.0)).float()
        
        # C: Zero out the corrupted frames before returning
        motion_vector = motion_vector_2d * valid_physics_mask
        # ==========================================================

        assert motion_vector.size(0) == resample_num_gop, \
            "motion vector gop number is not correct, got {}, expect {}".format(motion_vector.size(0), resample_num_gop)
        assert motion_vector.size(1) == resample_num_mv, \
            "motion vector mv number is not correct, got {}, expect {}".format(motion_vector.size(1), resample_num_mv)
        timer("stack_motion")
        ret = {"iframe": iframe, "motion_vector": motion_vector,
               "input_mask_gop": input_mask_gop, "input_mask_mv": input_mask_mv, "type_ids_mv": type_ids_mv}
        if with_residual:
            residual = []
            input_mask_res = []
            for gop_idx in range(len(mv_frame_gop)):
                gop_res = torch.stack(
                    [torch.from_numpy(f["residual"].transpose(2, 0, 1)) for f in res_frame_gop[gop_idx]]
                )
                input_mask_res.append(torch.tensor([0] * gop_res.size(0) + [1] * (resample_num_res - gop_res.size(0)),
                                                   dtype=torch.bool))
                if sample == "pad" and gop_res.size(0) < resample_num_res:
                    gop_res = pad_tensor(gop_res, target_size=resample_num_res, dim=0)
                residual.append(gop_res)
            residual = torch.stack(residual)
            input_mask_res = torch.stack(input_mask_res)
            if sample == "pad" and residual.size(0) < resample_num_gop:  # pad for gop number
                residual = pad_tensor(residual, target_size=resample_num_gop, dim=0, pad_value=128)
                input_mask_res = pad_tensor(input_mask_res, target_size=resample_num_gop, dim=0, pad_value=1)
            ret["residual"] = residual
            ret["input_mask_res"] = input_mask_res
            timer("stack_residual")
        if with_bp_rgb:
            bp_rgb_padded_list=[]
            for g in mv_frame_gop:
                # stack P-frames in the current gop
                gop_rgb = torch.stack([torch.from_numpy(f["rgb"]).permute(2, 0, 1) for f in g])/ 255.0
                #pad missing P-frames with zeros (balck) if this GOP is short
                if sample == "pad" and gop_rgb.size(0)<resample_num_mv:
                    gop_rgb = pad_tensor(gop_rgb, target_size=resample_num_mv, dim=0, pad_value=0)
                bp_rgb_padded_list.append(gop_rgb)

            bp_rgb = torch.stack(bp_rgb_padded_list)

            #Pad missing GOPs with zeros (black) if total GOP number is short
            if sample == "pad" and bp_rgb.size(0)<resample_num_gop:
                bp_rgb = pad_tensor(bp_rgb, target_size=resample_num_gop, dim=0, pad_value=0)

            ret["bp_rgb"] = bp_rgb
            timer("stack_bp_rgb")
        logger.debug(timer.get_info(averaged=False))  # debug output about speed
        return ret, True
    except Exception:  # TODO: too broad exception
        print(f"video load error: {video_path}")
        traceback.print_exc()
        traceback.print_exc(file=open("video_reader_error.log", "a"))
        # create a dummy return data
        ret = {
            "iframe": torch.zeros((resample_num_gop, 3, 224, 224), dtype=torch.float),
            "motion_vector": torch.zeros((resample_num_gop, resample_num_mv, 4, 56, 56), dtype=torch.float),
            "input_mask_gop": torch.ones((resample_num_gop,), dtype=torch.bool),
            "input_mask_mv": torch.ones((resample_num_gop, resample_num_mv), dtype=torch.bool),
            "type_ids_mv": torch.zeros((resample_num_gop, resample_num_mv), dtype=torch.long)
        }
        return ret, False


def get_video_len(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen
