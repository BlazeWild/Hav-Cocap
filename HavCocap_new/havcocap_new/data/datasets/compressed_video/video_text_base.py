# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 14:54
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : video_text_base.py
import logging
import os.path
import random
from dataclasses import dataclass
from typing import Optional

import torch
import subprocess
import tempfile
import torchaudio

logger = logging.getLogger(__name__)


def extract_audio_from_video(
    video_path,
    max_length_sec=15,
    sample_rate=16000,
    audio_config=None,
    gop_center_frame_idx: Optional[torch.Tensor] = None,
    video_fps: Optional[float] = None,
    **kwargs,
):
    """
    Extracts the audio track from the given video file.
    First checks if a pre-extracted .pt tensor exists to drastically speed up data loading.
    If not, uses FFmpeg to extract, resample, and pad to max_length_sec.

    Extra kwargs are accepted for compatibility with dataset-specific callers
    (e.g. max_frames, sample_mode) and are intentionally ignored here.
    """
    gop_window_sec = 1.0
    if audio_config:
        # Optional per-dataset overrides (kept backward compatible)
        sample_rate = int(audio_config.get("sample_rate", sample_rate))
        # Some configs use `audio_length` in seconds
        max_length_sec = float(audio_config.get("audio_length", max_length_sec))
        gop_window_sec = float(audio_config.get("gop_audio_window_sec", gop_window_sec))

    max_samples = int(max_length_sec * sample_rate)
    
    # Try loading pre-calculated .pt file to save IO/CPU cost.
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Assuming standard structure where Charades_audio_tensors is next to Charades_filtered_240
    audio_pt_path = os.path.join(os.path.dirname(video_dir), "Charades_audio_tensors", f"{video_name}.pt")

    waveform = None
    if os.path.exists(audio_pt_path):
        try:
            waveform = torch.load(audio_pt_path)
            if isinstance(waveform, torch.Tensor) and waveform.dim() > 1:
                waveform = waveform.squeeze(0)
        except Exception:
            waveform = None

    # Fallback to realtime extraction via ffmpeg when cached audio is unavailable.
    # Important: when GOP slicing is requested we decode full audio (no -t truncation).
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_wav:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-ac', '1', '-ar', str(sample_rate)
        ]
        if gop_center_frame_idx is None:
            cmd += ['-t', str(max_length_sec)]
        cmd += ['-f', 'wav', tmp_wav.name]
        
        try:
            if waveform is None:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                waveform, _ = torchaudio.load(tmp_wav.name)
                waveform = waveform.squeeze(0)

            # GOP-centered slicing mode: return [num_gop, gop_window_sec * sample_rate]
            if gop_center_frame_idx is not None:
                if isinstance(gop_center_frame_idx, torch.Tensor):
                    center_idx = gop_center_frame_idx.to(torch.float32).view(-1)
                else:
                    center_idx = torch.tensor(gop_center_frame_idx, dtype=torch.float32).view(-1)

                # Keep valid (non-padding) GOP entries only.
                valid_mask = center_idx > 0
                if valid_mask.any():
                    center_idx = center_idx[valid_mask]
                num_gop_total = int(gop_center_frame_idx.numel() if isinstance(gop_center_frame_idx, torch.Tensor)
                                    else len(gop_center_frame_idx))

                if video_fps is None or float(video_fps) <= 0:
                    # If fps metadata is missing, use uniform centers over available audio duration.
                    duration_sec = float(max(1, waveform.numel()) / sample_rate)
                    if center_idx.numel() == 0:
                        center_times = torch.linspace(0.5, max(0.5, duration_sec - 0.5), steps=max(1, num_gop_total))
                    else:
                        center_times = torch.linspace(0.5, max(0.5, duration_sec - 0.5), steps=center_idx.numel())
                else:
                    center_times = center_idx / float(video_fps)

                win_samples = int(round(gop_window_sec * sample_rate))
                half = win_samples // 2
                clips = []
                total_samples = waveform.numel()
                for t in center_times.tolist():
                    center_sample = int(round(t * sample_rate))
                    start = center_sample - half
                    end = start + win_samples
                    left_pad = max(0, -start)
                    right_pad = max(0, end - total_samples)
                    start = max(0, start)
                    end = min(total_samples, end)
                    segment = waveform[start:end]
                    if left_pad or right_pad:
                        segment = torch.nn.functional.pad(segment, (left_pad, right_pad))
                    if segment.numel() != win_samples:
                        segment = torch.nn.functional.pad(segment[:win_samples], (0, max(0, win_samples - segment.numel())))
                    clips.append(segment)

                if len(clips) == 0:
                    clips = [torch.zeros(win_samples)]

                clips = torch.stack(clips, dim=0)
                # Re-pad back to requested GOP count when padded GOPs were present.
                if clips.size(0) < num_gop_total:
                    pad = clips.new_zeros(num_gop_total - clips.size(0), win_samples)
                    clips = torch.cat([clips, pad], dim=0)
                return clips

            # Legacy fixed-length single-audio mode: return [max_samples]
            num_samples = waveform.shape[0]
            if num_samples < max_samples:
                pad = max_samples - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            elif num_samples > max_samples:
                waveform = waveform[:max_samples]
            return waveform

        except Exception as e:
            # If video has no audio or ffmpeg fails, return zeros with expected shape.
            if gop_center_frame_idx is not None:
                n_gop = int(gop_center_frame_idx.numel()) if isinstance(gop_center_frame_idx, torch.Tensor) else len(gop_center_frame_idx)
                win_samples = int(round(gop_window_sec * sample_rate))
                return torch.zeros((n_gop, win_samples), dtype=torch.float32)
            return torch.zeros(max_samples, dtype=torch.float32)

def get_tokenized_words(sentence: str, tokenizer, max_words):
    words = tokenizer.tokenize(sentence)
    words = ["[CLS]"] + words
    total_length_with_cls = max_words - 1
    if len(words) > total_length_with_cls:
        words = words[:total_length_with_cls]
    words = words + ["[SEP]"]
    return words


def get_text_inputs(sentence: str, tokenizer, max_words):
    """
    1. tokenize
    2. add [CLS] and [SEP] token, limit the length
    3. create mask and token type
    4. pad to max_words
    :param sentence:
    :param tokenizer:
    :param max_words:
    :return: 1 dim tensor, shape is (max_words,)
    """
    words = get_tokenized_words(sentence, tokenizer, max_words)
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)  # 1 is keep, 0 is mask out
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == len(input_mask) == len(segment_ids) == max_words
    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)


def get_text_inputs_with_mlm(sentence: str, tokenizer, max_words):
    """
    Add mlm inputs and labels based on `get_text_inputs`
    :param sentence:
    :param tokenizer:
    :param max_words:
    :return: 1 dim tensor, shape is (max_words,)
    """
    input_ids, input_mask, segment_ids = get_text_inputs(sentence, tokenizer, max_words)

    # Mask Language Model <-----
    token_labels = []
    masked_tokens = get_tokenized_words(sentence, tokenizer, max_words)
    for token_id, token in enumerate(masked_tokens):
        if token_id == 0 or token_id == len(masked_tokens) - 1:
            token_labels.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            # 80% randomly change token to mask token
            if prob < 0.8:
                masked_tokens[token_id] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                masked_tokens[token_id] = random.choice(list(tokenizer.vocab.items()))[0]
            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            try:
                token_labels.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                token_labels.append(tokenizer.vocab["[UNK]"])
                logger.debug("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            token_labels.append(-1)
    # -----> Mask Language Model
    masked_token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    while len(masked_token_ids) < max_words:
        masked_token_ids.append(0)
        token_labels.append(-1)
    assert len(masked_token_ids) == len(token_labels) == max_words
    return input_ids, input_mask, segment_ids, torch.tensor(masked_token_ids), torch.tensor(token_labels)


@dataclass
class CVConfig:
    num_gop: int
    num_mv: int
    num_res: int
    with_residual: bool
    use_pre_extract: bool
    sample: str


def get_video(video_reader, video_path, max_frames, sample, hevc_config: None | CVConfig = None):
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    video_mask = torch.ones((max_frames,), dtype=torch.int)
    if video_reader.__name__ in ["read_frames_compressed_domain"]:
        assert hevc_config is not None, "hevc_config should be set when using read_frames_compressed_domain"
        video, _ = video_reader(video_path,
                                resample_num_gop=hevc_config.num_gop, resample_num_mv=hevc_config.num_mv,
                                resample_num_res=hevc_config.num_res,
                                with_residual=hevc_config.with_residual,
                                pre_extract=hevc_config.use_pre_extract,
                                sample=hevc_config.sample if hevc_config.sample == "pad" else sample)
    else:
        video, _ = video_reader(video_path, max_frames, sample)
    return video, video_mask
