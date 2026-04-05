# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 14:54
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : video_text_base.py
import logging
import os.path
import random
from dataclasses import dataclass

import torch
import subprocess
import tempfile
import torchaudio

logger = logging.getLogger(__name__)


def extract_audio_for_gops(video_path, sampled_gop_indices, fps=30, gop_size=30, sample_rate=16000):
    """
    Extracts 1.0-second audio windows perfectly centered on the given visual GOPs.
    Pads with zeros if the audio window exceeds the video boundaries.
    """
    # 1. Load the entire audio waveform into RAM (fast for 10s videos)
    try:
        # torchaudio can often read audio directly from mp4
        waveform, sr = torchaudio.load(video_path)
        # Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Resample to 16kHz for VGGish
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)  # Shape: [total_samples]
    except Exception as e:
        # Fallback if no audio track exists
        waveform = torch.zeros(0)

    total_samples = waveform.shape[0]
    required_samples_per_window = sample_rate * 1  # 16000 samples for 1 second
    half_window = required_samples_per_window // 2
    
    # 2. Calculate the timestamps and extract slices
    gop_duration_sec = gop_size / fps
    audio_slices = []

    for gop_idx in sampled_gop_indices:
        gop_idx = int(gop_idx)

        # padded/invalid GOP slots -> zero audio
        if gop_idx < 0:
            audio_slices.append(torch.zeros(required_samples_per_window))
            continue

        # Math: Find the exact center of this GOP in seconds
        gop_start_time = gop_idx * gop_duration_sec
        gop_center_time = gop_start_time + (gop_duration_sec / 2.0)
        
        # Convert center time to audio sample index
        center_sample = int(gop_center_time * sample_rate)
        
        start_sample = center_sample - half_window
        end_sample = center_sample + half_window
        
        # Create a container of zeros for our 1-second slice
        slice_tensor = torch.zeros(required_samples_per_window)
        
        # 3. Padding Logic (Handle out-of-bounds)
        if total_samples > 0:
            # Calculate valid boundaries within the actual audio
            valid_start = max(0, start_sample)
            valid_end = min(total_samples, end_sample)
            
            if valid_start < valid_end:
                # Calculate where to place the valid audio inside our 1-second zero-padded container
                insert_start = valid_start - start_sample
                insert_end = insert_start + (valid_end - valid_start)
                
                slice_tensor[insert_start:insert_end] = waveform[valid_start:valid_end]
                
        audio_slices.append(slice_tensor)

    # Stack into shape: [Num_GOPs, 16000]
    return torch.stack(audio_slices)


def extract_audio_from_video(
        video_path,
        max_frames=8,
        sampled_gop_indices=None,
        audio_config=None,
        **kwargs,
):
    """
    Backward-compatible wrapper.
    If GOP indices are given, extract 1s centered clips per GOP.
    Otherwise, use sequential GOPs [0..max_frames-1].
    """
    audio_config = audio_config or {}
    fps = int(audio_config.get("fps", 30))
    gop_size = int(audio_config.get("gop_size", 30))
    sample_rate = int(audio_config.get("sample_rate", 16000))

    if sampled_gop_indices is None:
        sampled_gop_indices = list(range(max_frames))

    return extract_audio_for_gops(
        video_path=video_path,
        sampled_gop_indices=sampled_gop_indices,
        fps=fps,
        gop_size=gop_size,
        sample_rate=sample_rate,
    )

    

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
