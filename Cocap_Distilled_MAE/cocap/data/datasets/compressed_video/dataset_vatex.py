# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @File    : dataset_vatex.py

import json
import os
import random
from collections import defaultdict
from typing import Literal

import torch
from torch.utils import data
from torchvision import transforms

# --- HAV-COCAP CHANGE: Swap CLIP Tokenizer for GPT-2 Tokenizer ---
from transformers import GPT2Tokenizer
# -----------------------------------------------------------------

from .transforms import (DictNormalize, DictCenterCrop, DictRandomHorizontalFlip)
from .video_readers import VIDEO_READER_REGISTRY
from .video_text_base import get_video, CVConfig

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

class VATEXCaptioningDataset(data.Dataset):
    def __init__(
            self,
            video_root: str,
            max_words: int,
            max_frames: int,
            unfold_sentences: bool,
            video_size: tuple[int, int],
            metadata: str,
            video_reader: str,
            cv_config: CVConfig,
            split: Literal["train", "val", "test"], # ADDED "val"
    ):
        self.split = split
        self.video_root = video_root
        self.max_words = max_words
        self.max_frames = max_frames
        self.unfold_sentences = unfold_sentences  
        self.height, self.width = video_size
        self.sentences = []  
        self.h265_cfg = cv_config
        metadata = load_json(metadata)

        # Initialize the correct tokenizer for Phase 3 SFT
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        split_video_ids = metadata[split].copy()
        
        if self.unfold_sentences:
            for item in metadata["metadata"]:
                if item["video_id"] in split_video_ids:
                    self.sentences.append([item["video_id"], [item["sentence"]]])
                    if split in ("val", "test"):
                        split_video_ids.remove(item["video_id"])
        else:
            vid2sentence = defaultdict(list)
            for item in metadata["metadata"]:
                if item["video_id"] in split_video_ids:
                    vid2sentence[item["video_id"]].append(item["sentence"])
            self.sentences = list(vid2sentence.items())

        self.video_reader = VIDEO_READER_REGISTRY.get(video_reader)
        
        # Transforms (Perfectly safe for our 2-channel MVs)
        normalize = DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if split == "train":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                DictRandomHorizontalFlip(),
                normalize
            ])
        elif split in ("val", "test"): # ADDED "val"
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                normalize
            ])
        else:
            raise NotImplementedError

        if split in ("val", "test"): # ADDED "val"
            json_ref = {k: [] for k in metadata[split]}
            for sentence in metadata["metadata"]:
                if sentence["video_id"] in json_ref:
                    json_ref[sentence["video_id"]].append(sentence["sentence"])
            self.json_ref = json_ref

    def __len__(self):
        return len(self.sentences)

    def _get_video_path(self, video_id):
        return os.path.join(self.video_root, f"{video_id}.mp4")

    def _get_video(self, video_id):
        video, video_mask = get_video(video_reader=self.video_reader,
                                      video_path=self._get_video_path(video_id),
                                      max_frames=self.max_frames,
                                      sample="rand" if self.split == "train" else "uniform",
                                      hevc_config=self.h265_cfg)
        if self.transform is not None:
            video = self.transform(video)
        return video, video_mask

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)

        # --- HAV-COCAP CHANGE: GPT-2 Tokenization ---
        # Encode the text using GPT-2's BPE vocabulary
        encoded_dict = self.tokenizer(
            sentence,
            max_length=self.max_words,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded_dict["input_ids"].squeeze(0)
        input_mask = encoded_dict["attention_mask"].squeeze(0)

        # Autoregressive shifting for GPT-2 targets
        input_labels = input_ids.clone()
        # tell pytorch to ignore padding in the loss calculation
        input_labels[input_ids == self.tokenizer.pad_token_id] = -100
        # --------------------------------------------

        video, video_mask = self._get_video(video_id)
        
        return {
            "video": video,
            "video_mask": video_mask,
            "input_ids": input_ids,
            "input_labels": input_labels,
            "input_mask": input_mask,
            "metadata": (video_id, sentence)
        }