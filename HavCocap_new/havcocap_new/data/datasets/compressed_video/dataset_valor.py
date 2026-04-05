import os
import random
from collections import defaultdict
from typing import Literal, Dict, Any

import torch
from torch.utils import data
from torchvision import transforms

from havcocap_new.modules.clip import clip
from havcocap_new.utils.json import load_json
from .transforms import (DictNormalize, DictCenterCrop, DictRandomHorizontalFlip)
from .video_readers import VIDEO_READER_REGISTRY
from .video_text_base import get_video, CVConfig, extract_audio_from_video


class ValorCaptioningDataset(data.Dataset):

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
            split: Literal["train", "val", "test"],
            audio_config: dict = None,  # Added to handle HavCocap 1.0s audio slices
    ):
        self.split = split
        self.video_root = video_root
        self.max_words = max_words
        self.max_frames = max_frames
        self.unfold_sentences = unfold_sentences  
        self.height, self.width = video_size
        self.sentences = []  
        self.h265_cfg = cv_config
        self.audio_config = audio_config or {}
        
        metadata_dict = load_json(metadata)

        split_video_ids = set()
        for v in metadata_dict['videos']:
            if v['split'] == split:
                split_video_ids.add(v['video_id'])

        if self.unfold_sentences:
            for item in metadata_dict["sentences"]:
                if item["video_id"] in split_video_ids:
                    self.sentences.append([item["video_id"], [item["caption"]]])
        else:
            vid2sentence = defaultdict(list)
            for item in metadata_dict["sentences"]:
                if item["video_id"] in split_video_ids:
                    vid2sentence[item["video_id"]].append(item["caption"])
            self.sentences = list(vid2sentence.items())

        self.video_reader = VIDEO_READER_REGISTRY.get(video_reader)
        
        # transforms
        normalize = DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if split == "train":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                DictRandomHorizontalFlip(),
                normalize
            ])
        elif split in ("test", "val"):
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                normalize
            ])
        else:
            raise NotImplementedError

        if split in ("test", "val"):
            json_ref = defaultdict(list)
            for sentence in metadata_dict["sentences"]:
                if sentence["video_id"] in split_video_ids:
                    json_ref[sentence["video_id"]].append(sentence["caption"])
            self.json_ref = dict(json_ref)

    def __len__(self):
        return len(self.sentences)

    def _get_video_path(self, video_id):
        # Primary format from our HPC preprocessing is .mp4
        mp4_path = os.path.join(self.video_root, f"{video_id}.mp4")
        if not os.path.exists(mp4_path):
            avi_path = os.path.join(self.video_root, f"{video_id}.avi")
            if os.path.exists(avi_path):
                return avi_path
        return mp4_path

    def _get_video(self, video_id):
        video_path = self._get_video_path(video_id)
        
        # Video extraction (returns dictionary of I-frames, P-frames, MVs, etc.)
        video, video_mask = get_video(
            video_reader=self.video_reader,
            video_path=video_path,
            max_frames=self.max_frames,
            sample="rand" if self.split == "train" else "uniform",
            hevc_config=self.h265_cfg
        )
        
        # Audio extraction: Passing audio_config so the extractor knows to 
        # slice 1.0s windows perfectly aligned with the sampled visual GOPs
        audio = extract_audio_from_video(
            video_path=video_path,
            audio_config=self.audio_config,
            gop_center_frame_idx=video.get("gop_center_frame_idx", None),
            video_fps=float(video.get("video_fps", torch.tensor(0.0)).item()) if "video_fps" in video else None,
            max_frames=self.max_frames,
            sample_mode="rand" if self.split == "train" else "uniform"
        )

        if self.transform is not None:
            video = self.transform(video)
            
        return video, video_mask, audio

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)

        input_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        input_mask = torch.zeros(self.max_words, dtype=torch.long)
        input_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1

        video, video_mask, audio = self._get_video(video_id)
        input_labels = torch.cat((input_ids[1:], torch.IntTensor([0])))
        
        return {
            # video
            "video": video,
            "video_mask": video_mask,
            "audio": audio,
            # text
            "input_ids": input_ids,
            "input_labels": input_labels,
            "input_mask": input_mask,
            # metadata
            "metadata": (video_id, sentence)
        }
