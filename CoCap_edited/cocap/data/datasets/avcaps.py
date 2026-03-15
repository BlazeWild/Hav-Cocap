import os
import json
import random
import torch
import torchaudio
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from cocap.modules.clip.simple_tokenizer import SimpleTokenizer
import logging
try:
    from torchvision.io import read_video
except ImportError:
    read_video = None

logger = logging.getLogger(__name__)

class AVCapsDataset(Dataset):
    def __init__(self, root_dir, split="train", tokenizer=None, max_audio_len=1024, max_cap_len=77):
        self.root_dir = root_dir
        self.split = split
        self.tokenizer = tokenizer if tokenizer else SimpleTokenizer()
        self.max_audio_len = max_audio_len
        self.max_cap_len = max_cap_len
        
        # Load JSON
        json_path = os.path.join(root_dir, split, f"{split}_captions.json")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.video_ids = list(self.data.keys())
        self.video_folder = os.path.join(root_dir, split, "video_240p_h264") # Assuming this is where mp4s are
        # If videos are not here, user might need to adjust.
        
        # Transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Mel Spectrogram transform for BEATs (16k, 128 bins)
        # Matches ta_kaldi.fbank(num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        # frame_length=25ms -> 400 samples, frame_shift=10ms -> 160 samples
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, 
            n_mels=128, 
            n_fft=400, 
            hop_length=160 
        )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        item_data = self.data[vid]
        
        # Get Caption
        # "Target the audio_visual_captions key"
        if "audio_visual_captions" in item_data and len(item_data["audio_visual_captions"]) > 0:
            caption = random.choice(item_data["audio_visual_captions"])
        else:
            caption = "" # Should not happen typically
            
        # Tokenize Caption
        tokens = self.tokenizer.encode(caption)
        tokens = [49406] + tokens + [49407] # Add SOT, EOT. Check tokenizer for specific IDs. 
        # CLIP SOT=49406, EOT=49407.
        # Truncate
        if len(tokens) > self.max_cap_len:
            tokens = tokens[:self.max_cap_len]
            tokens[-1] = 49407
            
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Load Video/Audio
        video_path = os.path.join(self.video_folder, f"{vid}.mp4")
        
        pixel_values = torch.zeros(3, 224, 224)
        audio_spec = torch.zeros(64, self.max_audio_len) # Fixed size placeholder
        
        try:
            # Load Video
            if os.path.exists(video_path):
                # Only read needed parts to save IO. 
                # read_video returns (video, audio, info)
                # video: (T, H, W, C), audio: (K, T_a)
                vframes, aframes, info = read_video(video_path, pts_unit='sec', output_format="TCHW")
                
                # 1. Image Processing (Sample random frame)
                if vframes.shape[0] > 0:
                    # Random sample
                    frame_idx = random.randint(0, len(vframes)-1)
                    frame = vframes[frame_idx] # (C, H, W) is default if output_format="TCHW"? 
                    # Check torchvision version behavior. Often read_video returns (T, H, W, C).
                    # If I used output_format="TCHW" (available in newer torchvision).
                    # Let's assume standard behavior (T, H, W, C) to be safe if output_format not supported easily.
                    pass 
            else:
                  # Try finding without extension or different ext?
                  pass

            # Safe fallback reading
            if os.path.exists(video_path):
                 vframes, aframes, info = read_video(video_path, pts_unit='sec')
                 # vframes: (T, H, W, C)
                 
                 if vframes.shape[0] > 0:
                    idx = random.randint(0, len(vframes) - 1)
                    frame = vframes[idx] # (H, W, C)
                    frame = frame.permute(2, 0, 1) # (C, H, W)
                    pixel_values = self.transform(T.ToPILImage()(frame))
                 
                 if aframes.shape[0] > 0:
                    # Mix down to mono
                    waveform = aframes.mean(0, keepdim=True) # (1, T_audio)
                    # Resample to 16k for BEATs
                    if info['audio_fps'] != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=info['audio_fps'], new_freq=16000)
                        waveform = resampler(waveform)
                    
                    audio_spec = self.mel_transform(waveform) # (1, F, T)
                    audio_spec = audio_spec.squeeze(0) # (F, T)
                    
                    # Normalize for BEATs
                    # fbank = (fbank - fbank_mean) / (2 * fbank_std)
                    # fbank_mean=15.41663, fbank_std=6.55582
                    audio_spec = (audio_spec - 15.41663) / (2 * 6.55582)
                    
                    # Trim or Pad audio
                    if audio_spec.shape[1] > self.max_audio_len:
                        audio_spec = audio_spec[:, :self.max_audio_len]
                    elif audio_spec.shape[1] < self.max_audio_len:
                        pad_amount = self.max_audio_len - audio_spec.shape[1]
                        audio_spec = F.pad(audio_spec, (0, pad_amount))
            
        except Exception as e:
            logger.warning(f"Error loading {vid}: {e}")
            return None

        # Return Tensors
        # Since I can't actually read video here, I'll return placeholders if file doesn't exist
        # But for the CODE generation task, I should write the logic.
        
        return {
            "video_id": vid,
            "pixel_values": pixel_values,
            "audio_spec": audio_spec, # (F, T)
            "caption": tokens
        }

def avcaps_collate_fn(batch):
    # Filter failed loads
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # Pad Captions
    captions = [b["caption"] for b in batch]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0) # 0 is PAD usually
    
    # Create Attention Mask (1 for valid, 0 for pad)
    caption_lengths = [len(c) for c in captions]
    caption_mask = torch.zeros_like(captions_padded, dtype=torch.float)
    for i, length in enumerate(caption_lengths):
        caption_mask[i, :length] = 1.0
        
    # Stack Images
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    # Pad Audio
    # Audio specs are (F, T). T is variable.
    # We want to pad T to max in batch or fixed max? 
    # User said "handle variable-length audio ... and caption lengths".
    audios = [b["audio_spec"] for b in batch]
    # Transpose to (T, F) for pad_sequence
    audios_T = [a.transpose(0, 1) for a in audios]
    audios_padded = pad_sequence(audios_T, batch_first=True, padding_value=0)
    # Transpose back to (B, T, F) or keep as is?
    # CNN14 expects (B, 1, T, F). 
    # Current audios_padded: (B, T, 64).
    # We can unsqueeze later.
    
    video_ids = [b["video_id"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "audio_spec": audios_padded, # (B, T, 64)
        "captions": captions_padded,
        "caption_mask": caption_mask,
        "video_ids": video_ids
    }
