
import os
import glob
import torch
import numpy as np
import av
import logging
import collections
import json
import random
import sys
from torch.utils.data import IterableDataset

# Add CoCap path for tokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../CoCap")))
try:
    from cocap.modules.clip.simple_tokenizer import SimpleTokenizer
except ImportError:
    print("Warning: Could not import SimpleTokenizer. Captions will not be tokenized.")
    SimpleTokenizer = None

class GOPDataloader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.container = av.open(video_path)
        self.audio_resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        # Suppress verbose ffmpeg logs
        av.logging.set_level(av.logging.ERROR)

    def __iter__(self):
        self.container.seek(0)
        video_stream = self.container.streams.video[0]
        audio_stream = self.container.streams.audio[0] if self.container.streams.audio else None

        # Decode Video
        frames = []
        keyframe_indices = []
        try:
            for i, frame in enumerate(self.container.decode(video_stream)):
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
                if frame.key_frame:
                    keyframe_indices.append(i)
        except Exception as e:
            # logging.warning(f"Error decoding video {self.video_path}: {e}")
            return

        # Decode Audio
        audio_samples = np.array([], dtype=np.int16)
        if audio_stream:
            self.container.seek(0) # Logic separation, safer to seek for audio? 
                                   # Actually decoding different streams in one loop is better for sync, 
                                   # but here we decode sequentially for simplicity.
            audio_chunks = []
            try:
                for frame in self.container.decode(audio_stream):
                    resampled_frames = self.audio_resampler.resample(frame)
                    for resampled_frame in resampled_frames:
                        chunk = resampled_frame.to_ndarray()
                        # Flatten if (1, N)
                        if chunk.ndim == 2:
                            chunk = chunk[0]
                        audio_chunks.append(chunk)
            except Exception as e:
                # logging.warning(f"Error decoding audio {self.video_path}: {e}")
                pass
            
            if audio_chunks:
                audio_samples = np.concatenate(audio_chunks)
        
        if not frames:
            return

        total_frames = len(frames)
        fps = float(video_stream.average_rate)
        if fps <= 0: fps = 30.0 # Standard fallback
        sample_rate = 16000
        samples_per_frame = sample_rate / fps

        # Iterate GOPs
        for k_idx, start_frame in enumerate(keyframe_indices):
            end_frame = keyframe_indices[k_idx+1] if k_idx+1 < len(keyframe_indices) else total_frames
            
            # I-Frame
            iframe_np = frames[start_frame]
            
            # Audio Segment (1 second @ 16kHz)
            start_sample = int(start_frame * samples_per_frame)
            seg_len = 16000
            
            audio_seg = np.zeros(seg_len, dtype=np.float32)
            
            # Extract
            source_seg = []
            if len(audio_samples) > start_sample:
                source_seg = audio_samples[start_sample : start_sample + seg_len]
            
            if len(source_seg) > 0:
                source_seg = source_seg.astype(np.float32) / 32768.0 # Normalize -1..1
                length = min(len(source_seg), seg_len)
                audio_seg[:length] = source_seg[:length]
            
            yield {
                "iframe": iframe_np,
                "audio": audio_seg
            }

class HavCocapDataset(IterableDataset):
    def __init__(self, dataset_root, split="train", max_words=77, tokenizer_path=None, blacklist_file=None, update_blacklist=False):
        self.dataset_root = dataset_root
        self.split = split
        self.max_words = max_words
        self.split_dir = os.path.join(dataset_root, split)
        self.blacklist_file = blacklist_file
        self.update_blacklist = update_blacklist
        
        # Load Captions
        caption_file = os.path.join(self.split_dir, f"{split}_captions.json")
        self.captions_data = {}
        if os.path.exists(caption_file):
            print(f"Loading captions from {caption_file}...")
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    self.captions_data = json.load(f)
            except Exception as e:
                print(f"Failed to load captions: {e}")
        
        # Init Tokenizer
        if SimpleTokenizer:
            self.tokenizer = SimpleTokenizer()
        else:
            self.tokenizer = None
        
        # Find Videos
        video_files = glob.glob(os.path.join(self.split_dir, "**", "*.mp4"), recursive=True)
        print(f"Found {len(video_files)} videos in {self.split_dir}")

        # Filter Blacklist
        if blacklist_file and os.path.exists(blacklist_file):
            print(f"Loading blacklist from {blacklist_file}...")
            try:
                with open(blacklist_file, 'r') as f:
                    blacklist = json.load(f)
                # Normalize blacklist to absolute paths for comparison
                self.blacklist_set = set(os.path.normpath(os.path.abspath(p)) for p in blacklist)
                
                # Filter
                self.video_files = []
                for v in video_files:
                    if os.path.normpath(os.path.abspath(v)) not in self.blacklist_set:
                        self.video_files.append(v)
                
                print(f"Filtered {len(video_files) - len(self.video_files)} blacklisted videos. Remaining: {len(self.video_files)}")
            except Exception as e:
                print(f"Failed to load blacklist: {e}")
                self.video_files = video_files
                self.blacklist_set = set()
        else:
            self.video_files = video_files
            self.blacklist_set = set()
            
    # ... existing methods ...

    def _append_to_blacklist(self, video_path):
        if not self.blacklist_file: return
        try:
            abs_path = os.path.normpath(os.path.abspath(video_path))
            # Check if already added (in memory set)
            if abs_path in self.blacklist_set: return
            
            # Append to file
            # Ideally we read, append, write. But for speed/concurrency, simple append to a plain text or update json might be risky?
            # JSON requires read-write.
            # We will try to read, update, write.
            # This is race-condition prone with multiple workers, but better than nothing.
            # Or better: write to a separate "new_blacklist.txt" and merge later?
            # User wants it "included in code". Let's try direct JSON update with a lock file or just risk it (rare events).
            # Actually, standard JSON append is hard for concurrent.
            # Use a separate line-based log file for new blacklist items?
            # Let's write to blacklist_file but handle it carefully.
            
            # Simple approach: Load, append, Save.
            # If multiple workers do this, it might corrupt.
            # Safe approach: Append to a side-file "corrupt_files_detected.txt"
            # User can merge.
            # BUT user said "blacklist_file should be included".
            # Let's try:
            
            # Load current
            current_list = []
            if os.path.exists(self.blacklist_file):
                try:
                    with open(self.blacklist_file, 'r') as f:
                        current_list = json.load(f)
                except: pass
            
            if abs_path not in current_list:
                current_list.append(abs_path)
                with open(self.blacklist_file, 'w') as f:
                    json.dump(current_list, f, indent=4)
                    
            self.blacklist_set.add(abs_path)
            print(f"Added {video_path} to blacklist.")
            
        except Exception as e:
            print(f"Failed to update blacklist: {e}")

    def _get_caption_tokens(self, video_id):
        text = ""
        if video_id in self.captions_data:
            entry = self.captions_data[video_id]
            # Collect all caps
            caps = []
            if isinstance(entry, dict):
                if "audio_captions" in entry: caps.extend(entry["audio_captions"])
                if "visual_captions" in entry: caps.extend(entry["visual_captions"])
            
            if caps:
                text = random.choice(caps)
        
        # Tokenize
        if not self.tokenizer:
            # Return dummy if no tokenizer
            return torch.zeros(self.max_words, dtype=torch.long), torch.zeros(self.max_words, dtype=torch.long)
        
        if not text:
            text = "unknown" # Fallback
            
        return self._tokenize(text)

    def _tokenize(self, text):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        
        # SimpleTokenizer.encode returns list of ints
        # We need to handle truncation/padding manually
        tokens = self.tokenizer.encode(text)
        tokens = [sot_token] + tokens + [eot_token]
        
        result = torch.zeros(self.max_words, dtype=torch.long)
        mask = torch.zeros(self.max_words, dtype=torch.long)
        
        if len(tokens) > self.max_words:
            tokens = tokens[:self.max_words]
            tokens[-1] = eot_token
            
        result[:len(tokens)] = torch.tensor(tokens)
        mask[:len(tokens)] = 1
        
        return result, mask

    def __iter__(self):
        # Determine strict or random iteration?
        # IterableDataset usually iterates once.
        # Shuffle files
        files = list(self.video_files)
        random.shuffle(files)
        
        for video_path in files:
            # Extract ID from filename
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            # Prepare Caption
            input_ids, input_mask = self._get_caption_tokens(video_id)
            
            # Process Video
            try:
                loader = GOPDataloader(video_path)
            except Exception as e:
                # print(f"Skipping corrupt video {video_path}: {e}")
                if self.update_blacklist:
                    self._append_to_blacklist(video_path)
                continue
            
            gops_iframe = []
            gops_audio = []
            
            try:
                iterator = iter(loader)
                # Check for empty or error inside iterator
                # GOPDataloader.__iter__ catches errors but might yield nothing
                
                has_data = False
                for gop_data in iterator:
                    has_data = True
                    try:
                        # Transform I-Frame
                        img = torch.tensor(gop_data['iframe']).permute(2, 0, 1).float() / 255.0
                        # Resize (Bilinear)
                        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                        # Norm
                        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                        img = (img - mean) / std
                        
                        gops_iframe.append(img)
                        gops_audio.append(torch.tensor(gop_data['audio']))
                    except Exception as e:
                         # print(f"Error processing GOP for {video_path}: {e}")
                         continue
                
                if not has_data:
                     # Treat as corrupt if no GOPs yielded
                     if self.update_blacklist:
                        self._append_to_blacklist(video_path)
                     continue

            except Exception as e:
                # print(f"Error iterating video {video_path}: {e}")
                if self.update_blacklist:
                     self._append_to_blacklist(video_path)
                continue
            
            if not gops_iframe:
                continue
            
            # Stack GOPs
            # Limit to defined number of GOPs ? CoCap uses 8 typically.
            # If video has > 8, we sample or take first 8.
            num_gops = 8
            if len(gops_iframe) > num_gops:
                indices = np.linspace(0, len(gops_iframe)-1, num_gops, dtype=int)
                gops_iframe = [gops_iframe[i] for i in indices]
                gops_audio = [gops_audio[i] for i in indices]
            elif len(gops_iframe) < num_gops:
                # Pad with last frame? or zeros?
                # For prototype, duplicate last
                while len(gops_iframe) < num_gops:
                    gops_iframe.append(gops_iframe[-1])
                    gops_audio.append(gops_audio[-1])
            
            # Create Tensors
            # I-Frame: (T, C, H, W)
            video_iframe = torch.stack(gops_iframe)
            video_audio = torch.stack(gops_audio)
            
            # Dummy Motion/Residual
            T = video_iframe.shape[0] # Should be 8
            # Motion: (T, N_mv=1, C=2, H=56, W=56) - CoCap standard
            # Residual: (T, N_res=1, C=3, H=224, W=224)
            # Actually CoCap uses N_mv=1 usually, if not specified in config.
            # We'll use defaults.
            
            # Using basic shapes compatible with model defaults
            video_motion = torch.zeros(T, 1, 4, 56, 56) # 4 channels for motion
            video_residual = torch.zeros(T, 1, 3, 224, 224)
            
            yield {
                "iframe": video_iframe,
                "motion": video_motion,
                "residual": video_residual,
                "audio": video_audio,
                "input_ids": input_ids,
                "input_mask": input_mask,
                "video_id": video_id
            }
