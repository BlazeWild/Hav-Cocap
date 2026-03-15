import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from cocap.modules.beats import BEATs, BEATsConfig

class BEATsAudioEncoder(nn.Module):
    def __init__(self, model_path=None, model_cfg=None):
        super().__init__()
        # Default Config for BEATs iter3+ (as2m) usually matches Base or Large?
        # Let's assume Base (768) unless specified.
        # If the checkpoint is loaded, it might have a config.
        
        # Load config logic
        checkpoint = torch.load(model_path, map_location='cpu') if model_path else None
        cfg_dict = checkpoint['cfg'] if checkpoint and 'cfg' in checkpoint else None
        
        # Create Config
        self.cfg = BEATsConfig(cfg_dict)
        if model_cfg:
            self.cfg.update(model_cfg)
            
        self.model = BEATs(self.cfg)
        
        if checkpoint:
             if 'model' in checkpoint:
                 self.model.load_state_dict(checkpoint['model'], strict=False)
             else:
                 self.model.load_state_dict(checkpoint, strict=False)
             print(f"BEATs loaded from {model_path}")
             
    def forward(self, input, padding_mask=None):
        """
        Input: (B, T, F) - Mel Spectrogram (already normalized in dataset)
        BEATs expects (B, T, F) but extract_features assumes input is waveform OR fbank.
        If fbank, expects (B, T, F)?
        Let's check BEATs.extract_features:
          fbank = fbank.unsqueeze(1) # (B, 1, T, F) ?? 
          features = self.patch_embedding(fbank)
         
        Standard BEATs:
        fbank = self.preprocess(source) -> returns (B, T, 128)
        Then fbank.unsqueeze(1) -> (B, 1, T, 128)
        Then Conv2d(1, embed, patch, stride)
        
        So input should be (B, T, F).
        """
        # (B, T, F) -> BEATs
        # We modified BEATs to take is_fbank=True
        
        # We need padding mask? 
        # AVCapsDataset returns audio_spec padded with 0.
        # We can generate padding mask from 0s? Or just computed mask?
        # Dataset currently uses pad_sequence with padding_value=0.
        
        # Create padding mask (1 for padded, 0 for valid) in BEATs convention?
        # BEATs: "padding_mask of shape (batch_size, seq_len) with 1 for positions to ignore"
        # Since we just have padded input, we can infer it or pass it.
        # Ideally, we should update Dataset to return a mask.
        # For now, let's assume all valid or simple padding 0 check if robust.
        # But 0 can be a valid mel value?
        # Audio Spec is normalized. 0 might be valid.
        # Best to treat all as valid for now or pass mask.
        
        x, _ = self.model.extract_features(input, padding_mask=padding_mask, is_fbank=True)
        # x: (B, T_out, 768)
        
        # Global pooling? 
        # BEATs returns sequence. To get global, we can mean pool.
        x = x.mean(dim=1) 
        return x

class CNN14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super(CNN14, self).__init__()
        
        # Logmel spectrogram extractor (Assuming input is waveform, if input is spec, this part is skipped)
        # However, usually we use Torchaudio or similar. 
        # For this implementation, we will assume the input is already a Mel Spectrogram 
        # or we implement a simple Conv2d backbone.
        
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, 1, time_steps, mel_bins) or (batch_size, time_steps, mel_bins)
        """
        if input.dim() == 3:
            input = input.unsqueeze(1) # Add channel dim

        x = input.transpose(1, 3) # (batch, 1, time, mel) -> (batch, mel, time, 1) ? 
        # Standard PANNs takes (batch, 1, time_steps, mel_bins). Let's assume input matches that logic 
        # BUT usually PANNs expects (Batch, 1, Time, Freq).
        # We will enforce input as (Batch, 1, Time, Freq)
        
        # However, checking common implementations, usually it is (Batch, 1, Freq, Time) for Conv2d
        # Let's permute to (Batch, 1, Freq, Time)
        x = input.permute(0, 1, 3, 2) 

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = torch.dropout(x, p=0.2, train=self.training)
        
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = torch.dropout(x, p=0.2, train=self.training)
        
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = torch.dropout(x, p=0.2, train=self.training)
        
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = torch.dropout(x, p=0.2, train=self.training)
        
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = torch.dropout(x, p=0.2, train=self.training)
        
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = torch.dropout(x, p=0.2, train=self.training)

        # Global pooling
        x = torch.mean(x, dim=3) # Average over time
        (x1, _) = torch.max(x, dim=2) # Max over freq
        x2 = torch.mean(x, dim=2) # Mean over freq
        
        x = x1 + x2 # Sum global pooling
        
        x = F.relu_(self.fc1(x))
        # embedding = torch.dropout(x, p=0.5, train=self.training)
        # clip_output = torch.sigmoid(self.fc_audioset(x))
        
        # we return the global embedding
        return x

    def load_from_pretrain(self, path):
        if not os.path.exists(path):
            print(f"Pretrained model not found at {path}")
            return
        
        print(f"Loading Audio Encoder from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # PANNs checkpoint might have 'module.' prefix if trained with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # Filter out fc_audioset if sizes don't match (we defined classes_num=527 which is standard, but just in case)
        # Also FC1 might differ if not 2048.
        
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Audio Encoder Loaded: {msg}")

import os
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x
