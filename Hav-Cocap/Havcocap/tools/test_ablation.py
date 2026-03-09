import os
import sys
import torch
import json
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Havcocap.model.hav_cocap import HavCocapModel, HavCocapCaptioner
from DataLoader.dataloader import HavCocapDataset
from torch.utils.data import DataLoader
from Havcocap.modules.compressed_video.compressed_video_captioner import CaptionHead
from Havcocap.modules.clip.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Hav-Cocap/dataset/AVCaps")
    parser.add_argument("--checkpoint", type=str, default="Hav-Cocap/checkpoints/checkpoint_epoch_100.pth")
    parser.add_argument("--blacklist_file", type=str, default="Hav-Cocap/corrupt_files.json")
    # Point dataset to the new 5 video folder
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    # Initialize typical dataset
    dataset = HavCocapDataset(
        dataset_root=args.data_root,
        split="test", 
        blacklist_file=args.blacklist_file,
    )
    
    # OVERRIDE Dataset internals for the 5 videos
    data_path = os.path.join(args.data_root, "test5videos")
    dataset.split_dir = data_path
    
    # Load captions for the 5 videos
    caption_file = os.path.join(data_path, 'test5_captions.json')
    with open(caption_file, 'r') as f:
         dataset.captions_data = json.load(f)
         
    # Force the dataloader to only iterate over the 5 custom videos
    import glob
    dataset.video_files = glob.glob(os.path.join(data_path, 'video', '*.mp4'))
    
    print(f"Found {len(dataset.video_files)} videos in test5videos subset.")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize Model
    tokenizer = SimpleTokenizer()
    base_model = HavCocapModel(embed_dim=512)
    caption_head = CaptionHead(
        word_embedding_size=512,
        visual_feature_size=512,
        max_v_len=16,
        max_t_len=77,
        hidden_size=512,
        vocab_size=49408,
        verbose=False 
    )
    model = HavCocapCaptioner(base_model, caption_head).to(device)
    
    # Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Checkpoint loaded successfully.")
    
    sot_token_id = tokenizer.encoder["<|startoftext|>"]
    eot_token_id = tokenizer.encoder["<|endoftext|>"]
    max_t_len = 77
    
    results_comparison = []
    
    for batch in dataloader:
        vid_id = batch['video_id'][0]
        # Move to GPU
        b_size = batch['iframe'].size(0)
        
        iframe = batch['iframe'].to(device)
        audio = batch['audio'].to(device)
        
        if batch['motion'] is not None and torch.is_tensor(batch['motion']):
            motion = batch['motion'].to(device)
            if motion.shape[3] == 2:
                padded = torch.zeros(motion.shape[0], motion.shape[1], motion.shape[2], 4, motion.shape[4], motion.shape[5], device=device)
                padded[:, :, :, :2, :, :] = motion
                motion = padded
        else:
            motion = torch.zeros(b_size, iframe.size(1), 1, 4, 56, 56).to(device)

        if batch['residual'] is not None and torch.is_tensor(batch['residual']):
            residual = batch['residual'].to(device)
        else:
            residual = torch.zeros(b_size, iframe.size(1), 1, 3, 224, 224).to(device)
            
        def generate_caption_ablation(iframe_t, motion_t, residual_t, audio_t):
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    visual_output = model.hav_cocap_model(
                        iframe=iframe_t,
                        motion=motion_t,
                        residual=residual_t,
                        audio=audio_t,
                        bp_type_ids=torch.zeros((b_size, iframe_t.size(1), 1), dtype=torch.long, device=device)
                    )
                    
                    input_ids = torch.full((b_size, 1), sot_token_id, dtype=torch.long, device=device)
                    finished = torch.zeros(b_size, dtype=torch.bool, device=device)
                    
                    for step in range(max_t_len - 1):
                        current_len = input_ids.size(1)
                        input_mask = torch.ones(b_size, current_len, dtype=torch.long, device=device)
                        
                        padded_input_ids = torch.zeros(b_size, max_t_len, dtype=torch.long, device=device)
                        padded_input_mask = torch.zeros(b_size, max_t_len, dtype=torch.long, device=device)
                        
                        padded_input_ids[:, :current_len] = input_ids
                        padded_input_mask[:, :current_len] = input_mask
                        
                        prediction_scores = model.caption_head(
                            visual_output=visual_output,
                            input_ids=padded_input_ids,
                            input_mask=padded_input_mask
                        )
                        
                        next_token_logits = prediction_scores[:, current_len - 1, :].clone()
                        next_token_logits[:, 0] = float('-inf')  # Penalize pad token
                        
                        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_tokens], dim=1)
                        
                        is_eot = (next_tokens.squeeze(-1) == eot_token_id)
                        finished |= is_eot
                        
                        if finished.all():
                            break
                            
            tokens = input_ids[0].tolist()
            clean_tokens = []
            for t in tokens:
                if t == eot_token_id:
                    break
                if t != sot_token_id and t != 0:
                    clean_tokens.append(t)
            
            return tokenizer.decode(clean_tokens).strip()

        # 1. Full Multi-Modal Forward Pass
        full_caption = generate_caption_ablation(iframe, motion, residual, audio)
        
        # 2. I-Frame Only Forward Pass (Zero out Motion, Residual, Audio)
        b, t = iframe.shape[:2]
        zero_motion = torch.zeros_like(motion)
        zero_residual = torch.zeros_like(residual)
        zero_audio = torch.zeros_like(audio)
        
        iframe_only_caption = generate_caption_ablation(iframe, zero_motion, zero_residual, zero_audio)
        
        gt_metadata = dataset.captions_data[vid_id]
        gt_captions = []
        if isinstance(gt_metadata, dict):
            for k, v in gt_metadata.items():
                if isinstance(v, list):
                    gt_captions.extend(v)
        else:
             gt_captions = [gt_metadata]
        
        print("\n" + "="*80)
        print(f"Video ID: {vid_id}")
        print("-" * 40)
        print(f"Ground Truth (first 2): {gt_captions[:2]}")
        print(f"I-Frame ONLY Generated : {iframe_only_caption}")
        print(f"Full Model Generated   : {full_caption}")

    print("\n" + "="*80)
    print("Ablation testing complete!")
