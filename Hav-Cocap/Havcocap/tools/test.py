import os
import sys
import torch
import argparse
import json
from tqdm import tqdm
import logging
import numpy as np

# Add paths to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Havcocap.model.hav_cocap import HavCocapModel, HavCocapCaptioner
from DataLoader.dataloader import HavCocapDataset
from Havcocap.modules.compressed_video.compressed_video_captioner import CaptionHead
from Havcocap.utils.logging import setup_logging
from Havcocap.modeling.eval_captioning import EvalCap, PTBTokenizer

logger = logging.getLogger(__name__)

def generate_caption(model, inputs, tokenizer, max_t_len=77, device="cuda"):
    """
    Greedy decoding loop for generation.
    """
    # 1. Unpack inputs and extract visual features once
    iframe = inputs['iframe'].to(device, non_blocking=True)
    audio = inputs['audio'].to(device, non_blocking=True)
    
    # Handle Motion/Residual placeholders
    if inputs['motion'] is not None and torch.is_tensor(inputs['motion']):
        motion = inputs['motion'].to(device, non_blocking=True)
        if motion.shape[3] == 2:
            padded = torch.zeros(motion.shape[0], motion.shape[1], motion.shape[2], 4, motion.shape[4], motion.shape[5], device=device)
            padded[:, :, :, :2, :, :] = motion
            motion = padded
    else:
        b, t = iframe.shape[:2]
        motion = torch.zeros(b, t, 1, 4, 56, 56, device=device)

    if inputs['residual'] is not None and torch.is_tensor(inputs['residual']):
        residual = inputs['residual'].to(device, non_blocking=True)
    else:
        b, t = iframe.shape[:2]
        residual = torch.zeros(b, t, 1, 3, 224, 224, device=device)
        
    b_size = iframe.size(0)

    # Prepare inputs dict
    model_inputs = {
        "iframe": iframe,
        "motion": motion,
        "residual": residual,
        "audio": audio,
    }

    # Forward the base model to get visual/audio features
    # (Forwarding the HavCocapModel directly as HavCocapCaptioner handles captioning)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            visual_output = model.hav_cocap_model(
                iframe=model_inputs["iframe"],
                motion=model_inputs["motion"],
                residual=model_inputs["residual"],
                audio=model_inputs["audio"],
                bp_type_ids=torch.zeros((b_size, iframe.size(1), 1), dtype=torch.long, device=device) # default bp_type_ids
            )
                
            # Start token
            sot_token_id = tokenizer.encoder["<|startoftext|>"]
            eot_token_id = tokenizer.encoder["<|endoftext|>"]
            
            # Initialize generated sequence with SOT
            input_ids = torch.full((b_size, 1), sot_token_id, dtype=torch.long, device=device)
            
            finished = torch.zeros(b_size, dtype=torch.bool, device=device)
            
            # Autoregressive generation loop
            for step in range(max_t_len - 1): # -1 because we already have SOT
                # Create mask for the current sequence
                current_len = input_ids.size(1)
                input_mask = torch.ones(b_size, current_len, dtype=torch.long, device=device)
                
                # Pad sequence to max_t_len as required by BERT caption head
                # It expects fixed size input_ids (B, max_t_len)
                padded_input_ids = torch.zeros(b_size, max_t_len, dtype=torch.long, device=device)
                padded_input_ids[:, :current_len] = input_ids
                
                padded_input_mask = torch.zeros(b_size, max_t_len, dtype=torch.long, device=device)
                padded_input_mask[:, :current_len] = input_mask
                
                # Forward caption head
                prediction_scores = model.caption_head(
                    visual_output=visual_output,
                    input_ids=padded_input_ids,
                    input_mask=padded_input_mask
                )
                
                # Get logits for the LAST generated token (current_len - 1)
                # Prediction scores shape: (B, max_t_len, Vocab)
                next_token_logits = prediction_scores[:, current_len - 1, :].clone()
                
                # Prevent predicting PAD token (index 0)
                next_token_logits[:, 0] = float('-inf')
                
                # Greedy choice
                next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                
                # Check if all batches generated EOT
                is_eot = (next_tokens.squeeze(-1) == eot_token_id)
                finished |= is_eot
                
                if finished.all():
                    break
                    
    # Decode sequences
    generated_texts = []
    for i in range(b_size):
        tokens = input_ids[i].tolist()
        # Remove SOT, pad, and anything after EOT
        clean_tokens = []
        for t in tokens:
            if t == eot_token_id:
                break
            if t != sot_token_id and t != 0: # 0 is PAD
                clean_tokens.append(t)
        
        text = tokenizer.decode(clean_tokens)
        generated_texts.append(text.strip())
        
    return generated_texts

def evaluate(args):
    logger = setup_logging(output_dir=os.path.dirname(args.output_file))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on {device}")
    
    # 1. Dataset
    logger.info(f"Loading test dataset from {args.data_root}")
    test_dataset = HavCocapDataset(
        args.data_root, 
        split="test", 
        blacklist_file=args.blacklist_file,
    )
    
    tokenizer = test_dataset.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer not found. Cannot perform generation.")
        
    # Test loader (1 worker to ensure orderly processing for simplicity, can increase)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Model
    logger.info("Initializing Model...")
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
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Checkpoint loaded successfully.")
    
    # 3. Generation Loop
    results = [] # list of {"image_id": "vid123", "caption": "..."}
    
    logger.info("Starting generation...")
    for batch in tqdm(test_loader, desc="Generating Captions"):
        video_ids = batch['video_id']
        
        # Generate captions
        generated_captions = generate_caption(model, batch, tokenizer, max_t_len=77, device=device)
        
        for vid_id, cap in zip(video_ids, generated_captions):
            results.append({
                "image_id": vid_id,
                "caption": cap
            })
            
    # Save results
    logger.info(f"Saving generated captions to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
         json.dump({"results": {r["image_id"]: [{"sentence": r["caption"]}] for r in results}}, f, indent=4)
         
    # 4. Evaluation
    logger.info("Computing metrics...")
    
    # Load Ground Truth
    gt_file = os.path.join(args.data_root, "test", "test_captions.json")
    if not os.path.exists(gt_file):
        logger.warning(f"Ground truth file not found at {gt_file}. Cannot compute metrics.")
        return
        
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    # Format GT for pycocoevalcap
    # Expected: {image_id: [caption1, caption2, ...]}
    formatted_gt = {}
    for vid_id, data in gt_data.items():
        caps = []
        if isinstance(data, dict):
            if "audio_captions" in data: caps.extend([c.strip() for c in data["audio_captions"] if c.strip()])
            if "visual_captions" in data: caps.extend([c.strip() for c in data["visual_captions"] if c.strip()])
        elif isinstance(data, list):
            caps.extend([c.strip() for c in data if c.strip()])
        elif isinstance(data, str):
            caps.append(data.strip())
            
        if caps:
            formatted_gt[vid_id] = caps
            
    # Format predictions for EvalCap
    # Expected rests mapping: [ {"image_id": "1", "caption": "cap"}, ... ]
    formatted_preds = [{"image_id": r["image_id"], "caption": r["caption"]} for r in results]
    
    evaluator = EvalCap(formatted_gt, formatted_preds, PTBTokenizer)
    evaluator.evaluate()
    
    logger.info("-----------------------------")
    logger.info("Evaluation Results:")
    for metric, score in evaluator.eval.items():
        logger.info(f"{metric}: {score:.4f}")
    logger.info("-----------------------------")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Hav-Cocap/dataset/AVCaps")
    parser.add_argument("--checkpoint", type=str, default="Hav-Cocap/checkpoints/checkpoint_epoch_100.pth", help="Path to evaluation checkpoint")
    parser.add_argument("--blacklist_file", type=str, default="Hav-Cocap/corrupt_files.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default="Hav-Cocap/test_results.json")
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    evaluate(args)
