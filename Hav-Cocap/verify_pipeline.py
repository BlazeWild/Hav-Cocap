
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add paths
sys.path.append(os.getcwd())
# Add CoCap path (redundant if handled in modules, but safe)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../CoCap")))

from model.hav_cocap import HavCocapModel, HavCocapCaptioner
from DataLoader.dataloader import HavCocapDataset
from cocap.modules.compressed_video.compressed_video_captioner import CaptionHead


def test_pipeline():
    print("Testing Hav-Cocap Captioning Pipeline...")
    
    # Init Models
    print("Initializing Models...")
    try:
        base_model = HavCocapModel(embed_dim=512)
        # Initialize CaptionHead with default config (CLIP ViT-B/16 dims)
        # embed_dim=512, vocab_size=49408
        caption_head = CaptionHead(
            word_embedding_size=512,
            visual_feature_size=512, # CoCap typically projects to 512
            max_v_len=16, # 8 GOPs * 2 (ctx+act) ? CoCap uses 8*2?
            max_t_len=77,
            hidden_size=512,
            vocab_size=49408,
            verbose=False
        )
        
        model = HavCocapCaptioner(base_model, caption_head)
        print("Model Initialized.")
        
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Dataset
    data_root = os.path.abspath("dataset/AVCaps")
    print(f"Loading Dataset from {data_root}")
    
    dataset = HavCocapDataset(data_root, split="val")
    dataloader = DataLoader(dataset, batch_size=2)
    
    print("Iterating...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        
        # Move to keys expected by forward
        # DataLoader collates into stacked tensors
        # Check shapes
        iframe = batch['iframe']
        motion = batch['motion'] # Currently None in initial design, need zeros?
        residual = batch['residual']
        audio = batch['audio']
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        
        print(f"  I-Frame: {iframe.shape}")
        print(f"  Audio: {audio.shape}")
        print(f"  Input IDs: {input_ids.shape}")
        
        # Prepare inputs for model
        inputs = {
            "iframe": iframe,
            "motion": motion,
            "residual": residual,
            "audio": audio,
            "input_ids": input_ids,
            "input_mask": input_mask
        }
        
        # Forward
        try:
            output = model(inputs)
            print("  Forward Pass Successful.")
            scores = output['prediction_scores']
            visual = output['visual_output']
            
            print(f"  Prediction Specs: {scores.shape}") # (B, T, Vocab)
            # Decode a prediction?
            # Greedy decode from scores
            pred_ids = torch.argmax(scores, dim=-1)
            # Need tokenizer to decode? Or CaptionHead helper?
            # CaptionHead has ids2text but it relies on 'cocap.trainer.cocap_trainer' which might crash.
            # We can use dataset.tokenizer.decode manually if available.
            
            if hasattr(dataset, 'tokenizer') and dataset.tokenizer:
                print(f"  Ground Truth: {dataset.tokenizer.decode(input_ids[0].tolist())}")
                print(f"  Prediction (Untrained): {dataset.tokenizer.decode(pred_ids[0].tolist())}")
                
        except Exception as e:
            print(f"  Forward Failed: {e}")
            import traceback
            traceback.print_exc()
            
        if i >= 1: break # Test 2 batches
        
if __name__ == "__main__":
    test_pipeline()
