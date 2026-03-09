import os
import sys
import torch
from torchinfo import summary

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Havcocap.model.hav_cocap import HavCocapModel, HavCocapCaptioner
from Havcocap.modules.compressed_video.compressed_video_captioner import CaptionHead

def main():
    print("Initializing Model...")
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
    
    model = HavCocapCaptioner(base_model, caption_head)
    
    # We need to provide dummy inputs to torchinfo.summary
    # Let's check the forward pass of HavCocapCaptioner
    # forward(self, inputs)
    # inputs: dict with 'video', 'audio', 'input_ids', 'input_mask'
    
    # Let's just use summary(model, depth=5) without input data if possible, 
    # or we can just iterate through named_parameters to list learnable/non-learnable.
    
    # torchinfo.summary can take input_data or input_size.
    # Since the input is a complex dictionary, it might be easier to just pass a dummy dict.
    
    batch_size = 2
    n_gop = 8
    
    dummy_inputs = {
        "video": {
            "iframe": torch.randn(batch_size, n_gop, 3, 224, 224),
            "motion_vector": torch.randn(batch_size, n_gop, 4, 2, 56, 56), # Assuming 4 MVs per GOP, 2 channels (x,y)
            "residual": torch.randn(batch_size, n_gop, 4, 3, 224, 224), # Assuming 4 residuals per GOP
            "type_ids_mv": torch.zeros(batch_size, n_gop, 4, dtype=torch.long)
        },
        "audio": torch.randn(batch_size, 16000 * 10), # 10 seconds of audio at 16kHz
        "input_ids": torch.randint(0, 49408, (batch_size, 77)),
        "input_mask": torch.ones(batch_size, 77, dtype=torch.long)
    }
    
    # Let's check the exact shapes expected by HavCocapModel
    
    try:
        model_stats = summary(
            model, 
            input_data=[dummy_inputs], 
            depth=5, 
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0
        )
        summary_str = str(model_stats)
    except Exception as e:
        print(f"Failed to run torchinfo.summary with dummy inputs: {e}")
        print("Running summary without inputs...")
        model_stats = summary(
            model, 
            depth=5, 
            col_names=("num_params", "trainable"),
            verbose=0
        )
        summary_str = str(model_stats)

    # Also list what are learnable and what are not
    learnable_params = []
    non_learnable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            learnable_params.append(name)
        else:
            non_learnable_params.append(name)
            
    output_file = os.path.join(os.path.dirname(__file__), "model_summary.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("================ MODEL SUMMARY ================\n")
        f.write(summary_str)
        f.write("\n\n================ LEARNABLE PARAMETERS ================\n")
        for name in learnable_params:
            f.write(f"{name}\n")
        f.write("\n================ NON-LEARNABLE PARAMETERS ================\n")
        for name in non_learnable_params:
            f.write(f"{name}\n")
            
    print(f"Model summary saved to {output_file}")

if __name__ == "__main__":
    main()
