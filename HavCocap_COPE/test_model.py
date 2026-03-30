import torch
from havcocap_new.modules.compressed_video.compressed_video_transformer import CompressedVideoTransformer, IFrameEncoder, MotionCompressor
from havcocap_new.modules.compressed_video.compressed_video_captioner import CompressedVideoCaptioner

def test():
    # 1. Instantiate compressed video transformer
    print("Loading models...")
    # we don't need real weights for just shape testing
    rgb_encoder, _, _, embed_dim = IFrameEncoder.from_pretrained("ViT-B/16")
    motion_encoder = MotionCompressor(input_dim=256, hidden_dim=1024, embed_dim=embed_dim, num_queries=4, num_heads=8)
    
    cv_transformer = CompressedVideoTransformer(rgb_encoder=rgb_encoder, motion_encoder=motion_encoder, output_dim=embed_dim)
    
    captioner = CompressedVideoCaptioner(compressed_video_transformer=cv_transformer, tinystories_model_path="model_zoo/tinystories-33m")
    
    # 2. Forge dummy inputs
    bsz = 2
    n_gop = 8
    n_mv = 7
    iframe = torch.randn(bsz, n_gop, 3, 224, 224)
    # motion shape: [batch, n_gop, n_mv, 4, 56, 56]
    motion = torch.randn(bsz, n_gop, n_mv, 4, 56, 56)
    
    input_ids = torch.randint(0, 1000, (bsz, 20))
    input_mask = torch.ones(bsz, 20, dtype=torch.long)
    
    inputs = {
        "video": {
            "iframe": iframe,
            "motion_vector": motion
        },
        "input_ids": input_ids,
        "input_mask": input_mask,
        "input_labels": input_ids.clone()
    }
    
    print("Running forward pass...")
    outputs = captioner(inputs)
    print("LM Loss:", outputs["lm_loss"].item())
    print("Aux Loss:", outputs["aux_loss"].item())
    print("Prediction Scores shape:", outputs["prediction_scores"].shape)
    print("Visual Seq shape:", outputs["visual_output"]["visual_seq"].shape)
    print("Test passed successfully!")

if __name__ == "__main__":
    test()
