Based on a thorough search of the codebase operations and hydra configuration files, here are the exact hyperparameters and architectures being used:

1. Resampling Hyperparameters and Video Resolution
All three datasets (MSVD, MSRVTT, and VATEX) share uniform sampling rules defined in their respective YAML configurations (CoCap/configs/dataset/...):

resample_num_gop: 8
resample_num_mv: 59
resample_num_res: 59
Video Resolution Handling:

Source Files: Handled dataset strings indicate videos were pre-processed (or downscaled) externally to keep the shortest spatial dimension to 240 pixels (e.g., videos_240_h264_keyint_60).
Passed to Model: The model specifically enforces a video_size of [224, 224]. The frames are centrally cropped or resized to 224x224 in the dataloader logic before arriving at the vision encoders. For motion vectors, the inputs are sized down accordingly due to pooling grids (56x56 for AVC vectors).
(Code Ref: msvd.yaml, msrvtt.yaml, vatex.yaml)

2. ViT Variants Used for Respective Encoders
The pipeline hinges on the ViT-Base architecture (specifically ViT-B/16) mapped from OpenAI's CLIP, but heavily customizes parameter scaling for motion and residuals. This is defined in compressed_video_transformer.py under the from_pretrained instantiation method:

I-Frame Encoder (RGB):

Architecture: Pre-trained ViT-B/16 (CLIP)
Configuration: 12 Layers, 12 Heads, 768 Width (Hidden Size)
Image Patch Size: 16x16
Behavior: Directly inherits pre-trained visual dictionary states from the pretrained_clip_name_or_path: str = "ViT-B/16" default fallback.
Motion Vector Encoder (MV):

Architecture: Extremely truncated / slim custom ViT ("ViT-Tiny/Micro" adjacent)
Configuration: 2 Layers, 8 Heads, 192 Width (vision_width // 4)
Image Patch Size: 8x8
Input Resolution: 56x56 (image_resolution // 4), feeding into 4 input channels.
Behavior: It initializes from scratch without pre-trained weights.
Residual Encoder:

Architecture: Shallow ViT-Base
Configuration: 2 Layers, 8 Heads, 768 Width (vision_width)
Image Patch Size: 64x64
Input Resolution: 224x224 feeding into 3 channels.
Behavior: Operates heavily strided (massive patches) using only 2 layers. Initialized from scratch.
Action Encoder (Temporal Context):

Architecture: Very shallow sequence Transformer
Configuration: 1 Layer, 8 Heads, hidden width inherits the projection output dimension (typically 512).
(Code Ref: CoCap/cocap/modules/compressed_video/compressed_video_transformer.py#L90-L105)