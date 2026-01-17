# Hav-Cocap (CoCap)

**Accurate and Fast Compressed Video Captioning**

## Overview

Hav-Cocap (CoCap) is an end-to-end framework designed to generate captions directly from compressed video bitstreams. Instead of fully decoding the video to RGB frames, it leverages the inherent structure of H.264/AVC encoding—specifically I-frames, Motion Vectors, and Residuals—to extract visual features efficiently. 

This approach significantly reduces computational overhead by processing the "coarse" motion information directly from the compressed domain while maintaining high accuracy by fusing it with static visual context from I-frames.

## Architecture

The architecture consists of several specialized encoders and a multimodal fusion mechanism:

1.  **I-Frame Encoder (ViT-B/16)**: Processes full RGB I-frames to capture static visual context (objects, scenes).
2.  **Motion Encoder (ViT-Small)**: Processes Motion Vectors (MV) to capture temporal dynamics.
3.  **Residual Encoder (ViT-Small)**: Processes error residuals to refine motion estimation.
4.  **Action Encoder**: Aggregates motion and residual embeddings, contextualized by I-frame features via Cross-Attention, to produce a compact "Action Token" for each Group of Pictures (GOP).
5.  **Multimodal Decoder**: A BERT-based captioning head that attends to the sequence of Context (I-frame) and Action tokens to generate natural language descriptions.

### Efficient Batching
CoCap employs a flattened batching strategy to process variable-length videos efficiently. By sampling a fixed number of GOPs and inner-GOP frames, it treats the video as a dense representation, allowing for standard 2D ViT processing on "super-batches" before restoring temporal structure for the final caption generation.

## Directory Structure

- **`CoCap/`**: Main source code for the framework.
    - **`cocap/modeling/`**: Model definitions (Encoders, Decoders, Loss functions).
    - **`cocap/data/`**: Data loaders and dataset implementations (MSR-VTT, MSVD, VATEX).
    - **`cocap/modules/`**: Core modules including CLIP, BERT, and compressed video transformers.
- **`Havcocap/`**: (Additional project files/scripts if any).

## Setup and Usage

*(Note: Detailed setup instructions to be added. Ensure you have the necessary dependencies installed.)*

1.  **Install Dependencies**: Use `pip` to install requirements (check `requirements.txt` if available).
2.  **Data Preparation**: Ensure your video data is compatible (H.264/AVC compressed).
3.  **Training/Inference**: Refer to scripts in `CoCap/cocap/` for running experiments.

## Acknowledgements

This project builds upon research in compressed video understanding and efficient video captioning.
