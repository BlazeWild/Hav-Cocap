# Hav-Cocap (Hybrid Audio-Visual CoCap)

**Multimodal Compressed Video Captioning with Audio Integration**

This project is an extension of the **CoCap** (Accurate and Fast Compressed Video Captioning) architecture. We have integrated an **Audio Encoder** to leverage multimodal information (Visual + Audio) for richer caption generation and evaluated the performance on the **AVCaps** dataset.

## Original Work & Citation

The core architecture is based on the paper and code from:
- **Repository**: [https://github.com/Yaojie-Shen/CoCap](https://github.com/Yaojie-Shen/CoCap)
- **Paper**: *CoCap: Accurate and Fast Compressed Video Captioning*

We acknowledge and thank the original authors (Yaojie Shen et al.) for their foundational work.

## Our Contributions: Audio Integration

While the original CoCap focuses on efficient visual feature extraction from H.264/AVC compressed streams (I-frames, Motion Vectors, Residuals), **Hav-Cocap** introduces an additional modality:

-   **Audio Encoder**: We integrated an audio processing module to capture acoustic events and speech, which are critical for comprehensive video understanding.
-   **AVCaps Dataset**: The model has been adapted and trained/tested on the **AVCaps** dataset, demonstrating the effectiveness of combining compressed visual features with audio embeddings.

## Overview

Hav-Cocap is an end-to-end framework designed to generate captions directly from compressed video bitstreams. Instead of fully decoding the video to RGB frames, it leverages the inherent structure of H.264/AVC encoding to extract visual features efficiently.

## Architecture

The architecture consists of several specialized encoders and a multimodal fusion mechanism:

1.  **I-Frame Encoder (ViT-B/16)**: Processes full RGB I-frames to capture static visual context (objects, scenes).
2.  **Motion Encoder (ViT-Small)**: Processes Motion Vectors (MV) to capture temporal dynamics.
3.  **Residual Encoder (ViT-Small)**: Processes error residuals to refine motion estimation.
4.  **Audio Encoder**: Processes the audio track to extract sound features.
5.  **Action Encoder**: Aggregates motion and residual embeddings, contextualized by I-frame features via Cross-Attention.
6.  **Multimodal Decoder**: A BERT-based captioning head that attends to the sequence of visual (Context + Action) and Audio tokens to generate natural language descriptions.

### Efficient Batching
CoCap employs a flattened batching strategy to process variable-length videos efficiently. By sampling a fixed number of GOPs and inner-GOP frames, it treats the video as a dense representation, allowing for standard 2D ViT processing on "super-batches".

## Directory Structure

- **`CoCap/`**: Main source code for the framework.
    - **`cocap/modeling/`**: Model definitions (Encoders, Decoders, Loss functions).
    - **`cocap/data/`**: Data loaders and dataset implementations (MSR-VTT, MSVD, VATEX, **AVCaps**).
    - **`cocap/modules/`**: Core modules including CLIP, BERT, and compressed video transformers.
- **`Havcocap/`**: Additional project components.

## Setup and Usage

*(Note: Detailed setup instructions to be added. Ensure you have the necessary dependencies installed.)*

1.  **Install Dependencies**: Use `pip` to install requirements.
2.  **Data Preparation**: Ensure your video data is compatible (H.264/AVC compressed) and prepare the AVCaps dataset.
3.  **Training/Inference**: Refer to scripts in `CoCap/cocap/` for running experiments.
