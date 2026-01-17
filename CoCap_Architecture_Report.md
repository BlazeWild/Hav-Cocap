# CoCap Architecture Report

## 1. Overview
CoCap (Accurate and Fast Compressed Video Captioning) is an end-to-end framework designed to generate captions directly from compressed video bitstreams. Instead of fully decoding the video to RGB frames, it leverages the inherent structure of H.264/AVC encoding—specifically I-frames, Motion Vectors, and Residuals—to extract visual features efficiently. This report details the architectural mechanisms that enable batching, variable-length handling, and multimodal fusion.

## 2. Module Inputs and Dimensions

The framework utilizes specialized encoders for different compressed domain components. Below is the summary of inputs for a video batch of shape $(B, T, M, \dots)$, where:
- $B$: Batch size
- $T$: Number of Groups of Pictures (GOPs)
- $M$: Number of P/B frames sampled per GOP
- $H, W$: Video resolution (typically $224 \times 224$ pixels)
- $D$: Embedding dimension

### Summary Table

| Module | Input Tensor Shape | Description |
| :--- | :--- | :--- |
| **I-Frame Encoder** | $(B, T, 3, H, W)$ | Processes the full RGB I-frame (Intra-coded) for each GOP. Provides the static visual context. |
| **Motion Encoder** | $(B, T, M, 4, H/4, W/4)$ | Processes Motion Vectors (MV). Input resolution is $1/4$ of the RGB frame. Contains 4 channels (likely horizontal/vertical components for forward/backward prediction). |
| **Residual Encoder** | $(B, T, M, 3, H, W)$ | Processes the error residuals (differences between predicted and actual frames). Same resolution as I-frames. |
| **Action Encoder** | $(B \cdot T, M, D)$ | Aggregates the sequence of $M$ motion/residual embeddings into a single action representation. |

### Significance of Resolutions
- **Pixel Resolution ($H \times W$):** Used for I-frames and Residuals. These contain detailed visual textures and fine-grained error corrections that require full spatial fidelity (e.g., $224 \times 224$).
- **Macroblock Resolution ($H/4 \times W/4$):** Used for Motion Vectors. In H.264, the standard macroblock size is $16 \times 16$ pixels, but blocks can be partitioned down to $4 \times 4$ pixels. The CoCap Motion Encoder operates on a $56 \times 56$ grid (assuming $H=224$), which corresponds to a $4 \times 4$ pixel granularity. This "coarse" resolution suffices for motion as it represents block-wise displacement rather than pixel-wise color, offering computational efficiency.

## 3. Handling Variable Lengths

Video data inherently varies in length (duration) and structure (GOP size). CoCap maintains a batchable shape through a **Sampling and Padding** strategy implemented in the data loader (`read_frames_compressed_domain`):

1.  **GOP Sampling ($T$):** The loader uniformly or randomly samples a fixed number of GOPs (e.g., $T=8$) from the video. If a video has fewer GOPs, it is padded (though random sampling is preferred for training).
2.  **Inner-GOP Sampling ($M$):** Within each GOP, a fixed number of P/B frames (e.g., $M$) are sampled to extract Motion Vectors and Residuals.
3.  **Result:** This process guarantees that every video in a batch produces tensors of identical shape $(B, T, \dots)$ and $(B, T, M, \dots)$, eliminating the need for complex variable-length masking during the Vision Transformer encoding stage. The network treats the sampled frames as a dense representation of the video.

## 4. The Batching Mechanism ("Flattening")

To maximize GPU utilization and leverage standard 2D Vision Transformers (like CLIP-ViT), CoCap employs a "Flattening" strategy. It reshapes the higher-dimensional video tensors into 4D tensors that standard image encoders can process in parallel.

For a 5D/6D input video tensor, the dimensions are collapsed into the batch dimension:

-   **I-Frame Flattening:**
    -   Input: $(B, T, 3, H, W)$
    -   Operation: `rearrange(iframe, "b t c h w -> (b t) c h w")`
    -   Result: A "super-batch" of $B \times T$ independent images.

-   **Motion/Residual Flattening:**
    -   Input: $(B, T, M, C, H, W)$
    -   Operation: `rearrange(motion, "b t m c h w -> (b t m) c h w")`
    -   Result: A "super-batch" of $B \times T \times M$ independent maps.

The Encoder processes this massive batch in one go. After encoding, the output features are reshaped (un-flattened) back to $(B, T, D)$ or $(B, T, M, D)$ to restore the temporal structure.

## 5. Temporal Aggregation (Action Encoder)

The **Action Encoder** is responsible for collapsing the detailed motion information from $M$ frames into a single meaningful "Action Token" for the GOP.

1.  **Fusion:** First, the Motion embedding ($MV_{cls}$) and Residual embedding ($Res_{cls}$) for each of the $M$ frames are summed: $F_{bp} = MV_{cls} + Res_{cls}$.
2.  **Cross-Attention:** The sequence of $M$ embeddings undergoes Transformer layers. Crucially, it employs **Cross-Attention** to the I-frame spatial features ($F_{ctx, all\_hidden}$) of the corresponding GOP. This allows the motion features to be contextualized by the static scene content.
3.  **Aggregation:** The encoder applies a mean pooling operation over the $M$ temporal tokens:
    $$ \text{ActionToken}_{gop} = \text{MeanPool}_{m=1}^{M}( \text{Transformer}(F_{bp}) ) $$
4.  **Output:** A single vector of dimension $D$ (768) per GOP.

## 6. Multimodal Decoder Input

The final input sent to the Multimodal Decoder (BERT-based Caption Head) is a concatenation of the static context and the dynamic action for each GOP.

-   **Context Tokens:** $(B, T, D)$ from I-frames.
-   **Action Tokens:** $(B, T, D)$ from the Action Encoder.
-   **Combined Sequence:** The two tensors are concatenated along the time dimension (dim 1), resulting in a sequence of length $2T$.
    $$ \text{VisualOutput} \in \mathbb{R}^{B \times 2T \times D} $$
-   **Input Types:** The decoder receives this visual sequence along with type embeddings to distinguish between "Context" tokens (Type 1) and "Action" tokens (Type 0).

## 7. Data Flow Summary

1.  **Raw Bitstream** $\rightarrow$ **Sampler** $\rightarrow$ Fixed tensors for I-frames ($B, T, \dots$) and P/B-frames ($B, T, M, \dots$).
2.  **Flattening** $\rightarrow$ Merge $B, T, M$ dimensions.
3.  **Encoders** $\rightarrow$
    -   ViT-B/16 (I-frames) $\rightarrow$ Context Tokens.
    -   ViT-Small (Motion/Residuals) $\rightarrow$ Dense Motion Features.
4.  **Action Encoder** $\rightarrow$ Fuse Motion+Residuals, attend to I-frame, pool $M$ frames $\rightarrow$ Action Tokens.
5.  **Concatenation** $\rightarrow$ Sequence of $\{ \text{Context}_1, \dots, \text{Context}_T, \text{Action}_1, \dots, \text{Action}_T \}$.
6.  **Decoder** $\rightarrow$ BERT Cross-Attention over visual tokens $\rightarrow$ Autoregressive generation of [EOS] caption token.
