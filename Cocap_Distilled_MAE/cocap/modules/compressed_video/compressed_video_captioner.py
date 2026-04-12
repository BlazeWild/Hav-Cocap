# -*- coding: utf-8 -*-
# @Time    : 8/6/23
# @Author  : Yaojie Shen (Updated for HavCocap GPT-2 136-Token Architecture)
# @Project : CoCap
# @File    : compressed_video_captioner.py

__all__ = [
    "CompressedVideoCaptioner",
    "compressed_video_captioner_cfg",
    "compressed_video_captioner_pretrained_cfg",
]

import logging
from typing import *

import torch
from torch import Tensor
from torch import nn
from hydra_zen import builds

from cocap.modules.hav_cocap_gpt2 import HavCoCapGPT2
from cocap.modules.compressed_video.compressed_video_transformer import CompressedVideoTransformer, \
    compressed_video_transformer_pretrained_cfg, compressed_video_transformer_cfg

logger = logging.getLogger(__name__)

class CompressedVideoCaptioner(nn.Module):
    def __init__(
            self,
            compressed_video_transformer: CompressedVideoTransformer,
            gpt2_model_path: str = "model_zoo/gpt2_model",
            motion_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.compressed_video_transformer = compressed_video_transformer
        self.dropout_motion = nn.Dropout(motion_dropout_prob)

        # 1. Initialize GPT-2 Decoder
        self.gpt2 = HavCoCapGPT2.from_pretrained(gpt2_model_path)
        
        # Determine the hidden dimensions
        gpt_dim = self.gpt2.config.hidden_size # 768
        cv_dim = self.compressed_video_transformer.output_dim # 768 (from CLIP ViT-B)
        
        # ==========================================================
        # THE PROJECTION LAYERS (The "Translators")
        # No residuals. Just raw linear translation.
        # ==========================================================
        self.proj_spatial = nn.Linear(cv_dim, gpt_dim)
        
        # Motion queries have their own projection to map them to text space
        if hasattr(self.compressed_video_transformer.motion_encoder, 'output_dim'):
            mv_width = self.compressed_video_transformer.motion_encoder.output_dim
            self.proj_motion = nn.Linear(mv_width, gpt_dim)
        else:
            self.proj_motion = nn.Linear(cv_dim, gpt_dim)

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]]):
        """
        Phase 2 Forward Pass:
        Extracts visual tokens via MGDTR, builds the 136-token Visual Prefix, and feeds it to GPT-2.
        """
        # --- 1. VISUAL EXTRACTION ---
        if "visual_output" not in inputs:
            iframe = inputs["video"]["iframe"]
            motion = self.dropout_motion(inputs["video"]["motion_vector"])
            input_mask_gop = inputs["video"]["input_mask_gop"]
            input_mask_mv = inputs["video"]["input_mask_mv"]

            # Process through the backbone and MGDTR
            compressed_visual_features = self.compressed_video_transformer(
                iframe=iframe, 
                motion=motion, 
                residual=torch.zeros_like(iframe), # Dummy to satisfy signature
                bp_type_ids=torch.zeros((iframe.size(0), iframe.size(1), motion.size(2)), dtype=torch.long, device=iframe.device),
                input_mask_gop=input_mask_gop,
                input_mask_mv=input_mask_mv
            )
        else:
            compressed_visual_features = inputs["visual_output"]

        # --- 2. TRANSLATE VISUALS TO GPT-2 SPACE ---
        # A. I-Frame Context [B, 8, 768]
        f_ctx_proj = self.proj_spatial(compressed_visual_features["feature_context"])
        
        # B. MGDTR Routed Spatial Patches [B, 64, 768]
        f_spatial_proj = self.proj_spatial(compressed_visual_features["routed_spatial_tokens"])
        
        # C. Motion Queries [B, 8, 8, 256] -> Flatten to [B, 64, 256] -> Translate to [B, 64, 768]
        f_mot = compressed_visual_features["feature_motion"]
        B, n_gop, num_t, mv_width = f_mot.shape # num_t is exactly 8 from your MotionStudent
        f_mot_flat = f_mot.reshape(B, n_gop * num_t, mv_width) 
        f_mot_proj = self.proj_motion(f_mot_flat) # [B, 64, 768]
        
        # ORDER: Context (8) -> Spatial Details (64) -> Temporal Motion (64)
        visual_prompt = torch.cat([f_ctx_proj, f_spatial_proj, f_mot_proj], dim=1) # [B, 136, 768]
        visual_mask = torch.ones((B, visual_prompt.size(1)), dtype=torch.long, device=visual_prompt.device)

        # --- 3. CONCATENATE WITH TEXT & BOS TOKEN ---
        text_ids = inputs["input_ids"] # [B, max_words] (Starts with <BOS>)
        text_mask = inputs["input_mask"]
        text_labels = inputs["input_labels"]
        
        # Grab GPT-2's internal word embedding matrix to translate IDs into embeddings
        text_embeds = self.gpt2.transformer.wte(text_ids) # [B, max_words, 768]
        
        # Sequence: [Visual Tokens (136)] + [<BOS> + Text Tokens (max_words)]
        full_embeds = torch.cat([visual_prompt, text_embeds], dim=1)
        full_mask = torch.cat([visual_mask, text_mask], dim=1)
        
        # For the labels, pad the 136 visual tokens with -100 to prevent calculating text loss on video frames
        visual_labels = torch.full((B, visual_prompt.size(1)), -100, dtype=torch.long, device=text_labels.device)
        full_labels = torch.cat([visual_labels, text_labels], dim=1)

        # --- 4. GPT-2 FORWARD PASS ---
        outputs = self.gpt2(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels
        )
        
        return {
            "loss": outputs.loss,
            "prediction_scores": outputs.logits, 
            "visual_output": compressed_visual_features
        }

    @torch.no_grad()
    def generate(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]], max_new_tokens: int = 20):
        """
        Called during validation/testing to auto-regressively generate captions.
        """
        iframe = inputs["video"]["iframe"]
        motion = inputs["video"]["motion_vector"]
        input_mask_gop = inputs["video"]["input_mask_gop"]
        input_mask_mv = inputs["video"]["input_mask_mv"]

        visual_features = self.compressed_video_transformer(
            iframe=iframe, 
            motion=motion, 
            residual=torch.zeros_like(iframe), 
            bp_type_ids=torch.zeros((iframe.size(0), iframe.size(1), motion.size(2)), dtype=torch.long, device=iframe.device),
            input_mask_gop=input_mask_gop,
            input_mask_mv=input_mask_mv
        )
        
        f_ctx_proj = self.proj_spatial(visual_features["feature_context"])
        f_spatial_proj = self.proj_spatial(visual_features["routed_spatial_tokens"])
        
        f_mot = visual_features["feature_motion"]
        B, n_gop, num_t, mv_width = f_mot.shape
        f_mot_proj = self.proj_motion(f_mot.reshape(B, n_gop * num_t, mv_width))
        
        visual_prompt = torch.cat([f_ctx_proj, f_spatial_proj, f_mot_proj], dim=1)
        visual_attention_mask = torch.ones((B, visual_prompt.size(1)), dtype=torch.long, device=visual_prompt.device)
        
        generated_ids = self.gpt2.generate(
            inputs_embeds=visual_prompt,
            attention_mask=visual_attention_mask,
            max_new_tokens=max_new_tokens,
            bos_token_id=self.gpt2.config.bos_token_id,
            eos_token_id=self.gpt2.config.eos_token_id,
            pad_token_id=self.gpt2.config.eos_token_id
        )
        
        return generated_ids

# Build configs for organizing modules with hydra
compressed_video_captioner_cfg = builds(
    CompressedVideoCaptioner,
    compressed_video_transformer=compressed_video_transformer_cfg,
    populate_full_signature=True
)
compressed_video_captioner_pretrained_cfg = builds(
    CompressedVideoCaptioner,
    compressed_video_transformer=compressed_video_transformer_pretrained_cfg,
    populate_full_signature=True
)