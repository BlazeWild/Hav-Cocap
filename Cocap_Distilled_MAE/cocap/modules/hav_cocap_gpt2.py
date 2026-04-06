# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @Description: Integrated Bridge Architecture utilizing ViT-B (CLIP), 
#               Distilled VideoMAE (Student), and GPT-2 (LoRA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

from cocap.modules.clip.model import build_model as build_clip 

# =====================================================================
# THE CAUSAL MASKING & SAFE POOLER
# =====================================================================
def safe_causal_motion_pooling(p_frame_mvs, eps=0.01, cut_threshold=0.15):
    """
    Kills duplicate frames, masks out imposter frames from scene cuts, 
    and safely pools 29 frames down to 8 without dilution.
    Expects input: [B*G, 2, 29, 56, 56]
    """
    B_G, C, T, H, W = p_frame_mvs.shape
    
    # 1. Filter Duplicates
    mag = p_frame_mvs.abs().mean(dim=(1, 3, 4))
    is_real = (mag > eps).float() 
    
    # 2. Detect Scene Cuts (Imposter P-frames)
    flat_mvs = p_frame_mvs.permute(0, 2, 1, 3, 4).reshape(B_G, T, -1)
    sim = F.cosine_similarity(flat_mvs[:, :-1, :], flat_mvs[:, 1:, :], dim=2)
    sim = torch.cat([torch.ones(B_G, 1, device=sim.device), sim], dim=1)
    is_cut = (sim < cut_threshold).float()
    
    # 3. Causal Mask: Everything after the cut becomes 0
    valid_scene = (is_cut.cumsum(dim=1) == 0).float()
    final_mask = (is_real * valid_scene).view(B_G, 1, T, 1, 1)
    masked_mvs = p_frame_mvs * final_mask
    
    # 4. Safe Average Pooling (29 -> 8)
    pool_raw = F.adaptive_avg_pool3d(masked_mvs, (8, H, W))
    pool_mask = F.adaptive_avg_pool3d(final_mask, (8, 1, 1))
    
    # 5. Prevent Division by Zero
    pool_mask_clamped = pool_mask.clamp(min=1e-3)
    corrected = pool_raw / pool_mask_clamped
    corrected = corrected.masked_fill(pool_mask < eps, 0.0)
    
    return corrected # Returns [B*G, 2, 8, 56, 56]


class HavCoCapGPT2(nn.Module):
    def __init__(
        self, 
        clip_state_dict: Dict[str, torch.Tensor], 
        gpt2_model_path: str, 
        motion_encoder: nn.Module, 
        spatial_dim: int = 512,
        gpt_dim: int = 768,
        projection_depth: int = 2
    ):
        super().__init__()
        
        # 1. Vision Backbones
        self.clip = build_clip(clip_state_dict)
        self.motion_encoder = motion_encoder
        self._freeze_module(self.clip)
        self._freeze_module(self.motion_encoder)

        # 2. Trainable Bridge (Spatial only)
        self.proj_spatial = self._build_mlp(spatial_dim, gpt_dim, projection_depth)

        # 3. Language Decoder (LoRA)
        base_gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["c_attn"], 
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.gpt2 = get_peft_model(base_gpt2, lora_config)
        self.phase = 2 # Default fallback
        self.set_training_phase(phase=2)

    def _freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _build_mlp(self, in_dim: int, out_dim: int, depth: int) -> nn.Sequential:
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def set_training_phase(self, phase: int):
        self.phase = phase  # Save phase state so forward pass knows when to do Delta Math
        print(f"\n--- Setting Model to Training Phase {phase} ---")
        
        for param in self.parameters():
            param.requires_grad = False
        
        if phase == 1:
            for param in self.motion_encoder.parameters(): param.requires_grad = True
        elif phase == 2:
            for param in self.proj_spatial.parameters(): param.requires_grad = True
            for param in self.motion_encoder.proj_gpt.parameters(): param.requires_grad = True
        elif phase == 3:
            for param in self.proj_spatial.parameters(): param.requires_grad = True
            for param in self.motion_encoder.proj_gpt.parameters(): param.requires_grad = True
            for name, param in self.gpt2.named_parameters():
                if "lora" in name: param.requires_grad = True

    def _extract_and_pool_visuals(self, iframes: torch.Tensor, next_iframes: torch.Tensor, motion_vectors: torch.Tensor):
        B, G = iframes.shape[:2]
        
        with torch.no_grad():
            # 1. Encode CURRENT I-Frame (t)
            cls_curr, spatial_curr = self.clip.encode_image(iframes.view(-1, 3, 224, 224), output_all_features=True)
            
            # 2. DELTA DISTILLATION (t+1 - t) ONLY computed during Phase 1
            target_delta = None
            if self.training and self.phase == 1:
                _, spatial_next = self.clip.encode_image(next_iframes.view(-1, 3, 224, 224), output_all_features=True)
                target_delta = spatial_next - spatial_curr # [B*G, 196, 512]

        # --- 3. CAUSAL MASKING & SAFE POOLING ---
        # Reshape to [B*G, 2, 29, 56, 56] for the preprocessor
        bg_mvs = motion_vectors.view(B * G, 29, 2, 56, 56).permute(0, 2, 1, 3, 4)
        safe_pooled_mvs = safe_causal_motion_pooling(bg_mvs) # Output is now exactly [B*G, 2, 8, 56, 56]
        
        # GPT Head outputs 8 tokens per GOP
        motion_embeds = self.motion_encoder(safe_pooled_mvs, return_for_gpt=True)
        motion_embeds = motion_embeds.view(B, -1, 768)
        
        # Teacher Head outputs prediction for the Delta
        pred_delta = None
        if self.training and self.phase == 1:
            pred_delta = self.motion_encoder(safe_pooled_mvs, return_for_gpt=False)

        # --- 4. SPATIAL POOLING (196 -> 49) ---
        grid_14x14 = spatial_curr.transpose(1, 2).view(-1, 512, 14, 14)
        grid_7x7 = F.avg_pool2d(grid_14x14, kernel_size=2, stride=2)
        pooled_patches = grid_7x7.view(-1, 512, 49).transpose(1, 2)

        # Combine: 1 CLS + 49 Spatial = 50 Tokens
        spatial_prefix = torch.cat([cls_curr.unsqueeze(1), pooled_patches], dim=1)
        spatial_embeds = self.proj_spatial(spatial_prefix).view(B, -1, 768)

        return spatial_embeds, motion_embeds, pred_delta, target_delta

    def forward(
        self, iframes, motion_vectors, input_ids, attention_mask, labels
    ) -> dict:
        B = iframes.size(0)

        # --- INTERNAL T+1 SHIFT HACK ---
        # Shifts frames left by 1 to get t+1. Duplicates the last frame to maintain G=8.
        # This requires ZERO changes to your dataloader.
        next_iframes = torch.cat([iframes[:, 1:], iframes[:, -1:]], dim=1)

        # 1. Extraction
        spatial_embeds, motion_embeds, pred_delta, target_delta = self._extract_and_pool_visuals(
            iframes, next_iframes, motion_vectors
        )
        
        # 50 Spatial + 8 Motion per GOP concatenated sequentially
        visual_embeds = torch.cat([spatial_embeds, motion_embeds], dim=1)

        # 2. Decoder Pass
        text_embeds = self.gpt2.base_model.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        v_mask = torch.ones((B, visual_embeds.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([v_mask, attention_mask], dim=1)
        
        v_labels = torch.full((B, visual_embeds.size(1)), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([v_labels, labels], dim=1)

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels, return_dict=True)
        
        # Return as dict so cocap.py can pass it exactly to PhaseAwareLoss
        return {
            "prediction_scores": outputs.logits,
            "teacher_predicted": pred_delta,
            "teacher_target": target_delta
        }

    @torch.no_grad()
    def generate(self, iframes, motion_vectors, tokenizer, prompt="", **kwargs):
        self.eval()
        B = iframes.size(0)
        
        # Generate doesn't need next_iframes (it's inference only), pass iframes just to satisfy the function signature
        spatial_embeds, motion_embeds, _, _ = self._extract_and_pool_visuals(iframes, iframes, motion_vectors)
        
        inputs_embeds = torch.cat([
            torch.cat([spatial_embeds, motion_embeds], dim=1),
            self.gpt2.base_model.model.transformer.wte(tokenizer(prompt, return_tensors="pt").input_ids.to(iframes.device))
        ], dim=1)
        
        output_ids = self.gpt2.generate(inputs_embeds=inputs_embeds, pad_token_id=tokenizer.eos_token_id, **kwargs)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)