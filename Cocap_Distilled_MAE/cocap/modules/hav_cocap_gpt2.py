# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @Description: Integrated Bridge Architecture utilizing pre-extracted 768-dim ViT-B (CLIP), 
#               Distilled VideoMAE (Student), and GPT-2 (LoRA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model
from cocap.modules.clip.model import build_model as build_clip 

# =====================================================================
# CORE MODEL
# =====================================================================
class HavCoCapGPT2(nn.Module):
    def __init__(
        self, 
        gpt2_model_path: str, 
        motion_encoder: nn.Module, 
        clip_state_dict: Optional[Any] = None, 
        spatial_dim: int = 768,                
        gpt_dim: int = 768,                    
        projection_depth: int = 2
    ):
        super().__init__()
        
        # 1. Vision Backbones
        self.motion_encoder = motion_encoder
        self._freeze_module(self.motion_encoder)
        
        # Load CLIP only if provided (e.g., for deployment/generation on raw MP4s)
        self.clip = None
        if clip_state_dict is not None:
            if isinstance(clip_state_dict, str):
                clip_state_dict = torch.load(clip_state_dict, map_location="cpu")
            self.clip = build_clip(clip_state_dict)
            self._freeze_module(self.clip)

        # 2. Trainable Bridge (Spatial only)
        self.proj_spatial = self._build_mlp(spatial_dim, gpt_dim, projection_depth)

        # 3. Language Decoder (LoRA)
        base_gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["c_attn"], 
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.gpt2 = get_peft_model(base_gpt2, lora_config)
        
        # Default Phase
        self.phase = 2 
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
        self.phase = phase
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

    # =====================================================================
    # SPATIAL PROCESSOR (Static Pooling Alternative to MGDTR)
    # =====================================================================
    def _process_spatial_features(self, cls_768: torch.Tensor, spatial_768: torch.Tensor) -> torch.Tensor:
        B_G = spatial_768.shape[0]
        
        # 1. Spatial Pooling (196 → 49) 
        grid_14x14 = spatial_768.transpose(1, 2).view(B_G, 768, 14, 14)
        grid_7x7 = F.avg_pool2d(grid_14x14, kernel_size=2, stride=2)
        pooled_patches = grid_7x7.view(B_G, 768, 49).transpose(1, 2) 
        
        # 2. COMBINE: 1 CLS (768) + 49 Spatial (768) = 50 Tokens @ 768
        spatial_prefix = torch.cat([cls_768.unsqueeze(1), pooled_patches], dim=1) 
        
        # 3. FIX: NO MICRO-RESIDUAL ADDITION. Just pure projection to text-space.
        spatial_embeds = self.proj_spatial(spatial_prefix)
        
        return spatial_embeds # [B*G, 50, 768]

    def _gpt2_forward(self, B, spatial_embeds, motion_embeds, input_ids, attention_mask, labels):
        # Cat Visuals: [B, GOP*50, 768] + [B, GOP*8, 768] = [B, GOP*58, 768]
        visual_embeds = torch.cat([spatial_embeds, motion_embeds], dim=1)

        # Get Text Embeddings natively from GPT-2
        text_embeds = self.gpt2.base_model.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Build Masks and Labels (Visual tokens are not predicted, so label = -100)
        v_mask = torch.ones((B, visual_embeds.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([v_mask, attention_mask], dim=1)
        
        v_labels = torch.full((B, visual_embeds.size(1)), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([v_labels, labels], dim=1)

        return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels, return_dict=True)

    # =====================================================================
    # FORWARD PASS
    # =====================================================================
    def forward(
        self, 
        clip_i_cls: torch.Tensor,        
        clip_i_spatial: torch.Tensor,    
        motion_vectors: torch.Tensor,    
        input_ids: torch.Tensor,         
        attention_mask: torch.Tensor,    
        labels: torch.Tensor,            
        input_mask_mv: torch.Tensor,                 # MUST pass the mask now
        target_delta: Optional[torch.Tensor] = None  
    ) -> dict:
        
        B, G = clip_i_cls.shape[:2]
        num_mv = motion_vectors.shape[2]
        
        cls_flat = clip_i_cls.view(B * G, 768)
        spatial_flat = clip_i_spatial.view(B * G, 196, 768)
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        
        # Flatten mask for motion perceiver
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)

        # PHASE 1: DISTILLATION
        if self.training and self.phase == 1:
            pred_delta = self.motion_encoder(bg_mvs, input_mask_mv=flat_mask_mv, iframe_spatial=spatial_flat, return_for_gpt=False)
            return {
                "teacher_predicted": pred_delta,
                "teacher_target": target_delta.view(B * G, 196, 768) if target_delta is not None else None
            }

        # PHASE 2 & 3: GENERATIVE SFT
        spatial_embeds = self._process_spatial_features(cls_flat, spatial_flat).view(B, -1, 768)
        motion_embeds = self.motion_encoder(bg_mvs, input_mask_mv=flat_mask_mv, return_for_gpt=True).view(B, -1, 768)

        pred_delta = None
        if self.phase == 3:
            pred_delta = self.motion_encoder(bg_mvs, input_mask_mv=flat_mask_mv, iframe_spatial=spatial_flat, return_for_gpt=False)

        outputs = self._gpt2_forward(B, spatial_embeds, motion_embeds, input_ids, attention_mask, labels)
        
        return {
            "prediction_scores": outputs.logits,
            "teacher_predicted": pred_delta,
            "teacher_target": target_delta.view(B * G, 196, 768) if target_delta is not None else None
        }

    # =====================================================================
    # INFERENCE / GENERATION
    # =====================================================================
    @torch.no_grad()
    def generate(self, iframes: torch.Tensor, motion_vectors: torch.Tensor, input_mask_mv: torch.Tensor, tokenizer, prompt="", **kwargs):
        assert self.clip is not None, "You must provide clip_state_dict to use generate() on raw iframes."
        self.eval()
        B, G = iframes.shape[:2]
        
        _, spatial_curr = self.clip.encode_image(iframes.view(-1, 3, 224, 224), output_all_features=True)
        spatial_curr = spatial_curr.float()
        
        cls_flat = spatial_curr[:, 0, :]
        spatial_flat = spatial_curr[:, 1:, :]

        spatial_embeds = self._process_spatial_features(cls_flat, spatial_flat).view(B, -1, 768)
        
        num_mv = motion_vectors.shape[2]
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)
        
        motion_embeds = self.motion_encoder(bg_mvs, input_mask_mv=flat_mask_mv, return_for_gpt=True).view(B, -1, 768)
        
        visual_embeds = torch.cat([spatial_embeds, motion_embeds], dim=1)
        num_v_tokens = visual_embeds.size(1)
        
        # FIX: Generate proper dummy IDs to prevent HuggingFace crash
        text_inputs = tokenizer(prompt, return_tensors="pt").to(iframes.device)
        text_ids = text_inputs.input_ids
        text_embeds = self.gpt2.base_model.model.transformer.wte(text_ids)
        
        inputs_embeds = torch.cat([visual_embeds, text_embeds.repeat(B, 1, 1)], dim=1)
        
        dummy_v_ids = torch.full((B, num_v_tokens), tokenizer.eos_token_id, dtype=torch.long, device=iframes.device)
        full_input_ids = torch.cat([dummy_v_ids, text_ids.repeat(B, 1)], dim=1)
        
        output_ids = self.gpt2.generate(
            inputs_embeds=inputs_embeds, 
            pad_token_id=tokenizer.eos_token_id, 
            **kwargs
        )
        return tokenizer.decode(output_ids[0][num_v_tokens:], skip_special_tokens=True)