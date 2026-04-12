# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen (Updated for HavCocap MGDTR Architecture)
# @Project : CoCap
# @File    : compressed_video_transformer.py

__all__ = [
    "IFrameEncoder",
    "CompressedVideoTransformer",
    "iframe_encoder_cfg",
    "iframe_encoder_pretrained_cfg",
    "compressed_video_transformer_cfg",
    "compressed_video_transformer_pretrained_cfg",
]

import logging
from typing import Optional, Dict, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra_zen import builds

from cocap.modules.clip.clip import get_model_path
from cocap.modules.clip.model import VisionTransformer, LayerNorm, CLIP

# IMPORT YOUR CUSTOM MOTION STUDENT
from .motion_encoder import MotionStudent 

logger = logging.getLogger(__name__)


class IFrameEncoder(VisionTransformer):
    def __init__(
            self,
            input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
            in_channels: int = 3
    ):
        super().__init__(
            input_resolution=input_resolution, patch_size=patch_size, width=width, layers=layers, heads=heads,
            output_dim=output_dim, in_channels=in_channels
        )

        scale = width ** -0.5
        self.ln_post_hidden = LayerNorm(width)
        self.proj_hidden = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, output_all_features: bool = False, output_attention_map: bool = False):
        x = self.conv1(x)  
        grid = x.size(2)
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  
        x, attn = self.transformer(x)
        x = x.permute(1, 0, 2)  

        cls_feature = self.ln_post(x[:, 0, :]) @ self.proj

        outputs = (cls_feature,)
        if output_all_features:
            outputs += (self.ln_post_hidden(x[:, 1:, :]) @ self.proj_hidden,)
        if output_attention_map:
            outputs += (einops.rearrange(attn[:, :, :, 0, 1:],
                                         "n_layers b n_heads (h w)->n_layers b n_heads h w", h=grid, w=grid),)
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_clip_name_or_path: str) -> Tuple["IFrameEncoder", int, int, int]:
        model_path = get_model_path(pretrained_clip_name_or_path)
        pretrained_model: CLIP = torch.jit.load(model_path, map_location="cpu")
        state_dict = pretrained_model.state_dict()

        vision_width: int = state_dict["visual.conv1.weight"].shape[0]
        vision_layers: int = len([k for k in state_dict.keys()
                                  if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        embed_dim: int = state_dict["text_projection"].shape[1]
        vision_patch_size: int = state_dict["visual.conv1.weight"].shape[-1]
        grid_size: int = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution: int = vision_patch_size * grid_size
        vision_heads = vision_width // 64

        rgb_encoder = cls(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width, layers=vision_layers, heads=vision_heads,
            output_dim=embed_dim
        )
        visual_state_dict = pretrained_model.visual.state_dict()
        visual_state_dict.update({k: v for k, v in rgb_encoder.state_dict().items() if k.startswith("ln_post_hidden")})
        visual_state_dict.update({k: v for k, v in rgb_encoder.state_dict().items() if k.startswith("proj_hidden")})
        rgb_encoder.load_state_dict(visual_state_dict, strict=True)

        return rgb_encoder, image_resolution, vision_width, embed_dim


class MotionGuidedDynamicTokenRouter(nn.Module):
    def __init__(self, embed_dim: int, total_tokens: int = 64, base_tokens: int = 2, max_gops: int = 8):
        super().__init__()
        self.total_tokens = total_tokens
        self.base_tokens = base_tokens
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_embed = nn.Embedding(max_gops, embed_dim)

    def forward(self, spatial_patches, motion_tokens, gop_mask):
        B, num_gops, num_patches, D = spatial_patches.shape
        
        # 1. GOP-level Motion Weights
        motion_magnitude = motion_tokens.norm(dim=-1).mean(dim=2) # [B, 8]
        valid_mask = (gop_mask == 0).float()
        motion_magnitude = motion_magnitude * valid_mask
        
        batch_selected_tokens = []
        
        for b in range(B):
            valid_gops = valid_mask[b].nonzero(as_tuple=True)[0]
            num_valid = len(valid_gops)
            
            if num_valid == 0:
                batch_selected_tokens.append(torch.zeros(self.total_tokens, D, device=spatial_patches.device))
                continue
                
            # 2. Dynamic Budget Allocation
            budget = self.total_tokens - (num_valid * self.base_tokens)
            weights = motion_magnitude[b, valid_gops] / (motion_magnitude[b, valid_gops].sum() + 1e-6)
            
            exact_allocation = weights * budget
            floor_allocation = exact_allocation.floor().int()
            remainder = int(budget - floor_allocation.sum().item())
            
            fractional_parts = exact_allocation - floor_allocation
            top_remainder_indices = torch.topk(fractional_parts, remainder).indices
            floor_allocation[top_remainder_indices] += 1
            
            final_allocation = floor_allocation + self.base_tokens 
            
            # 3. Localized Selection with Gradient Bridge
            video_tokens = []
            for idx, gop_idx in enumerate(valid_gops):
                k = final_allocation[idx].item()
                
                gop_motion_query = self.q_proj(motion_tokens[b, gop_idx].mean(dim=0)) 
                gop_spatial_keys = self.k_proj(spatial_patches[b, gop_idx])           
                
                relevance = torch.einsum('d, sd -> s', gop_motion_query, gop_spatial_keys) 
                
                topk_indices = torch.topk(relevance, k).indices
                selected_patches = spatial_patches[b, gop_idx, topk_indices] 
                
                # ---> THE GRADIENT BRIDGE (Allows GPT-2 Loss to train MotionStudent) <---
                patch_weights = F.softmax(relevance[topk_indices], dim=-1).unsqueeze(-1)
                selected_patches = selected_patches * patch_weights 
                
                # Add temporal embedding for GPT-2
                selected_patches = selected_patches + self.temporal_embed(gop_idx)
                
                video_tokens.append(selected_patches)
                
            batch_selected_tokens.append(torch.cat(video_tokens, dim=0))
            
        return torch.stack(batch_selected_tokens, dim=0) # [B, 64, D]


class CompressedVideoTransformer(nn.Module):
    def __init__(
            self,
            rgb_encoder: nn.Module,
            motion_encoder: nn.Module,
            output_dim: int,
    ):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.motion_encoder = motion_encoder
        self.output_dim = output_dim
        
        # Initialize MGDTR
        self.mgdtr = MotionGuidedDynamicTokenRouter(embed_dim=output_dim, total_tokens=64, base_tokens=2)

    def forward(
            self,
            iframe: torch.FloatTensor,
            motion: torch.FloatTensor,
            residual: torch.FloatTensor,
            bp_type_ids: torch.LongTensor,
            input_mask_gop: torch.Tensor,
            input_mask_mv: torch.Tensor,
            bp_rgb: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # FIX: Ensure motion has 2 channels (dx, dy)
        assert iframe.size(2) == 3 and motion.size(3) == 2 and residual.size(3) == 3, "channel number is not correct"

        _bsz, n_gop, n_bp = iframe.size(0), motion.size(1), motion.size(2)

        # 1. ENCODE I-FRAMES
        f_ctx_cls, f_ctx_all_hidden, _ = self.rgb_encoder(
            einops.rearrange(iframe, "bsz n_gop c h w -> (bsz n_gop) c h w"),
            output_all_features=True, output_attention_map=False
        )
        f_ctx_all_hidden = einops.rearrange(f_ctx_all_hidden, "(bsz n_gop) hw c -> bsz n_gop hw c", bsz=_bsz)

        # 2. ENCODE MOTION
        flat_mask_mv = einops.rearrange(input_mask_mv, "bsz n_gop n_bp -> (bsz n_gop) n_bp")
        
        # Output is exactly [B*G, 8, embed_dim]
        mv_tokens = self.motion_encoder(
            einops.rearrange(motion, "bsz n_gop n_bp c_mv h w -> (bsz n_gop) n_bp c_mv h w"),
            input_mask_mv=flat_mask_mv
        )
        
        mv_tokens = einops.rearrange(mv_tokens, "(bsz n_gop) num_t c -> bsz n_gop num_t c", 
                                     bsz=_bsz, n_gop=n_gop)
        
        # 3. MGDTR ROUTING
        routed_spatial_tokens = self.mgdtr(
            spatial_patches=f_ctx_all_hidden, 
            motion_tokens=mv_tokens, 
            gop_mask=input_mask_gop
        )

        return {
            "feature_context": f_ctx_cls,
            "feature_context_spatial": f_ctx_all_hidden,
            "routed_spatial_tokens": routed_spatial_tokens,
            "feature_motion": mv_tokens, 
            "residual": residual
        }

    @classmethod
    def from_pretrained(
            cls,
            pretrained_clip_name_or_path: str = "ViT-B/16",
            motion_embed_dim: int = 256, motion_layers: int = 4, motion_heads: int = 4,
    ):
        rgb_encoder, image_resolution, vision_width, embed_dim = IFrameEncoder.from_pretrained(
            pretrained_clip_name_or_path
        )

        # Initialize the custom MotionStudent imported from motion_encoder.py
        motion_encoder = MotionStudent(
            in_channels=2, 
            embed_dim=motion_embed_dim, 
            num_heads=motion_heads, 
            num_layers=motion_layers
        )
        
        # We pass output_dim=embed_dim to ensure MGDTR projects cleanly to 768 later
        return cls(
            rgb_encoder=rgb_encoder,
            motion_encoder=motion_encoder,
            output_dim=embed_dim
        )


# Build configs for organizing modules with hydra
iframe_encoder_cfg = builds(IFrameEncoder, populate_full_signature=True)
iframe_encoder_pretrained_cfg = builds(IFrameEncoder.from_pretrained, populate_full_signature=True)

compressed_video_transformer_cfg = builds(
    CompressedVideoTransformer,
    rgb_encoder=iframe_encoder_cfg,
    motion_encoder=None, # Will be injected by the main script
    populate_full_signature=True
)
compressed_video_transformer_pretrained_cfg = builds(
    CompressedVideoTransformer.from_pretrained,
    populate_full_signature=True
)