# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : compressed_video_transformer.py

__all__ = [
    "IFrameEncoder",
    "MotionCompressor",
    "CompressedVideoTransformer",
    "iframe_encoder_cfg",
    "iframe_encoder_pretrained_cfg",
    "motion_compressor_cfg",
    "compressed_video_transformer_cfg",
    "compressed_video_transformer_pretrained_cfg",
]

import logging
from typing import Optional, Dict, Tuple

import einops
import torch
import torch.nn as nn
from hydra_zen import builds

from havcocap_new.modules.clip.clip import get_model_path
from havcocap_new.modules.clip.model import VisionTransformer, LayerNorm, CLIP

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
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid = x.size(2)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_feature = self.ln_post(x[:, 0, :]) @ self.proj

        outputs = (cls_feature,)
        if output_all_features:
            # cls token is not included, f_hidden shape: [*, grid**2, output_dim]
            outputs += (self.ln_post_hidden(x[:, 1:, :]) @ self.proj_hidden,)
        if output_attention_map:
            # attention_map: n_layers, batch_size, n_heads, h, w
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


class MotionCompressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, embed_dim=768, num_queries=4, num_heads=8):
        super().__init__()
        
        # 1. The Coordinate MLP (Translates 2D MV math into feature space)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 2. The Learnable Query Tokens (The "Sponges")
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        nn.init.xavier_uniform_(self.query_tokens)
        
        # 3. The Cross-Attention Mechanism
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        
        # 4. Standard Transformer Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, mv_patches):
        """
        mv_patches: Tensor of shape [Batch, 49, 512] 
        """
        batch_size = mv_patches.size(0)
        
        # Step 1: Pass raw MV math through the MLP
        kv_features = self.mlp(mv_patches)
        
        # Step 2: Expand our 4 Query Tokens
        q = self.query_tokens.expand(batch_size, -1, -1)
        
        # Step 3: Cross-Attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_k(kv_features)
        
        attn_output, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        
        # Add residual connection
        out = q + attn_output
        
        # Step 4: Final Feed-Forward
        out = out + self.ffn(self.norm_ffn(out))
        
        return out


class CompressedVideoTransformer(nn.Module):
    def __init__(
            self,
            rgb_encoder: nn.Module,
            motion_encoder: nn.Module,
            output_dim: int,
    ):
        """
        Encode visual feature from video compressed domain
        :param rgb_encoder: IFrameEncoder
        :param motion_encoder: MotionCompressor
        :param output_dim: width of output visual feature (usually 768)
        """
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.motion_encoder = motion_encoder
        self.output_dim = output_dim
        
        # Adaptive pooling to reduce 196 tokens -> 16 tokens for the I-frames
        self.pool1d = nn.AdaptiveAvgPool1d(16)

    def forward(
            self,
            iframe: torch.FloatTensor,
            motion: torch.FloatTensor,
            residual: Optional[torch.FloatTensor] = None, # kept strictly for sig compat
            bp_type_ids: Optional[torch.LongTensor] = None # kept strictly for sig compat
    ) -> Dict[str, torch.Tensor]:
        """
        :param iframe:      [bsz, n_gop, 3, 224, 224]  (e.g., [B, 8, 3, 224, 224])
        :param motion:      [bsz, n_gop, n_bp, c_mv, h/4, w/4] 
                            For H264 MVs, c_mv=4. 
                            If n_gop=8, n_bp=7, shape=[B, 8, 7, 4, 56, 56]
        """
        bsz = iframe.size(0)
        n_gop = iframe.size(1)

        # 1. Encode I-Frames
        # Shape: (bsz * n_gop, 3, h, w)
        iframe_flat = einops.rearrange(iframe, "b g c h w -> (b g) c h w")
        
        _, f_ctx_all_hidden = self.rgb_encoder(
            iframe_flat,
            output_all_features=True, output_attention_map=False
        )
        # f_ctx_all_hidden: [bsz * n_gop, 196, 768]
        # Pool: [batch*gop, 196, 768] -> permute -> pool over token dimension
        x_ctx = f_ctx_all_hidden.permute(0, 2, 1) # [bsz*8, 768, 196]
        x_ctx = self.pool1d(x_ctx) # [bsz*8, 768, 16]
        x_ctx = x_ctx.permute(0, 2, 1) # [bsz*8, 16, 768]
        
        f_ctx = einops.rearrange(x_ctx, "(b g) t d -> b g t d", b=bsz) 
        # Output Context Feature -> shape: [B, 8, 16, 768]

        # 2. Encode P-Frames
        # Raw mv shape: [bsz, n_gop, n_bp, 4, 56, 56]
        n_bp = motion.size(2)
        # Flatten into patches for the Motion Compressor MLP
        # The coordinate MLP expects [Batch, 49, input_dim].
        # Oh wait, the original motion shape is [B, 8, 7, 4, 56, 56]. 
        # Wait, if we use 16x16 patch sizes on 56x56, we don't cleanly get 49 patches unless we pool.
        # But NEW_MV.py says: "49 patches from a 56x56 grid" - 56 / 8 = 7 -> 7x7 grid = 49 patches.
        # So we should unflatten: (56/8)*(56/8) = 49 patches of size 8x8.
        # Let's pool or rearrange it into 49 patches.
        # Each patch is 4 channels * 8 * 8 = 256. If input_dim=512, maybe 4 * 8 * 16? 
        # No, the previous `MotionEncoder` extracted features. Let's look at `NEW_MV.py` input_dim=512.
        # If we use `nn.Unfold` or rearrange: 
        # (bsz*n_gop*n_bp, 4, 56, 56) -> patch_size = 8 -> (bsz*n_gop*n_bp, 4 * 8 * 8=256). 
        # Wait, "(49 patches from a 56x56 grid, 512 channels per patch)". How did the user get 512 channels from 4 channels?
        # Maybe they used a small conv stem? Or maybe they mean the previous motion encoder output?
        # In NEW_MV.py there's no conv, just nn.Linear(input_dim, hidden). 
        # So we must provide `[Batch, 49, 512]`. If we just project the 256 vector to 512, or just pass input_dim=256 to MotionCompressor.
        # Let's specify input_dim=256 (which is 4*8*8).
        
        # Reshape motion into patches: [bsz*n_gop*n_bp, 4, 8x7, 8x7] -> [..., 49, 256]
        mv_flat = einops.rearrange(motion, "b g p c (h p1) (w p2) -> (b g p) (h w) (c p1 p2)", p1=8, p2=8)
        # That gives (..., 7x7=49, 4x8x8=256).
        
        # We need to make sure MotionCompressor uses input_dim=256 instead of 512.
        f_act_flat = self.motion_encoder(mv_flat)
        # Output shape: [(b g p), 4, 768]
        
        # Reshape back to [b, 8*7, 4, 768] (which represents 56 P-frames, 4 tokens each)
        f_act = einops.rearrange(f_act_flat, "(b g p) t d -> b (g p) t d", b=bsz, g=n_gop, p=n_bp)
        
        # Sequence Assemble logic will happen in the Captioner.
        # We'll just return these two streams.
        return {
            "feature_context": f_ctx,   # [B, 8, 16, 768]
            "feature_action": f_act,    # [B, 56, 4, 768]
        }

    @classmethod
    def from_pretrained(
            cls,
            pretrained_clip_name_or_path: str = "ViT-B/16",
    ):
        rgb_encoder, _, _, embed_dim = IFrameEncoder.from_pretrained(
            pretrained_clip_name_or_path  # ViT-B/16 -> embed_dim=768
        )

        # Motion Compressor with input_dim=256 (4 channels * 8 * 8 patch)
        motion_encoder = MotionCompressor(
            input_dim=256, 
            hidden_dim=1024, 
            embed_dim=embed_dim, 
            num_queries=4, 
            num_heads=8
        )

        return cls(
            rgb_encoder=rgb_encoder,
            motion_encoder=motion_encoder,
            output_dim=embed_dim
        )


# Build configs for organizing modules with hydra
iframe_encoder_cfg = builds(IFrameEncoder, populate_full_signature=True)
iframe_encoder_pretrained_cfg = builds(IFrameEncoder.from_pretrained, populate_full_signature=True)

motion_compressor_cfg = builds(MotionCompressor, populate_full_signature=True)

compressed_video_transformer_cfg = builds(
    CompressedVideoTransformer,
    rgb_encoder=iframe_encoder_cfg,
    motion_encoder=motion_compressor_cfg,
    populate_full_signature=True
)
compressed_video_transformer_pretrained_cfg = builds(
    CompressedVideoTransformer.from_pretrained,
    populate_full_signature=True
)
