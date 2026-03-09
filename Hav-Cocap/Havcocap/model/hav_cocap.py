import torch
import torch.nn as nn
import einops
from typing import Optional, Dict, Tuple
import sys
import os
from collections import OrderedDict

# Add CoCap path to allow importing modules from the original repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../CoCap")))

# Attempt to import CaptionHead. If it fails due to path issues, we might need adjustments.
try:
    from cocap.modules.compressed_video.compressed_video_captioner import CaptionHead
except ImportError:
    # Fallback or maybe we rely on the sys.path above
    pass

# Import BEATs
try:
    from learn.beats.beat_architecture import BEATsModel, BEATsConfig
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from learn.beats.beat_architecture import BEATsModel, BEATsConfig

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, key_padding_mask=padding_mask, average_attn_weights=False)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class CrossResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__(d_model, n_head, attn_mask)
        self.attn2 = nn.MultiheadAttention(d_model, n_head)
        self.ln_3 = LayerNorm(d_model)
        self.ln_4 = LayerNorm(d_model)

    def forward(self, x):
        x_student, x_teacher, self_mask = x
        x_student = x_student + self.attention(self.ln_1(x_student), padding_mask=None)[0]
        attn_out = self.attn2(
            self.ln_3(x_student), 
            self.ln_4(x_teacher), 
            self.ln_4(x_teacher),
            need_weights=False
        )[0]
        x_student = x_student + attn_out
        x_student = x_student + self.mlp(self.ln_2(x_student))
        return x_student

class HavCocapCaptioner(nn.Module):
    def __init__(
        self,
        hav_cocap_model: "HavCocapModel",
        caption_head, # Type hint removed to avoid import error if CaptionHead generic
        motion_dropout_prob: float = 0.2,
        residual_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.hav_cocap_model = hav_cocap_model
        self.caption_head = caption_head
        self.dropout_motion = nn.Dropout(motion_dropout_prob)
        self.dropout_residual = nn.Dropout(residual_dropout_prob)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if "visual_output" not in inputs:
            iframe = inputs["iframe"]
            motion = inputs["motion"]
            residual = inputs["residual"]
            audio = inputs["audio"]

            motion = self.dropout_motion(motion)
            residual = self.dropout_residual(residual)

            bp_type_ids = inputs.get("bp_type_ids")
            if bp_type_ids is None:
                b, t, m = motion.shape[:3]
                bp_type_ids = torch.zeros((b, t, m), dtype=torch.long, device=motion.device)

            visual_output = self.hav_cocap_model(
                iframe=iframe,
                motion=motion,
                residual=residual,
                audio=audio,
                bp_type_ids=bp_type_ids
            )
        else:
            visual_output = inputs["visual_output"]

        prediction_scores = self.caption_head(
            visual_output,
            inputs["input_ids"],
            inputs["input_mask"]
        )
        
        return {"prediction_scores": prediction_scores, "visual_output": visual_output}

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, in_channels: int = 3):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, output_all_features: bool = False):
        x = self.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        
        for layer in self.transformer:
            x = layer(x)
            
        x = x.permute(1, 0, 2)
        cls_token = self.ln_post(x[:, 0, :]) @ self.proj
        
        if output_all_features:
             return cls_token, x[:, 1:, :] 
        return cls_token, None

class IFrameEncoder(VisionTransformer):
    pass

class AudioEncoder(nn.Module):
    def __init__(self, model_path=None, output_dim=768):
        super().__init__()
        self.cfg = BEATsConfig({
            "encoder_layers": 12,
            "encoder_embed_dim": 768,
            "encoder_ffn_embed_dim": 3072,
            "encoder_attention_heads": 12,
        })
        self.model = BEATsModel(self.cfg)
        
        if model_path:
            print(f"Loading Audio Encoder from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.output_dim = output_dim
        self.proj = nn.Linear(768, output_dim) if output_dim != 768 else nn.Identity()

    def forward(self, audio_input):
        padding_mask = torch.zeros(audio_input.shape[0], audio_input.shape[1]).bool().to(audio_input.device)
        features, _ = self.model.extract_features(audio_input)
        feat_mean = features.mean(dim=1)
        feat_proj = self.proj(feat_mean)
        feat_proj = torch.nn.functional.layer_norm(feat_proj, (self.output_dim,))
        return feat_proj

class ActionEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, n_bp: int, n_bp_type: int):
        super().__init__()
        self.width = width
        self.resblocks = nn.Sequential(*[CrossResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.positional_embedding = nn.Embedding(n_bp, width)
        self.bp_type_embedding = nn.Embedding(n_bp_type, width)
        self.ln_post = LayerNorm(width)

    def forward(self, feature_bp, bp_type_ids, feature_ctx):
        bsz = feature_bp.size(0)
        pos_ids = torch.arange(feature_bp.size(1), device=feature_bp.device).unsqueeze(0).repeat(bsz, 1)
        if pos_ids.shape[1] > self.positional_embedding.num_embeddings:
             pos_ids = pos_ids[:, :self.positional_embedding.num_embeddings]
             feature_bp = feature_bp[:, :self.positional_embedding.num_embeddings, :]
             
        feature_bp = feature_bp + self.positional_embedding(pos_ids)
        feature_bp = feature_bp + self.bp_type_embedding(bp_type_ids)
        
        out = feature_bp.permute(1, 0, 2)
        ctx = feature_ctx.permute(1, 0, 2)
        
        for block in self.resblocks:
            out = block([out, ctx, None])
            
        out = out.permute(1, 0, 2)
        out = torch.mean(out, dim=1)
        return self.ln_post(out)


class HavCocapModel(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 audio_model_path=None,
                 motion_patches=8, residual_patches=64):
        super().__init__()
        self.iframe_encoder = IFrameEncoder(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=embed_dim)
        self.motion_encoder = VisionTransformer(input_resolution=56, patch_size=motion_patches, width=512, layers=4, heads=8, output_dim=embed_dim, in_channels=4)
        self.motion_proj = nn.Linear(embed_dim, embed_dim)
        self.residual_encoder = VisionTransformer(input_resolution=224, patch_size=residual_patches, width=512, layers=4, heads=8, output_dim=embed_dim, in_channels=3)
        self.residual_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_encoder = AudioEncoder(model_path=audio_model_path, output_dim=embed_dim)
        self.action_encoder = ActionEncoder(width=embed_dim, layers=2, heads=8, n_bp=6, n_bp_type=2)
        self.projection_fused = nn.Linear(embed_dim, embed_dim)
        self.ctx_proj = nn.Linear(768, embed_dim) if embed_dim != 768 else nn.Identity()

    def forward(self, iframe, motion, residual, audio, bp_type_ids):
        b, t, c, h, w = iframe.shape
        b, t, m, cm, hm, wm = motion.shape
        
        iframe_flat = einops.rearrange(iframe, 'b t c h w -> (b t) c h w')
        f_ctx_cls, f_ctx_tokens = self.iframe_encoder(iframe_flat, output_all_features=True)
        f_ctx_tokens = self.ctx_proj(f_ctx_tokens)
        
        motion_flat = einops.rearrange(motion, 'b t m c h w -> (b t m) c h w')
        mv_cls, _ = self.motion_encoder(motion_flat)
        mv_cls = einops.rearrange(mv_cls, '(b t m) d -> (b t) m d', b=b, t=t, m=m)
        
        residual_flat = einops.rearrange(residual, 'b t m c h w -> (b t m) c h w')
        res_cls, _ = self.residual_encoder(residual_flat)
        res_cls = einops.rearrange(res_cls, '(b t m) d -> (b t) m d', b=b, t=t, m=m)
        
        audio_flat = einops.rearrange(audio, 'b t s -> (b t) s')
        audio_feat = self.audio_encoder(audio_flat)
        
        f_bp = mv_cls + res_cls 
        audio_feat_expanded = audio_feat.unsqueeze(1).expand(-1, m, -1)
        f_bp = f_bp + audio_feat_expanded
        
        bp_type_ids_flat = einops.rearrange(bp_type_ids, 'b t m -> (b t) m')
        f_act = self.action_encoder(feature_bp=f_bp, bp_type_ids=bp_type_ids_flat, feature_ctx=f_ctx_tokens)
        
        f_act = einops.rearrange(f_act, '(b t) d -> b t d', b=b, t=t)
        f_ctx_cls = einops.rearrange(f_ctx_cls, '(b t) d -> b t d', b=b, t=t)
        
        return {
            "feature_action": f_act,
            "feature_context": f_ctx_cls
        }
