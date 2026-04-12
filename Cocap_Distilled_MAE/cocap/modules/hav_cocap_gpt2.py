# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @Description: Integrated Bridge Architecture utilizing pre-extracted 768-dim ViT-B (CLIP), 
#               Distilled VideoMAE (Student), and GPT-2 (LoRA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from transformers import GPT2LMHeadModel

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    _HAS_PEFT = False

from cocap.modules.clip.model import build_model as build_clip 

# =====================================================================
# MGDTR SELECTOR (Per-GOP Cross-Attention)
# =====================================================================
class MGDTRSelector(nn.Module):
    def __init__(self, spatial_dim: int = 768, model_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.model_dim = model_dim
        # Requested projection path: motion 256 -> 512 -> 768
        self.motion_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, model_dim),
        )
        self.motion_proj_768 = nn.Linear(768, model_dim)
        self.spatial_ln = nn.LayerNorm(spatial_dim)
        self.motion_ln = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.score_head = nn.Linear(model_dim, 1)

    def forward(self, spatial_tokens: torch.Tensor, motion_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        spatial_tokens: [B, G, N, 768]
        motion_tokens : [B, G, M, d] (d can be 256 or 768)
        """
        B, G, N, _ = spatial_tokens.shape
        _, _, M, _ = motion_tokens.shape

        spatial_flat = spatial_tokens.reshape(B * G, N, -1)
        motion_flat = motion_tokens.reshape(B * G, M, -1)

        if motion_flat.shape[-1] == 256:
            motion_768 = self.motion_proj(motion_flat)
        elif motion_flat.shape[-1] == 768:
            motion_768 = self.motion_proj_768(motion_flat)
        else:
            raise RuntimeError(f"Unsupported motion token dim for MGDTR: {motion_flat.shape[-1]}")
        q = self.spatial_ln(spatial_flat)
        kv = self.motion_ln(motion_768)

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        fused = spatial_flat + attn_out

        selector_logits = self.score_head(fused).squeeze(-1)     # [BG, N]
        selector_prob = torch.sigmoid(selector_logits)            # [BG, N]
        coverage_counts = selector_prob.sum(dim=-1).reshape(B, G)

        patches_g_mean = (
            (spatial_flat * selector_prob.unsqueeze(-1)).sum(dim=1)
            / coverage_counts.reshape(B * G, 1).clamp(min=1.0)
        ).reshape(B, G, -1)

        return {
            "patches_g_mean": patches_g_mean,
            "coverage_counts": coverage_counts,
            "selector_prob": selector_prob.reshape(B, G, N),
        }

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
        projection_depth: int = 2,
        use_lora: bool = False,
    ):
        super().__init__()
        
        # 1. Vision Backbones
        self.motion_encoder = motion_encoder
        self._freeze_module(self.motion_encoder)
        
        # Load CLIP only if provided (e.g., for deployment/generation on raw MP4s)
        self.clip = None
        if clip_state_dict is not None:
            if isinstance(clip_state_dict, str):
                # Robust loader: OpenAI CLIP checkpoints can be TorchScript archives.
                try:
                    clip_state_dict = torch.jit.load(clip_state_dict, map_location="cpu").state_dict()
                except RuntimeError:
                    clip_state_dict = torch.load(clip_state_dict, map_location="cpu", weights_only=False)
            self.clip = build_clip(clip_state_dict)
            self._freeze_module(self.clip)

        # 2. Trainable Bridge (Spatial only)
        self.proj_spatial = self._build_mlp(spatial_dim, gpt_dim, projection_depth)
        self.mgdtr = MGDTRSelector(spatial_dim=spatial_dim, model_dim=gpt_dim, num_heads=8)

        # Mandatory projection from motion student space -> GPT space.
        student_dim = 256
        if hasattr(self.motion_encoder, "student") and hasattr(self.motion_encoder.student, "motion_queries"):
            student_dim = int(self.motion_encoder.student.motion_queries.shape[-1])
        self.motion_to_gpt = nn.Linear(student_dim, gpt_dim)

        # 3. Language Decoder (No LoRA in this setup)
        base_gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
        if getattr(base_gpt2.config, "loss_type", None) is None:
            base_gpt2.config.loss_type = "ForCausalLMLoss"
        self.using_lora = False
        self.gpt2 = base_gpt2
        self._freeze_module(self.gpt2)

        # Default phase at construction: keep phase-1 components intact.
        self.phase = 1
        self.set_training_phase(phase=1)

    @property
    def selector(self):
        # Backward-compatible alias without registering a duplicated child module.
        return self.mgdtr

    def unload_gpt2(self):
        self.gpt2 = None

    def unload_clip(self):
        self.clip = None

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

    def _get_text_embedding_layer(self):
        if self.gpt2 is None:
            raise RuntimeError("GPT-2 is unloaded for this phase.")
        if hasattr(self.gpt2, "base_model") and hasattr(self.gpt2.base_model, "model"):
            return self.gpt2.base_model.model.transformer.wte
        return self.gpt2.transformer.wte

    def _forward_motion_delta(self, bg_mvs: torch.Tensor, flat_mask_mv: torch.Tensor, spatial_flat: torch.Tensor):
        """Return predicted delta [BG, 196, 768] across motion-encoder API variants."""
        try:
            return self.motion_encoder(
                bg_mvs,
                input_mask_mv=flat_mask_mv,
                iframe_spatial=spatial_flat,
                return_for_gpt=False,
            )
        except TypeError:
            # Legacy MotionTransformer signature (no return_for_gpt)
            return self.motion_encoder(
                bg_mvs,
                input_mask_mv=flat_mask_mv,
                iframe_spatial=spatial_flat,
            )

    def _forward_motion_tokens(self, bg_mvs: torch.Tensor, flat_mask_mv: torch.Tensor):
        """Return GPT-space motion tokens [BG, 8, 768] across motion-encoder API variants."""
        # Preferred path: always take student tokens and apply mandatory projection.
        if hasattr(self.motion_encoder, "student"):
            latent = self.motion_encoder.student(bg_mvs, flat_mask_mv)  # [BG, 8, d]
            if latent.shape[-1] != self.motion_to_gpt.in_features:
                raise RuntimeError(
                    f"motion_to_gpt expects input dim={self.motion_to_gpt.in_features}, got {latent.shape[-1]}"
                )
            return self.motion_to_gpt(latent)

        # Fallback for other encoders that already expose GPT-space tokens.
        try:
            return self.motion_encoder(
                bg_mvs,
                input_mask_mv=flat_mask_mv,
                return_for_gpt=True,
            )
        except TypeError:
            raise RuntimeError("Motion encoder has no student and does not support return_for_gpt=True")

    def _forward_motion_tokens_for_mgdtr(self, bg_mvs: torch.Tensor, flat_mask_mv: torch.Tensor):
        """Return motion tokens for MGDTR path, preferring raw student dim=256."""
        if hasattr(self.motion_encoder, "student"):
            return self.motion_encoder.student(bg_mvs, flat_mask_mv)
        # Fallback: if no explicit student is exposed, use available token path.
        return self._forward_motion_tokens(bg_mvs, flat_mask_mv)

    def set_training_phase(self, phase: int):
        self.phase = phase
        print(f"\n--- Setting Model to Training Phase {phase} ---")

        # Distillation decoder exists only for phase 1.
        if phase in (2, 3) and hasattr(self.motion_encoder, "decoder") and self.motion_encoder.decoder is not None:
            self.motion_encoder.decoder = None
        
        for param in self.parameters():
            param.requires_grad = False
        
        if phase == 1:
            for param in self.motion_encoder.parameters(): param.requires_grad = True
        elif phase == 2:
            for param in self.mgdtr.parameters():
                param.requires_grad = True
        elif phase == 3:
            # Phase 3 must NOT train the distillation decoder.
            # Train: motion student + MGDTR + projection layers (decoder remains frozen).
            if hasattr(self.motion_encoder, "student"):
                for param in self.motion_encoder.student.parameters():
                    param.requires_grad = True
            for param in self.mgdtr.parameters():
                param.requires_grad = True
            for param in self.proj_spatial.parameters(): param.requires_grad = True
            if hasattr(self.motion_encoder, "proj_gpt"):
                for param in self.motion_encoder.proj_gpt.parameters():
                    param.requires_grad = True
            for param in self.motion_to_gpt.parameters():
                param.requires_grad = True

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

    def _build_phase3_visual_tokens(self, cls_flat: torch.Tensor, spatial_flat: torch.Tensor, B: int, G: int):
        """Build visual tokens in strict order expected by phase-3 captioning:
        [8 CLS] + [64 Spatial] + [64 Motion] (motion is appended by caller).
        """
        # CLS tokens: one per GOP -> [B, G, 768]
        cls_tokens = self.proj_spatial(cls_flat.unsqueeze(1)).view(B, G, 768)

        # Spatial tokens: pool 196 patches -> 8 patches per GOP, then project.
        grid_14x14 = spatial_flat.transpose(1, 2).view(B * G, 768, 14, 14)
        grid_2x4 = F.adaptive_avg_pool2d(grid_14x14, output_size=(2, 4))
        spatial_8 = grid_2x4.view(B * G, 768, 8).transpose(1, 2)                # [BG,8,768]
        spatial_tokens = self.proj_spatial(spatial_8).view(B, G * 8, 768)        # [B,64,768]

        cls_tokens = cls_tokens.view(B, G, 768)                                   # [B,8,768]
        return cls_tokens, spatial_tokens

    def _gpt2_forward(self, B, cls_embeds, spatial_embeds, motion_embeds, input_ids, attention_mask, labels):
        if self.gpt2 is None:
            raise RuntimeError("GPT-2 is unloaded for this phase.")
        # Required order: [8 CLS] + [64 Spatial] + [64 Motion]
        visual_embeds = torch.cat([cls_embeds, spatial_embeds, motion_embeds], dim=1)

        # BOS + target text for decoder-only GPT2 training.
        bos_id = self.gpt2.config.bos_token_id
        if bos_id is None:
            bos_id = self.gpt2.config.eos_token_id
        bos = torch.full((B, 1), bos_id, dtype=input_ids.dtype, device=input_ids.device)
        text_ids = torch.cat([bos, input_ids], dim=1)

        text_embeds = self._get_text_embedding_layer()(text_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Build Masks and Labels (Visual tokens are not predicted, so label = -100)
        bos_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        text_mask = torch.cat([bos_mask, attention_mask], dim=1)
        v_mask = torch.ones((B, visual_embeds.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([v_mask, text_mask], dim=1)
        
        v_labels = torch.full((B, visual_embeds.size(1) + 1), -100, dtype=labels.dtype, device=labels.device)
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
        input_mask_gop: Optional[torch.Tensor] = None,
        target_delta: Optional[torch.Tensor] = None  
    ) -> dict:
        # Pre-extracted tensors are fp16 on disk; keep module math in fp32 for stability.
        clip_i_cls = clip_i_cls.float()
        clip_i_spatial = clip_i_spatial.float()
        motion_vectors = motion_vectors.float()

        B, G = clip_i_cls.shape[:2]
        num_mv = motion_vectors.shape[2]
        
        cls_flat = clip_i_cls.view(B * G, 768)
        spatial_flat = clip_i_spatial.view(B * G, 196, 768)
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        
        # Flatten mask for motion perceiver
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)

        if input_mask_gop is None:
            input_mask_gop = torch.zeros((B, G), dtype=torch.long, device=clip_i_cls.device)
        if input_mask_gop.dtype == torch.bool:
            valid_gop = (~input_mask_gop).float()
        else:
            valid_gop = (1 - input_mask_gop.float()).clamp(min=0.0, max=1.0)

        # PHASE 1: DISTILLATION
        if self.training and self.phase == 1:
            pred_delta = self._forward_motion_delta(bg_mvs, flat_mask_mv, spatial_flat)
            return {
                "teacher_predicted": pred_delta,
                "teacher_target": target_delta.view(B * G, 196, 768).float() if target_delta is not None else None
            }

        # PHASE 2: Selector warmup (no GPT / no projection-to-GPT loss)
        if self.training and self.phase == 2:
            # Cross-attention is done per GOP: [B,G,196,768] x [B,G,8,768]
            motion_tokens = self._forward_motion_tokens_for_mgdtr(bg_mvs, flat_mask_mv)
            motion_tokens = motion_tokens.view(B, G, motion_tokens.shape[1], motion_tokens.shape[2])
            mgdtr_out = self.mgdtr(clip_i_spatial.float(), motion_tokens)

            # Per-GOP motion intensity for variable token allocation (sum per sample = 64).
            motion_intensity = motion_tokens.pow(2).mean(dim=(-1, -2)).sqrt() * valid_gop  # [B, G]
            base_tokens = (2.0 * valid_gop).long()  # base 2 tokens per valid GOP
            remain_tokens = (64 - base_tokens.sum(dim=1)).clamp(min=0)  # [B]

            weights = motion_intensity / motion_intensity.sum(dim=1, keepdim=True).clamp(min=1e-6)
            extra_float = remain_tokens.unsqueeze(1).float() * weights
            extra_tokens = torch.floor(extra_float).long()
            frac = extra_float - extra_tokens.float()

            tokens_per_gop = base_tokens + extra_tokens
            residual = (remain_tokens - extra_tokens.sum(dim=1)).tolist()
            for b in range(B):
                r = int(residual[b])
                if r <= 0:
                    continue
                valid_idx = torch.where(valid_gop[b] > 0.5)[0]
                if valid_idx.numel() == 0:
                    continue
                order = valid_idx[torch.argsort(frac[b, valid_idx], descending=True)]
                for i in range(r):
                    tokens_per_gop[b, order[i % order.numel()]] += 1

            tokens_per_gop = tokens_per_gop * valid_gop.long()

            return {
                "patches_g_mean": mgdtr_out["patches_g_mean"],   # [B, G, 768]
                "cls_tokens": clip_i_cls.float(),   # [B, G, 768]
                "coverage_counts": tokens_per_gop.float(),        # [B, G], sum=64 over valid GOPs
                "selector_prob": mgdtr_out["selector_prob"],
                "valid_mask": valid_gop,
                "tokens_per_gop": tokens_per_gop,
            }

        # PHASE 2 & 3: GENERATIVE SFT
        # In phase 3, route I-frame spatial tokens through MGDTR so cross-attn is trained jointly.
        if self.phase == 3:
            motion_tokens_for_mgdtr = self._forward_motion_tokens_for_mgdtr(bg_mvs, flat_mask_mv)
            motion_tokens_for_mgdtr = motion_tokens_for_mgdtr.view(B, G, motion_tokens_for_mgdtr.shape[1], motion_tokens_for_mgdtr.shape[2])
            mgdtr_out = self.mgdtr(clip_i_spatial.float(), motion_tokens_for_mgdtr)
            refined_cls = 0.5 * (clip_i_cls.float() + mgdtr_out["patches_g_mean"])  # [B,G,768]
            cls_flat = refined_cls.view(B * G, 768)

        cls_embeds, spatial_embeds = self._build_phase3_visual_tokens(cls_flat, spatial_flat, B=B, G=G)
        motion_embeds = self._forward_motion_tokens(bg_mvs, flat_mask_mv).view(B, -1, 768)

        # Distillation decoder is phase-1 only; do not run it in phase 3.
        pred_delta = None

        outputs = self._gpt2_forward(B, cls_embeds, spatial_embeds, motion_embeds, input_ids, attention_mask, labels)
        
        return {
            "prediction_scores": outputs.logits,
            "teacher_predicted": pred_delta,
            "teacher_target": target_delta.view(B * G, 196, 768).float() if target_delta is not None else None,
            "i_spatial": clip_i_spatial.float(),
            "p_spatial": target_delta.add(clip_i_spatial).float() if target_delta is not None else None,
            "motion_tokens": motion_embeds.view(B, G, -1, 768),
            "valid_mask": valid_gop,
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

        cls_embeds, spatial_embeds = self._build_phase3_visual_tokens(cls_flat, spatial_flat, B=B, G=G)
        
        num_mv = motion_vectors.shape[2]
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)
        
        motion_embeds = self._forward_motion_tokens(bg_mvs, flat_mask_mv).view(B, -1, 768)
        
        visual_embeds = torch.cat([cls_embeds, spatial_embeds, motion_embeds], dim=1)
        num_v_tokens = visual_embeds.size(1)
        
        # No prompt => BOS-only decoding seed.
        if prompt:
            text_inputs = tokenizer(prompt, return_tensors="pt").to(iframes.device)
            text_ids = text_inputs.input_ids
        else:
            bos_id = tokenizer.bos_token_id if getattr(tokenizer, "bos_token_id", None) is not None else tokenizer.eos_token_id
            text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=iframes.device)
        text_embeds = self._get_text_embedding_layer()(text_ids)
        
        inputs_embeds = torch.cat([visual_embeds, text_embeds.repeat(B, 1, 1)], dim=1)
        full_mask = torch.ones((B, num_v_tokens + text_ids.size(1)), dtype=torch.long, device=iframes.device)
        
        output_ids = self.gpt2.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=full_mask,
            pad_token_id=tokenizer.eos_token_id, 
            **kwargs
        )
        decoded_ids = output_ids[0]
        if decoded_ids.numel() > num_v_tokens:
            decoded_ids = decoded_ids[num_v_tokens:]
        return tokenizer.decode(decoded_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_from_features(
        self,
        clip_i_cls: torch.Tensor,
        clip_i_spatial: torch.Tensor,
        motion_vectors: torch.Tensor,
        input_mask_mv: torch.Tensor,
        tokenizer,
        prompt="",
        **kwargs,
    ):
        """Generate captions directly from pre-extracted features (no raw iframe CLIP call)."""
        self.eval()
        if self.gpt2 is None:
            raise RuntimeError("GPT-2 is unloaded for this phase.")

        clip_i_cls = clip_i_cls.float()
        clip_i_spatial = clip_i_spatial.float()
        motion_vectors = motion_vectors.float()

        B, G = clip_i_cls.shape[:2]
        num_mv = motion_vectors.shape[2]
        cls_flat = clip_i_cls.view(B * G, 768)
        spatial_flat = clip_i_spatial.view(B * G, 196, 768)
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)

        if self.phase == 3:
            motion_tokens_for_mgdtr = self._forward_motion_tokens_for_mgdtr(bg_mvs, flat_mask_mv)
            motion_tokens_for_mgdtr = motion_tokens_for_mgdtr.view(B, G, motion_tokens_for_mgdtr.shape[1], motion_tokens_for_mgdtr.shape[2])
            mgdtr_out = self.mgdtr(clip_i_spatial.float(), motion_tokens_for_mgdtr)
            refined_cls = 0.5 * (clip_i_cls.float() + mgdtr_out["patches_g_mean"])
            cls_flat = refined_cls.view(B * G, 768)

        cls_embeds, spatial_embeds = self._build_phase3_visual_tokens(cls_flat, spatial_flat, B=B, G=G)
        motion_embeds = self._forward_motion_tokens(bg_mvs, flat_mask_mv).view(B, -1, 768)
        visual_embeds = torch.cat([cls_embeds, spatial_embeds, motion_embeds], dim=1)
        num_v_tokens = visual_embeds.size(1)

        if prompt:
            text_inputs = tokenizer(prompt, return_tensors="pt").to(visual_embeds.device)
            text_ids = text_inputs.input_ids
        else:
            bos_id = tokenizer.bos_token_id if getattr(tokenizer, "bos_token_id", None) is not None else tokenizer.eos_token_id
            text_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=visual_embeds.device)
        text_embeds = self._get_text_embedding_layer()(text_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds.repeat(B, 1, 1)], dim=1)
        full_mask = torch.ones((B, num_v_tokens + text_ids.size(1)), dtype=torch.long, device=visual_embeds.device)

        output_ids = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        decoded_ids = output_ids[0]
        if decoded_ids.numel() > num_v_tokens:
            decoded_ids = decoded_ids[num_v_tokens:]
        return tokenizer.decode(decoded_ids, skip_special_tokens=True)