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
# RESIDUAL BRIDGE (shared by CLSBridge and SpatialBridge)
# =====================================================================
class _ResidualBridge(nn.Module):
    """2-layer MLP with residual skip and post-LayerNorm.
    Input and output dims are both `dim` (768). The residual gives gradients
    a direct path from GPT-2 back to CLIP features and prevents the MLP
    from zeroing its output early in training.
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc2(F.gelu(self.fc1(x))))


# =====================================================================
# MS-DTR SELECTOR (Per-GOP Cross-Attention: Motion + Scene)
# K,V = motion tokens + CLS token for scene-aware spatial selection.
# =====================================================================
class MGDTRSelector(nn.Module):
    def __init__(self, spatial_dim: int = 768, model_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.model_dim = model_dim
        # Motion projection: 256 → 512 → 768
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

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        cls_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        spatial_tokens: [B, G, N, 768]   — CLIP spatial patches (Q)
        motion_tokens : [B, G, M, d]     — motion student output (K,V part 1)
        cls_tokens    : [B, G, 768]      — CLIP CLS tokens (K,V part 2, optional)
                                           When provided, concatenated to motion K,V
                                           so patches score relevance to motion AND scene.
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

        # MS-DTR: Concatenate CLS token to motion K,V for scene-aware selection.
        # K,V = [motion_tokens (M), cls_token (1)] → [BG, M+1, 768]
        # This allows each spatial patch to score relevance to overall scene
        # semantics (via CLS) in addition to motion (via motion tokens).
        if cls_tokens is not None:
            cls_flat = cls_tokens.reshape(B * G, 1, -1)           # [BG, 1, 768]
            kv_input = torch.cat([motion_768, cls_flat], dim=1)   # [BG, M+1, 768]
        else:
            kv_input = motion_768                                  # [BG, M, 768]

        q = self.spatial_ln(spatial_flat)
        kv = self.motion_ln(kv_input)

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
        use_lora: bool = True,
    ):
        super().__init__()

        # 1. Vision Backbones
        self.motion_encoder = motion_encoder
        self._freeze_module(self.motion_encoder)

        # Load CLIP only if provided
        self.clip = None
        if clip_state_dict is not None:
            if isinstance(clip_state_dict, str):
                try:
                    clip_state_dict = torch.jit.load(clip_state_dict, map_location="cpu").state_dict()
                except RuntimeError:
                    clip_state_dict = torch.load(clip_state_dict, map_location="cpu", weights_only=False)
            self.clip = build_clip(clip_state_dict)
            self._freeze_module(self.clip)

        # Store CLIP visual.proj as a frozen buffer for Phase 2 L_align (768 → 512).
        # Copied at init so CLIP can be unloaded without losing the projection.
        if self.clip is not None and hasattr(self.clip, "visual") and hasattr(self.clip.visual, "proj"):
            self.register_buffer("clip_visual_proj", self.clip.visual.proj.detach().clone())
        else:
            self.register_buffer("clip_visual_proj", torch.zeros(spatial_dim, 512))

        # 2. Bridges
        # CLSBridge and SpatialBridge: residual 2-layer MLPs (768→768).
        # MotionBridge (motion_to_gpt): plain Linear 256→768 (dimension change, no residual).
        self.cls_bridge = _ResidualBridge(dim=spatial_dim)
        self.spatial_bridge = _ResidualBridge(dim=spatial_dim)

        self.mgdtr = MGDTRSelector(spatial_dim=spatial_dim, model_dim=gpt_dim, num_heads=8)

        student_dim = 256
        if hasattr(self.motion_encoder, "student") and hasattr(self.motion_encoder.student, "motion_queries"):
            student_dim = int(self.motion_encoder.student.motion_queries.shape[-1])
        self.motion_to_gpt = nn.Linear(student_dim, gpt_dim)

        # 3. Token-group embeddings (one per token type, init to zero → no-op at t=0)
        self.cls_group_emb = nn.Parameter(torch.zeros(1, 1, gpt_dim))
        self.motion_group_emb = nn.Parameter(torch.zeros(1, 1, gpt_dim))
        self.spatial_group_emb = nn.Parameter(torch.zeros(1, 1, gpt_dim))

        # 4. Language Decoder: GPT-2 + optional LoRA (rank=8, Q/V via c_attn, alpha=16)
        base_gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
        if getattr(base_gpt2.config, "loss_type", None) is None:
            base_gpt2.config.loss_type = "ForCausalLMLoss"
        self.using_lora = False
        if use_lora and _HAS_PEFT:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn"],
                lora_dropout=0.1,
                bias="none",
            )
            base_gpt2 = get_peft_model(base_gpt2, lora_cfg)
            self.using_lora = True
        self.gpt2 = base_gpt2
        self._freeze_module(self.gpt2)

        # Default phase at construction
        self.phase = 1
        self.set_training_phase(phase=1)

    @property
    def selector(self):
        return self.mgdtr

    # Backward-compat alias: old checkpoints saved weights under proj_spatial
    @property
    def proj_spatial(self):
        return self.spatial_bridge

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

    def _forward_motion_delta(self, bg_mvs, flat_mask_mv, spatial_flat):
        try:
            return self.motion_encoder(
                bg_mvs, input_mask_mv=flat_mask_mv, iframe_spatial=spatial_flat, return_for_gpt=False,
            )
        except TypeError:
            return self.motion_encoder(bg_mvs, input_mask_mv=flat_mask_mv, iframe_spatial=spatial_flat)

    def _forward_motion_tokens(self, bg_mvs, flat_mask_mv):
        if hasattr(self.motion_encoder, "student"):
            latent = self.motion_encoder.student(bg_mvs, flat_mask_mv)  # [BG, 8, d]
            if latent.shape[-1] != self.motion_to_gpt.in_features:
                raise RuntimeError(
                    f"motion_to_gpt expects input dim={self.motion_to_gpt.in_features}, got {latent.shape[-1]}"
                )
            return self.motion_to_gpt(latent)
        try:
            return self.motion_encoder(bg_mvs, input_mask_mv=flat_mask_mv, return_for_gpt=True)
        except TypeError:
            raise RuntimeError("Motion encoder has no student and does not support return_for_gpt=True")

    def _forward_motion_tokens_for_mgdtr(self, bg_mvs, flat_mask_mv):
        if hasattr(self.motion_encoder, "student"):
            return self.motion_encoder.student(bg_mvs, flat_mask_mv)
        return self._forward_motion_tokens(bg_mvs, flat_mask_mv)

    def set_training_phase(self, phase: int):
        self.phase = phase
        print(f"\n--- Setting Model to Training Phase {phase} ---")

        # Distillation decoder exists only for phase 1
        if phase in (2, 3) and hasattr(self.motion_encoder, "decoder") and self.motion_encoder.decoder is not None:
            self.motion_encoder.decoder = None

        for param in self.parameters():
            param.requires_grad = False

        if phase == 1:
            for param in self.motion_encoder.parameters():
                param.requires_grad = True
            # Kendall log-variance scalars in the loss are trainable (handled by lm_cocap optimizer)

        elif phase == 2:
            for param in self.mgdtr.parameters():
                param.requires_grad = True
            # Sanity: clip_visual_proj must be non-zero (requires CLIP loaded at init).
            if self.clip_visual_proj.abs().max() < 1e-6:
                import warnings
                warnings.warn(
                    "Phase 2: clip_visual_proj is all-zeros — CLIP was not loaded at init. "
                    "L_align projections will be garbage. Pass clip_state_dict to HavCoCapGPT2.",
                    RuntimeWarning, stacklevel=2,
                )

        elif phase == 3:
            # MotionStudent (all layers, tiny LR in optimizer)
            if hasattr(self.motion_encoder, "student"):
                for param in self.motion_encoder.student.parameters():
                    param.requires_grad = True
            # Bridges
            for param in self.cls_bridge.parameters():
                param.requires_grad = True
            for param in self.spatial_bridge.parameters():
                param.requires_grad = True
            # MGDTR + projections
            for param in self.mgdtr.parameters():
                param.requires_grad = True
            for param in self.motion_to_gpt.parameters():
                param.requires_grad = True
            if hasattr(self.motion_encoder, "proj_gpt"):
                for param in self.motion_encoder.proj_gpt.parameters():
                    param.requires_grad = True
            # Token-group embeddings
            self.cls_group_emb.requires_grad = True
            self.motion_group_emb.requires_grad = True
            self.spatial_group_emb.requires_grad = True
            # LoRA adapters (frozen GPT-2 base, only LoRA params train)
            if self.using_lora and self.gpt2 is not None:
                for name, param in self.gpt2.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

    # =====================================================================
    # SPATIAL PROCESSOR (Static Pooling Alternative - kept for legacy)
    # =====================================================================
    def _process_spatial_features(self, cls_768, spatial_768):
        B_G = spatial_768.shape[0]
        grid_14x14 = spatial_768.transpose(1, 2).view(B_G, 768, 14, 14)
        grid_7x7 = F.avg_pool2d(grid_14x14, kernel_size=2, stride=2)
        pooled_patches = grid_7x7.view(B_G, 768, 49).transpose(1, 2)
        spatial_prefix = torch.cat([cls_768.unsqueeze(1), pooled_patches], dim=1)
        spatial_embeds = self.spatial_bridge(spatial_prefix)
        return spatial_embeds  # [B*G, 50, 768]

    def _build_phase3_visual_tokens(self, cls_flat: torch.Tensor, spatial_flat: torch.Tensor, B: int, G: int):
        """Build visual prefix tokens for Phase 3 captioning.

        Prefix layout (168 tokens total):
          positions  0- 7: 8 CLS tokens  (1 per GOP, via cls_bridge + cls_group_emb)
          positions  8-71: 64 motion tokens (8 per GOP, built by caller via motion_to_gpt + motion_group_emb)
          positions 72-167: 96 spatial tokens (12 per GOP, via spatial_bridge + spatial_group_emb)

        Returns cls_tokens [B, G, 768] and spatial_tokens [B, 96, 768].
        Motion tokens are added by the caller.
        """
        # CLS: 1 per GOP
        cls_tokens = self.cls_bridge(cls_flat).view(B, G, 768)        # [B, G, 768]
        cls_tokens = cls_tokens + self.cls_group_emb                   # broadcast [1,1,768]

        # Spatial: pool 196 patches → 12 per GOP (3×4 grid), apply bridge
        grid_14x14 = spatial_flat.transpose(1, 2).view(B * G, 768, 14, 14)
        grid_3x4 = F.adaptive_avg_pool2d(grid_14x14, output_size=(3, 4))
        spatial_12 = grid_3x4.view(B * G, 768, 12).transpose(1, 2)   # [BG, 12, 768]
        spatial_tokens = self.spatial_bridge(spatial_12)               # [BG, 12, 768]
        spatial_tokens = spatial_tokens.view(B, G * 12, 768)          # [B, 96, 768]
        spatial_tokens = spatial_tokens + self.spatial_group_emb       # broadcast [1,1,768]

        return cls_tokens, spatial_tokens

    def _gpt2_forward(self, B, cls_embeds, motion_embeds, spatial_embeds, input_ids, attention_mask, labels):
        if self.gpt2 is None:
            raise RuntimeError("GPT-2 is unloaded for this phase.")
        # Prefix order: [8 CLS] + [64 Motion] + [96 Spatial] = 168 tokens
        visual_embeds = torch.cat([cls_embeds, motion_embeds, spatial_embeds], dim=1)

        bos_id = self.gpt2.config.bos_token_id
        if bos_id is None:
            bos_id = self.gpt2.config.eos_token_id
        bos = torch.full((B, 1), bos_id, dtype=input_ids.dtype, device=input_ids.device)
        text_ids = torch.cat([bos, input_ids], dim=1)

        text_embeds = self._get_text_embedding_layer()(text_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

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
        input_mask_mv: torch.Tensor,
        input_mask_gop: Optional[torch.Tensor] = None,
        target_delta: Optional[torch.Tensor] = None,
    ) -> dict:
        clip_i_cls = clip_i_cls.float()
        clip_i_spatial = clip_i_spatial.float()
        motion_vectors = motion_vectors.float()

        B, G = clip_i_cls.shape[:2]
        num_mv = motion_vectors.shape[2]

        cls_flat = clip_i_cls.view(B * G, 768)
        spatial_flat = clip_i_spatial.view(B * G, 196, 768)
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)

        if input_mask_gop is None:
            input_mask_gop = torch.zeros((B, G), dtype=torch.long, device=clip_i_cls.device)
        if input_mask_gop.dtype == torch.bool:
            valid_gop = (~input_mask_gop).float()
        else:
            valid_gop = (1 - input_mask_gop.float()).clamp(min=0.0, max=1.0)

        # ------------------------------------------------------------------
        # PHASE 1: DISTILLATION
        # ------------------------------------------------------------------
        if self.training and self.phase == 1:
            pred_delta = self._forward_motion_delta(bg_mvs, flat_mask_mv, spatial_flat)
            return {
                "teacher_predicted": pred_delta,
                "teacher_target": target_delta.view(B * G, 196, 768).float() if target_delta is not None else None,
            }

        # ------------------------------------------------------------------
        # PHASE 2: Selector warmup — CLIP text embedding alignment
        # Token budget: K=96, ρ=4 base per GOP, variable budget=64
        # ------------------------------------------------------------------
        if self.training and self.phase == 2:
            motion_tokens = self._forward_motion_tokens_for_mgdtr(bg_mvs, flat_mask_mv)
            motion_tokens = motion_tokens.view(B, G, motion_tokens.shape[1], motion_tokens.shape[2])
            # MS-DTR: pass CLS tokens so selector uses motion + scene for scoring
            mgdtr_out = self.mgdtr(clip_i_spatial.float(), motion_tokens, cls_tokens=clip_i_cls.float())

            # MG-DTR: base ρ=4 tokens per valid GOP, variable budget = 96 − 4*G_valid
            motion_intensity = motion_tokens.pow(2).mean(dim=(-1, -2)).sqrt() * valid_gop  # [B, G]
            base_tokens = (4.0 * valid_gop).long()                     # ρ=4 base per valid GOP
            remain_tokens = (96 - base_tokens.sum(dim=1)).clamp(min=0) # [B] variable budget

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

            # Compute selected_mean_512 for CLIP text alignment:
            # Average patch means over valid GOPs → project 768→512 via frozen visual.proj
            valid_gop_f = valid_gop.float()  # [B, G]
            patches_mean_768 = (
                mgdtr_out["patches_g_mean"] * valid_gop_f.unsqueeze(-1)
            ).sum(dim=1) / valid_gop_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 768]
            selected_mean_512 = F.normalize(patches_mean_768 @ self.clip_visual_proj, dim=-1)  # [B, 512]

            return {
                "patches_g_mean": mgdtr_out["patches_g_mean"],      # [B, G, 768]
                "selected_mean_512": selected_mean_512,              # [B, 512] — L2-normalized
                # Soft per-GOP coverage = selector_prob.sum over patches [B, G].
                # Gradients flow through here into score_head → cross_attn → MGDTR.
                # tokens_per_gop is the integer budget; use soft counts for the loss.
                "coverage_counts": mgdtr_out["coverage_counts"],     # [B, G] soft, differentiable
                "selector_prob": mgdtr_out["selector_prob"],
                "valid_mask": valid_gop,
                "tokens_per_gop": tokens_per_gop,                   # integer budget (informational)
            }

        # ------------------------------------------------------------------
        # PHASE 3: GENERATIVE SFT
        # Prefix: [8 CLS] + [64 Motion] + [96 Spatial] = 168 tokens
        # ------------------------------------------------------------------
        if self.phase == 3:
            motion_tokens_for_mgdtr = self._forward_motion_tokens_for_mgdtr(bg_mvs, flat_mask_mv)
            motion_tokens_for_mgdtr = motion_tokens_for_mgdtr.view(
                B, G, motion_tokens_for_mgdtr.shape[1], motion_tokens_for_mgdtr.shape[2]
            )
            mgdtr_out = self.mgdtr(clip_i_spatial.float(), motion_tokens_for_mgdtr)
            refined_cls = 0.5 * (clip_i_cls.float() + mgdtr_out["patches_g_mean"])  # [B, G, 768]
            cls_flat = refined_cls.view(B * G, 768)

        cls_embeds, spatial_embeds = self._build_phase3_visual_tokens(cls_flat, spatial_flat, B=B, G=G)
        # Motion: 8 tokens per GOP × G GOPs = 64, projected + group embedding
        motion_embeds = (
            self._forward_motion_tokens(bg_mvs, flat_mask_mv).view(B, -1, 768)
            + self.motion_group_emb
        )

        outputs = self._gpt2_forward(B, cls_embeds, motion_embeds, spatial_embeds, input_ids, attention_mask, labels)

        return {
            "prediction_scores": outputs.logits,
            "teacher_predicted": None,
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
    def generate(self, iframes, motion_vectors, input_mask_mv, tokenizer, prompt="", **kwargs):
        assert self.clip is not None, "You must provide clip_state_dict to use generate() on raw iframes."
        self.eval()
        B, G = iframes.shape[:2]

        _, spatial_curr = self.clip.encode_image(iframes.view(-1, 3, 224, 224), output_all_features=True)
        spatial_curr = spatial_curr.float()
        cls_flat = spatial_curr[:, 0, :]
        spatial_flat = spatial_curr[:, 1:, :]

        num_mv = motion_vectors.shape[2]
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)

        # Phase 3: MGDTR CLS refinement — must match training forward path
        if self.phase == 3:
            motion_tokens_for_mgdtr = self._forward_motion_tokens_for_mgdtr(bg_mvs, flat_mask_mv)
            motion_tokens_for_mgdtr = motion_tokens_for_mgdtr.view(
                B, G, motion_tokens_for_mgdtr.shape[1], motion_tokens_for_mgdtr.shape[2]
            )
            mgdtr_out = self.mgdtr(spatial_flat.view(B, G, 196, 768), motion_tokens_for_mgdtr)
            refined_cls = 0.5 * (cls_flat.view(B, G, 768) + mgdtr_out["patches_g_mean"])
            cls_flat = refined_cls.view(B * G, 768)

        cls_embeds, spatial_embeds = self._build_phase3_visual_tokens(cls_flat, spatial_flat, B=B, G=G)

        motion_embeds = (
            self._forward_motion_tokens(bg_mvs, flat_mask_mv).view(B, -1, 768)
            + self.motion_group_emb
        )

        # Prefix order: [8 CLS] + [64 Motion] + [96 Spatial]
        visual_embeds = torch.cat([cls_embeds, motion_embeds, spatial_embeds], dim=1)
        num_v_tokens = visual_embeds.size(1)

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
            **kwargs,
        )
        decoded_ids = output_ids[0]
        if decoded_ids.numel() > num_v_tokens:
            decoded_ids = decoded_ids[num_v_tokens:]
        return tokenizer.decode(decoded_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_from_features(
        self,
        clip_i_cls,
        clip_i_spatial,
        motion_vectors,
        input_mask_mv,
        tokenizer,
        prompt="",
        **kwargs,
    ):
        """Generate captions directly from pre-extracted features."""
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
            motion_tokens_for_mgdtr = motion_tokens_for_mgdtr.view(
                B, G, motion_tokens_for_mgdtr.shape[1], motion_tokens_for_mgdtr.shape[2]
            )
            mgdtr_out = self.mgdtr(clip_i_spatial.float(), motion_tokens_for_mgdtr)
            refined_cls = 0.5 * (clip_i_cls.float() + mgdtr_out["patches_g_mean"])
            cls_flat = refined_cls.view(B * G, 768)

        cls_embeds, spatial_embeds = self._build_phase3_visual_tokens(cls_flat, spatial_flat, B=B, G=G)
        motion_embeds = (
            self._forward_motion_tokens(bg_mvs, flat_mask_mv).view(B, -1, 768)
            + self.motion_group_emb
        )
        # Prefix order: [8 CLS] + [64 Motion] + [96 Spatial]
        visual_embeds = torch.cat([cls_embeds, motion_embeds, spatial_embeds], dim=1)
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
