# -*- coding: utf-8 -*-
# @Time    : 8/6/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : compressed_video_captioner.py

__all__ = [
    "CompressedVideoCaptioner",
    "compressed_video_captioner_cfg",
    "compressed_video_captioner_pretrained_cfg",
]

import logging
from typing import Dict, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from hydra_zen import builds

from havcocap_new.modules.compressed_video.compressed_video_transformer import CompressedVideoTransformer, \
    compressed_video_transformer_pretrained_cfg, compressed_video_transformer_cfg

logger = logging.getLogger(__name__)


class DummyTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(self.transformer(x))


class CompressedVideoCaptioner(nn.Module):
    def __init__(
            self,
            compressed_video_transformer: CompressedVideoTransformer,
            tinystories_model_path: str = "model_zoo/tinystories-33m",
            motion_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.compressed_video_transformer = compressed_video_transformer
        
        # Load TinyStories
        config = AutoConfig.from_pretrained(tinystories_model_path)
        self.tinystories = AutoModelForCausalLM.from_pretrained(
            tinystories_model_path, 
            config=config, 
            torch_dtype=torch.bfloat16
        )
        self.embed_dim = self.tinystories.config.hidden_size # normally 768
        
        # Projection bridge if visual dim != tinystories dimension
        visual_dim = self.compressed_video_transformer.output_dim
        if visual_dim != self.embed_dim:
            self.visual_proj = nn.Linear(visual_dim, self.embed_dim)
        else:
            self.visual_proj = nn.Identity()
            
        # Dummy Transformer for aux loss (predicts next I-frame features)
        self.dummy_transformer = DummyTransformer(d_model=self.embed_dim)
        self.dropout_motion = nn.Dropout(motion_dropout_prob)

    def sequence_assembler(self, f_ctx, f_act):
        """
        Concatenates chronologically: I + P1 + P2... P7
        Repeats for all 8 GOPs
        f_ctx: [B, 8, 16, 768]
        f_act: [B, 56, 4, 768] (which is B x 8 GOPs x 7 P-frames x 4 tokens)
        Output: [B, 352, 768]
        """
        b = f_ctx.size(0)
        g = f_ctx.size(1)

        f_act_gop = f_act.reshape(b, g, -1, f_act.size(-1)) # [B, 8, 28, 768]
        
        # Concat in chronological order: I-frames (16 tokens) then P-frames (28 tokens) = 44 tokens per GOP
        gop_tokens = torch.cat([f_ctx, f_act_gop], dim=2) # [B, 8, 44, 768]
        
        seq = gop_tokens.reshape(b, -1, gop_tokens.size(-1)) # [B, 352, 768]
        return seq

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]]):
        if "visual_output" not in inputs:
            iframe = inputs["video"]["iframe"]
            motion = inputs["video"]["motion_vector"]

            motion = self.dropout_motion(motion)
            
            # 1. Forward through our updated compressed video transformer
            features = self.compressed_video_transformer(
                iframe=iframe,
                motion=motion
            )
            
            f_ctx = features["feature_context"]
            f_act = features["feature_action"]
            
            # 2. Sequence Assembler
            visual_seq = self.sequence_assembler(f_ctx, f_act)
            visual_seq = self.visual_proj(visual_seq)
            features["visual_seq"] = visual_seq
        else:
            # reuse pre-extracted visual features
            features = inputs["visual_output"]
            visual_seq = features["visual_seq"]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["input_mask"]
        
        # 3. Brain: TinyStories-33M text generation conditioning
        text_embeds = self.tinystories.transformer.wte(input_ids.long())
        
        # Explicit Downcast: Match the visual prompt to the LLM's memory footprint
        visual_seq = visual_seq.to(dtype=text_embeds.dtype)
        
        inputs_embeds = torch.cat([visual_seq, text_embeds], dim=1)
        
        visual_mask = torch.ones(visual_seq.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([visual_mask, attention_mask], dim=1).long()
        
        # Use Tinystories native generation loss?
        # Only compute generation loss over the text tokens, so mask out the visual tokens
        labels = inputs.get("input_labels", input_ids.clone()).long()
        visual_labels = torch.full(visual_seq.shape[:2], -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([visual_labels, labels], dim=1)
        
        outputs = self.tinystories(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True
        )
        lm_loss = outputs.loss
        
        # 4. Dummy Transformer (The "Fake Pretraining" Hack)
        # Guesses what the next token looks like. The auxiliary MSE loss backpropagates through Temporal Wire
        # We temporarily push the sequence back up to Float32 so the native PyTorch module doesn't crash on BF16
        dummy_out = self.dummy_transformer(visual_seq.to(torch.float32)) 
        
        # Target is the visual_seq shifted by 1 time step
        shifted_preds = dummy_out[:, :-1, :]
        shifted_targets = visual_seq[:, 1:, :].detach().to(torch.float32)
        aux_loss = nn.functional.mse_loss(shifted_preds, shifted_targets)
        
        # Extract text predictions (remove logits corresponding to visual tokens)
        prediction_scores = outputs.logits[:, visual_seq.shape[1]:, :].contiguous()

        return {
            "prediction_scores": prediction_scores, 
            "visual_output": features,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss
        }


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
