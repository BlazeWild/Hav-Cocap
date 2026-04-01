# -*- coding: utf-8 -*-
# @Time    : 8/6/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : compressed_video_captioner.py

__all__ = [
    "CaptionHead",
    "CompressedVideoCaptioner",
    "caption_head_cfg",
    "caption_head_pretrained_cfg",
    "compressed_video_captioner_cfg",
    "compressed_video_captioner_pretrained_cfg",
]

import logging
from typing import *

import numpy as np
import torch
from easydict import EasyDict as edict
from hydra_zen import builds
from torch import Tensor
from torch import nn

from cocap.modules.bert import BertSelfEncoder, BertLMPredictionHead
from cocap.modules.clip.clip import get_model_path
from cocap.modules.clip.model import CLIP
from cocap.modules.compressed_video.compressed_video_transformer import CompressedVideoTransformer, \
    compressed_video_transformer_pretrained_cfg, compressed_video_transformer_cfg

logger = logging.getLogger(__name__)


class CaptionHead(nn.Module):

    def __init__(
            self,
            word_embedding_size: int, visual_feature_size: int,
            max_v_len: int, max_t_len: int, hidden_size: int,
            vocab_size: int, verbose: Optional[Union[int, bool]] = False
    ):
        super(CaptionHead, self).__init__()
        self.model_network = "Self"
        self.cap_config = edict(
            word_vec_size=word_embedding_size,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=hidden_size,
            video_feature_size=visual_feature_size,
            layer_norm_eps=1e-12,  # bert layernorm
            hidden_dropout_prob=0.1,  # applies everywhere except attention
            num_hidden_layers=2,  # number of transformer modules
            num_attention_heads=8,
            share_wd_cls_weight=False,
            vocab_size=vocab_size,
            BOS_id=vocab_size - 2,
            EOS_id=vocab_size - 1,
            PAD_id=0
        )
        logger.debug("Caption Head Configuration: %s", self.cap_config)
        self.cap_sa_decoder = BertSelfEncoder(self.cap_config)
        self.prediction_head = BertLMPredictionHead(self.cap_config, self.cap_sa_decoder.word_embeddings.weight)
        # debug output cfgs
        if verbose:
            if isinstance(verbose, bool):
                self.log_interval = 1
            else:
                self.log_interval = int(verbose)
        else:
            self.log_interval = float("inf")
        self.step_counter = 1

    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionHead.ids2text(predict_ids)

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids: Union[np.ndarray, Tensor]):
        from cocap.trainer.cocap_trainer import convert_ids_to_sentence
        if isinstance(gt_ids, np.ndarray) or isinstance(gt_ids, Tensor):
            assert 0 < len(gt_ids.shape) <= 2, f"gt_ids should be a 1 dim or 2 dim array/tensor, got {gt_ids.shape}"
        else:
            raise ValueError("gt_ids should be np.ndarray or Tensor")
        if isinstance(gt_ids, Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()
        if len(gt_ids.shape) == 1:
            return convert_ids_to_sentence(gt_ids.tolist())
        else:
            return [convert_ids_to_sentence(_gt_ids) for _gt_ids in gt_ids.tolist()]

    def forward(self, visual_output, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len
        
        f_ctx = visual_output["feature_context"] # [bsz n_gop c]
        f_mot = visual_output["feature_motion"] # [bsz n_gop n_bp 16 c]

        # flatten GOP, BP, and the 16 tokens into a continous sequence
        bsz, n_gop, n_bp, num_tokens, c = f_mot.shape
        f_mot_flat = f_mot.reshape(bsz, n_gop*n_bp*num_tokens, c)

        input_types = torch.concat(
            [
                torch.full((f_ctx.size(0), f_ctx.size(1)), 1, dtype=torch.long, device=f_ctx.device),
                torch.full((f_mot_flat.size(0), f_mot_flat.size(1)), 0, dtype=torch.long, device=f_mot_flat.device),
                torch.full((input_ids.size(0), input_ids.size(1)), 2, dtype=torch.long, device=input_ids.device)
            ], dim=1
        )
        visual_concat= torch.cat([f_ctx, f_mot_flat], dim=1)

        input_mask = torch.concat(
            [
                torch.ones(size=(visual_concat.size(0), visual_concat.size(1)),
                           dtype=torch.long, device=visual_concat.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_concat, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])

        if self.step_counter % self.log_interval == 0:
            logger.debug("GT  : %s", self.ids2text(input_ids))
            logger.debug("Pred: %s", self.probability2text(prediction_scores))
        self.step_counter += 1
        return prediction_scores

    @classmethod
    def from_pretrained(
            cls,
            pretrained_clip_name_or_path: str = "ViT-B/16", max_v_len: int = 1928, max_t_len: int = 77,
            verbose: Optional[Union[int, bool]] = False
    ):
        model_path = get_model_path(pretrained_clip_name_or_path, download_root="model_zoo/clip_model")
        pretrained_model: CLIP = torch.jit.load(model_path, map_location="cpu")
        state_dict = pretrained_model.state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]

        head = cls(
            word_embedding_size=transformer_width,
            visual_feature_size=embed_dim,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=embed_dim,
            vocab_size=vocab_size,
            verbose=verbose
        )
        logger.debug(
            "Pretrained embedding parameters: %s",
            [k for k, v in state_dict.items() if k.startswith("token_embedding")]
        )
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        head.cap_sa_decoder.word_embeddings.load_state_dict(pretrained_embedding, strict=True)
        head.prediction_head.decoder.load_state_dict(pretrained_embedding, strict=True)
        assert torch.equal(head.cap_sa_decoder.word_embeddings.weight, head.prediction_head.decoder.weight)
        return head


class TeacherTransformer(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8, layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, iframe_spatial, mv_tokens):
        # iframe_spatial: [Batch, 196, c]
        # mv_tokens: [Batch, num_tokens, c]
        x = torch.cat([iframe_spatial, mv_tokens], dim=1) # [Batch, 196 + num_tokens, c]
        out = self.transformer(x)
        # Predict only the 196 spatial patches of the P-frame
        return self.proj(out[:, :196, :]) 


class CompressedVideoCaptioner(nn.Module):
    def __init__(
            self,
            compressed_video_transformer: CompressedVideoTransformer,
            caption_head: CaptionHead,
            motion_dropout_prob: float = 0.2,
            residual_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.compressed_video_transformer = compressed_video_transformer
        self.caption_head = caption_head
        
        # ADDED THE TEACHER
        self.teacher = TeacherTransformer(d_model=caption_head.cap_config.video_feature_size)
        self.dropout_motion = nn.Dropout(motion_dropout_prob)

        if hasattr(self.compressed_video_transformer.motion_encoder, 'output_dim'):
            mv_width = self.compressed_video_transformer.motion_encoder.output_dim
            self.mv_proj = nn.Linear(mv_width, caption_head.cap_config.video_feature_size)
        else:
            self.mv_proj = nn.Identity()

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]]):
        if "visual_output" not in inputs:
            iframe = inputs["video"]["iframe"]
            motion = self.dropout_motion(inputs["video"]["motion_vector"])
            residual = inputs["video"]["residual"] / 128 - 1  
            bp_type_ids = inputs["video"]["type_ids_mv"]
            bp_rgb = inputs["video"].get("bp_rgb", None)

            compressed_visual_features = self.compressed_video_transformer(
                iframe=iframe, motion=motion, residual=residual, 
                bp_type_ids=bp_type_ids, bp_rgb=bp_rgb
            )
        else:
            compressed_visual_features = inputs["visual_output"]

        f_mot = compressed_visual_features["feature_motion"]
        if f_mot.size(-1) != self.caption_head.cap_config.video_feature_size:
            compressed_visual_features["feature_motion"] = self.mv_proj(f_mot)

        prediction_scores = self.caption_head(
            compressed_visual_features, inputs["input_ids"], inputs["input_mask"]
        )
        
        ret_dict = {
            "prediction_scores": prediction_scores, 
            "visual_output": compressed_visual_features
        }

        # --- TEACHER LOGIC (196 SPATIAL TOKENS) ---
        if self.training and "bp_rgb" in compressed_visual_features and compressed_visual_features["bp_rgb"] is not None:
            bp_rgb = compressed_visual_features["bp_rgb"]
            res = compressed_visual_features["residual"]
            
            bsz, n_gop, n_bp, num_tokens, c_mv = compressed_visual_features["feature_motion"].shape
            
            # 1. Pure Motion Frame = Decoded Frame - Residual
            pure_motion_rgb = bp_rgb - res
            pure_motion_rgb = torch.clamp(pure_motion_rgb, min=0.0, max=1.0)
            flat_rgb = pure_motion_rgb.reshape(-1, 3, pure_motion_rgb.size(-2), pure_motion_rgb.size(-1))
            
            # 2. Get Ground Truth 196 Patches from CLIP
            with torch.no_grad():
                gt_outputs = self.compressed_video_transformer.rgb_encoder(flat_rgb, output_all_features=True)
                gt_spatial = gt_outputs[1] # outputs[1] contains the spatial patches
            
            # 3. Setup Teacher Inputs
            i_spatial = compressed_visual_features["feature_context_spatial"] # [bsz, n_gop, 196, c]
            i_spatial_expanded = i_spatial.unsqueeze(2).expand(bsz, n_gop, n_bp, 196, -1).reshape(-1, 196, gt_spatial.size(-1))
            
            mv_tokens = compressed_visual_features["feature_motion"] # [bsz, n_gop, n_bp, num_tokens, c]
            mv_tokens_flat = mv_tokens.reshape(-1, num_tokens, mv_tokens.size(-1))
            
            # 4. Predict and Save
            predicted_spatial = self.teacher(i_spatial_expanded, mv_tokens_flat)
            
            ret_dict["teacher_predicted"] = predicted_spatial
            ret_dict["teacher_target"] = gt_spatial

        return ret_dict

# Build configs for organizing modules with hydra
caption_head_cfg = builds(CaptionHead, populate_full_signature=True)
caption_head_pretrained_cfg = builds(CaptionHead.from_pretrained, populate_full_signature=True)

compressed_video_captioner_cfg = builds(
    CompressedVideoCaptioner,
    compressed_video_transformer=compressed_video_transformer_cfg,
    caption_head=caption_head_cfg,
    populate_full_signature=True
)
compressed_video_captioner_pretrained_cfg = builds(
    CompressedVideoCaptioner,
    compressed_video_transformer=compressed_video_transformer_pretrained_cfg,
    caption_head=caption_head_pretrained_cfg,
    populate_full_signature=True
)
