# -*- coding: utf-8 -*-
# @Time    : 6/16/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : cocap.py
import copy
import logging
import os
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra_zen import builds
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LambdaLR

from havcocap_new.modules.bert import BertLayerNorm
from havcocap_new.modules.compressed_video import CompressedVideoCaptioner, compressed_video_captioner_pretrained_cfg
from .eval_captioning import evaluate
from .loss import LossBase, label_smoothing_loss_cfg
from .optimization import BertAdam
from ..utils.json import save_json
from ..utils.train_utils import gather_object_multiple_gpu, get_timestamp

logger = logging.getLogger(__name__)


from transformers import AutoTokenizer
_tinystories_tokenizer = None

def convert_ids_to_sentence(tokens):
    global _tinystories_tokenizer
    if _tinystories_tokenizer is None:
        # Load local TinyStories tokenizer exactly once natively in the scope
        _tinystories_tokenizer = AutoTokenizer.from_pretrained("model_zoo/tinystories-33m")
        
    text = _tinystories_tokenizer.decode(tokens, skip_special_tokens=True)
    return text.strip()


class CoCapLM(pl.LightningModule):
    """CoCap Lightning Module"""

    def __init__(
            self,
            cocap_model: CompressedVideoCaptioner,
            loss: LossBase,
            lr: float = 1e-4,
            clip_lr: float = 1e-6,
            warmup_ratio: float = 0.05,
            lr_decay_gamma: float = 0.95,
    ):
        super().__init__()
        self.model = cocap_model
        self.loss = loss
        self.lr = lr
        self.clip_lr = clip_lr
        self.warmup_ratio = warmup_ratio
        self.lr_decay_gamma = lr_decay_gamma

        self.batch_res = None

    @property
    def total_steps(self):
        return self.trainer.estimated_stepping_batches

    @property
    def epoch_steps(self):
        return self.trainer.estimated_stepping_batches // self.trainer.max_epochs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # based on:
        # https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
        model = self.model

        decay = set()
        no_decay = set()

        pretrained_modules = [
            "compressed_video_transformer.rgb_encoder.conv1",
            "compressed_video_transformer.rgb_encoder.class_embedding",
            "compressed_video_transformer.rgb_encoder.positional_embedding",
            "compressed_video_transformer.rgb_encoder.ln_pre",
            "compressed_video_transformer.rgb_encoder.transformer",
            "compressed_video_transformer.rgb_encoder.ln_post",
            "compressed_video_transformer.rgb_encoder.proj",
            "tinystories",
        ]
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding, BertLayerNorm)
        param_dict = {pn: p for pn, p in model.named_parameters()}
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if fpn not in param_dict:
                    continue

                if any(fpn.startswith(p_fpn) for p_fpn in pretrained_modules):  # pretrained
                    no_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("tokens"):
                    no_decay.add(fpn)
                elif pn.endswith("proj") or pn.endswith("projection"):
                    decay.add(fpn)
                elif fpn.endswith("embedding"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                               any(pn.startswith(p_pn) for p_pn in pretrained_modules)]
        not_pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                                   not any(pn.startswith(p_pn) for p_pn in pretrained_modules)]

        logger.debug("Parameter group decay_param: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(decay))]))
        logger.debug("Parameter group no_decay_pretrained_param: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(pretrained_no_decay))]))
        logger.debug("Parameter group no_decay_not_pretrained_param: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(not_pretrained_no_decay))]))

        decay_param = [param_dict[pn] for pn in sorted(list(decay))]
        no_decay_pretrained_param = [param_dict[pn] for pn in sorted(list(pretrained_no_decay))]
        no_decay_not_pretrained_param = [param_dict[pn] for pn in sorted(list(not_pretrained_no_decay))]

        optimizer_grouped_parameters = [
            {"params": decay_param},
            {"params": no_decay_pretrained_param, "weight_decay": 0.0, "lr": self.clip_lr},
            {"params": no_decay_not_pretrained_param, "weight_decay": 0.0}
        ]

        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=0.01,
            max_grad_norm=1.0
        )

        def lr_lambda(current_step):
            warmup_steps = self.warmup_ratio * self.total_steps
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                return self.lr_decay_gamma ** ((current_step - warmup_steps) // self.epoch_steps)

        # Step-based warmup, epoch-based decay scheduler
        warmup_decay_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_decay_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_decay"
            }
        }
    def setup(self, stage=None):
        # 1. ALWAYS FREEZE I-FRAMES
        for param in self.model.compressed_video_transformer.rgb_encoder.parameters():
            param.requires_grad = False
            
        # 2. AT EPOCH 0, Modality Warmup forces TinyStories to start FROZEN
        for param in self.model.tinystories.parameters():
            param.requires_grad = False

    def on_train_start(self):
        try:
            import torchinfo
            print("\n=== MODEL ARCHITECTURE SUMMARY ===")
            torchinfo.summary(self.model, depth=4)
            print("==================================\n")
        except ImportError:
            import pytorch_lightning as pl
            print("\n=== MODEL ARCHITECTURE SUMMARY ===")
            print(pl.utilities.model_summary.ModelSummary(self, max_depth=3))
            print("==================================\n")

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        print(f"\n--- EPOCH {epoch} START ---")
        
        # 1. ALWAYS FREEZE I-FRAMES (ViT-B/16)
        for param in self.model.compressed_video_transformer.rgb_encoder.parameters():
            param.requires_grad = False
            
        # 2. Phase 1 vs Phase 2
        if epoch < 3:
            print("Phase 1: Modality Warmup - TinyStories is FROZEN")
            for param in self.model.tinystories.parameters():
                param.requires_grad = False
        else:
            print("Phase 2: End-to-End Fine-Tuning - TinyStories is TRAINING")
            for param in self.model.tinystories.parameters():
                param.requires_grad = True
                
        # PROVE UNFREEZE TO USER: Print detailed Model Summary wall for Phase boundaries
        if epoch == 0 or epoch == 3:
            try:
                import torchinfo
                print(f"\n=== MODEL ARCHITECTURE SUMMARY (PHASE {'1' if epoch < 3 else '2'}) ===")
                torchinfo.summary(self.model, depth=4)
                print("==================================\n")
            except ImportError:
                import pytorch_lightning as pl
                print(f"\n=== MODEL ARCHITECTURE SUMMARY (PHASE {'1' if epoch < 3 else '2'}) ===")
                print(pl.utilities.model_summary.ModelSummary(self, max_depth=3))
                print("==================================\n")

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        lm_loss = outputs["lm_loss"]
        aux_loss = outputs.get("aux_loss", 0.0)
        total_loss = lm_loss + aux_loss
        self.log("loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["input_labels"].size(0))
        self.log("lm_loss", lm_loss, on_step=True, on_epoch=True)
        self.log("aux_loss", aux_loss, on_step=True, on_epoch=True)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.batch_res = {"version": "VERSION 1.0",
                          "results": defaultdict(list),
                          "external_data": {"used": "true", "details": "ay"}}

    def validation_step(self, batch, batch_idx):
        inputs_ids = batch["input_ids"]
        input_masks = batch["input_mask"]
        
        # TinyStories max length context usually around 2048, but here we cap it to the sentences length
        max_t_len = inputs_ids.size(1)

        inputs_ids[:, :] = 0.
        input_masks[:, :] = 0.
        assert torch.sum(input_masks[:, :]) == 0, "Initially, all text tokens should be masked"
        bsz = len(inputs_ids)
        # Using 50256 (or similar TinyStories BOS equivalent) if needed. TinyStories often uses <|endoftext|> as BOS.
        # But we can get it from the model configs or tokenizer.
        # we will use the same BOS_id as before for backwards compatibility if valid.
        BOS_id = 50256 # TinyStories BOS token
        next_symbols = torch.IntTensor([BOS_id] * bsz).to(inputs_ids.device)  # (N, )

        warn_visual_output = False
        for dec_idx in range(max_t_len):
            inputs_ids[:, dec_idx] = next_symbols.clone()
            input_masks[:, dec_idx] = 1
            outputs = self.model(batch)
            pred_scores = outputs["prediction_scores"]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words
            if "visual_output" in outputs:
                batch["visual_output"] = outputs["visual_output"]
            elif not warn_visual_output:
                logger.warning("visual_output is not in the output of model, this may slow down the caption test")
                warn_visual_output = True
        dec_seq = inputs_ids

        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, batch['metadata'][1])):
            cur_data = {
                "sentence": convert_ids_to_sentence(cur_gen_sen.tolist()),
                "gt_sentence": cur_meta
            }
            self.batch_res["results"][batch['metadata'][0][example_idx].split("video")[-1]].append(cur_data)

    def on_validation_epoch_end(self) -> None:
        json_res = copy.deepcopy(self.batch_res)
        if dist.is_initialized():
            all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
            json_res['results'] = {k: v for k, v in all_results}
            logger.debug("Caption test length: %s", len(json_res["results"].items()))

        # save result tp log for debug
        if not dist.is_initialized() or dist.get_rank() == 0:
            res_filepath = os.path.join(self.trainer.default_root_dir,
                                        "caption_greedy_pred_validation_{}.json".format(get_timestamp()))
            os.makedirs(os.path.dirname(res_filepath), exist_ok=True)
            save_json(json_res, res_filepath, save_pretty=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            json_ref = self.trainer.val_dataloaders.dataset.json_ref
            metrics = evaluate(json_res, json_ref)
            self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if dist.is_initialized():
            dist.barrier()


cocap_lm_cfg = builds(
    CoCapLM,
    cocap_model=compressed_video_captioner_pretrained_cfg,
    loss=label_smoothing_loss_cfg,
    populate_full_signature=True
)
