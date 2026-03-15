import argparse
import os
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

# Import our custom modules
from cocap.data.datasets.avcaps import AVCapsDataset, avcaps_collate_fn
from cocap.modeling.av_captioner import AVCaptioner
from cocap.modules.clip.simple_tokenizer import SimpleTokenizer
from cocap.modeling.eval_captioning import EvalCap
from cocap.utils.logger import setup_logger # Assuming this exists or use standard logging
# If logging utils not found, use standard:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="AVCaps Training")
    parser.add_argument("--data_root", type=str, default="/teamspace/studios/this_studio/Hav-Cocap/dataset/AVCaps", help="Path to AVCaps dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for DataLoader")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--output_dir", type=str, default="output/avcaps_run", help="Output directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, cider, output_dir, is_best=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'cider': cider
    }
    torch.save(state, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth"))
    if is_best:
        torch.save(state, os.path.join(output_dir, "checkpoint_best.pth"))

def train(args):
    # Setup DDP
    is_distributed = False
    if 'WORLD_SIZE' in os.environ:
        is_distributed = int(os.environ['WORLD_SIZE']) > 1
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_distributed:
        dist.init_process_group(backend='nccl', init_method=args.dist_url)
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        print(f"Initialized DDP on rank {args.local_rank}")
    
    # Enable Cudnn Benchmark
    torch.backends.cudnn.benchmark = True
    
    # Model
    model_zoo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_zoo")
    clip_path = os.path.join(model_zoo_dir, "RN50.pt")
    # a_path = os.path.join(model_zoo_dir, "Cnn14_mAP=0.431.pth")
    # audio_type = "cnn14"
    
    # Switch to BEATs
    a_path = os.path.join(model_zoo_dir, "BEATs_iter3_plus_AS2M.pt")
    audio_type = "beats"
    
    model = AVCaptioner(clip_path=clip_path, audio_path=a_path, audio_enc_type=audio_type)
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Dataset
    tokenizer = SimpleTokenizer()
    train_dataset = AVCapsDataset(root_dir=args.data_root, split="train", tokenizer=tokenizer)
    val_dataset = AVCapsDataset(root_dir=args.data_root, split="val", tokenizer=tokenizer)
    
    # Create GT dictionary for evaluation ONCE
    # val_dataset.data is {vid: {..., 'audio_visual_captions': [...]}}
    # EvalCap expects gts = {id: [{caption: "..."}]}
    val_gts = {}
    for vid, data in val_dataset.data.items():
        if "audio_visual_captions" in data:
            val_gts[str(vid)] = [d for d in data["audio_visual_captions"]]
            
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=avcaps_collate_fn,
        pin_memory=True, # Optimized for GPU
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=avcaps_collate_fn,
        pin_memory=True
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1) 
    
    # AMP Scaler
    scaler = GradScaler()
    
    best_cider = 0.0
    
    if args.local_rank in [-1, 0]:
        logger.info(f"Start training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        
        for i, batch in enumerate(train_loader):
            if batch is None:
                continue # Skip corrupted batches
            
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            audio_spec = batch["audio_spec"].to(device, non_blocking=True) # (B, T, F)
            captions = batch["captions"].to(device, non_blocking=True)
            caption_mask = batch["caption_mask"].to(device, non_blocking=True)
            
            # Input to model: captions[:, :-1]
            # Targets: captions[:, 1:]
            input_captions = captions[:, :-1]
            targets = captions[:, 1:]
            
            attention_mask = caption_mask[:, :-1]
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(pixel_values, audio_spec, input_captions, attention_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if i % args.log_interval == 0 and args.local_rank in [-1, 0]:
                logger.info(f"Epoch [{epoch}][{i}/{len(train_loader)}] Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validation
        if args.local_rank in [-1, 0]:
            logger.info("Evaluating...")
            metrics = evaluate_metrics(model, val_loader, tokenizer, device, val_gts)
            logger.info(f"Epoch {epoch} Results: {metrics}")
            
            cider_score = metrics.get("CIDEr", 0.0)
            is_best = cider_score > best_cider
            best_cider = max(cider_score, best_cider)
            
            save_checkpoint(model, optimizer, epoch, cider_score, args.output_dir, is_best)

@torch.no_grad()
def evaluate_metrics(model, val_loader, tokenizer, device, gts):
    model.eval()
    results = [] 
    
    # Store hypothesis
    # Format for pycocoevalcap: list of dicts [{'image_id': str, 'caption': str}]
    # But EvalCap class expects rests as list of dicts.
    
    res = {}
    
    for batch in val_loader:
        if batch is None: continue
        
        pixel_values = batch["pixel_values"].to(device)
        audio_spec = batch["audio_spec"].to(device)
        video_ids = batch.get("video_ids", [])
        
        # Check generate signature in model: images, audios
        if hasattr(model, 'module'):
            generated_tokens = model.module.generate(pixel_values, audio_spec)
        else:
            generated_tokens = model.generate(pixel_values, audio_spec)
            
        # Decode
        for i, tokens in enumerate(generated_tokens):
            # 49407 is EOT, 49406 is SOT. SimpleTokenizer decode handles them?
            # We usually slice until EOT.
            t_list = tokens.tolist()
            if 49407 in t_list:
                t_list = t_list[:t_list.index(49407)]
            if 49406 in t_list:
                 t_list = t_list[t_list.index(49406)+1:]
                 
            caption = tokenizer.decode(t_list)
            
            vid = str(video_ids[i])
            res[vid] = [{'caption': caption}]
            
    # Run EvalCap
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        
        # We can reuse the EvalCap class provided in cocap/modeling/eval_captioning.py
        # But it requires passing classes not instances in some parts or vice versa.
        # Let's instantiate scorers directly here for simplicity and robustness.
        
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        
        # Tokenization
        # PTBTokenizer expects dict {id: [{'caption': str}]}
        # We have gts and res in that format (almost, Res needs array)
        
        print("Tokenizing...")
        tokenizer_ptb = PTBTokenizer()
        gts_tok = tokenizer_ptb.tokenize(gts)
        res_tok = tokenizer_ptb.tokenize(res)
        
        final_scores = {}
        for scorer, method in scorers:
            print(f"Computing {method}...")
            score, scores = scorer.compute_score(gts_tok, res_tok)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
                
        return final_scores
        
    except ImportError:
        print("pycocoevalcap not installed. Skipping metrics.")
        return {"CIDEr": 0.0}
    except Exception as e:
        print(f"Eval failed: {e}")
        return {"CIDEr": 0.0}

if __name__ == "__main__":
    args = parse_args()
    train(args)
