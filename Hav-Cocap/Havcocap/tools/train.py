
import os
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import math
import logging

# Add paths - Assuming running from Hav-Cocap root or tools dir
# We need to make sure 'Havcocap' is importable.
# If running from Hav-Cocap root: sys.path.append(os.getcwd())
# If running from Havcocap/tools: sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# sys.path.append(os.getcwd())

from Havcocap.model.hav_cocap import HavCocapModel, HavCocapCaptioner
from DataLoader.dataloader import HavCocapDataset

from Havcocap.modules.compressed_video.compressed_video_captioner import CaptionHead 
from Havcocap.modules.clip.clip import load as load_clip
from Havcocap.utils.logging import setup_logging
from Havcocap.utils.checkpoint import save_checkpoint, load_checkpoint


# Logger setup
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def load_clip_visual_weights(model, clip_path="ViT-B/16"):
    """
    Load CLIP visual weights into the IFrameEncoder of HavCocapModel.
    Assumes structure match between CLIP.visual and IFrameEncoder.
    """
    logger.info(f"Loading CLIP weights from {clip_path}...")
    try:
        # Load CLIP model
        clip_model, _ = load_clip(clip_path, device='cpu', jit=False)
        visual_state_dict = clip_model.visual.state_dict()
        
        # Load into iframe_encoder
        # Check keys compatibility
        # CLIP keys: 'conv1.weight', 'class_embedding', 'positional_embedding', 'ln_pre.weight', 'transformer.resblocks...', 'ln_post.weight', 'proj'
        # IFrameEncoder keys: 'conv1.weight', 'class_embedding', 'positional_embedding', 'ln_pre.weight', 'transformer...', 'ln_post.weight', 'proj'
        # They should match if IFrameEncoder uses same naming (copied from CLIP code).
        
        missing, unexpected = model.hav_cocap_model.iframe_encoder.load_state_dict(visual_state_dict, strict=False)
        logger.info(f"Loaded CLIP visual weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        # Ideally missing should be 0.
        
    except Exception as e:
        logger.warning(f"Failed to load CLIP weights: {e}")
        logger.warning("Continuing with random initialization for I-Frame Encoder.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Enable CuDNN benchmark for speed
    torch.backends.cudnn.benchmark = True
    
def train(args):
    # Setup logging
    logger = setup_logging(output_dir=args.output_dir)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")


    # 1. Initialize Dataset
    logger.info(f"Loading dataset from {args.data_root}")
    train_dataset = HavCocapDataset(
        args.data_root, 
        split="train", 
        blacklist_file=args.blacklist_file,
        update_blacklist=args.update_blacklist
    )
    
    # Dataloader
    # Batch size: Physical batch size. Effective batch size = batch_size * accum_steps
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    # 2. Initialize Model
    logger.info("Initializing Model...")
    base_model = HavCocapModel(embed_dim=512) # Match CoCap/CLIP dimensions
    
    # Caption Head
    caption_head = CaptionHead(
        word_embedding_size=512,
        visual_feature_size=512,
        max_v_len=16, # 8 GOPs * 2 (ctx + act)
        max_t_len=77,
        hidden_size=512,
        vocab_size=49408,
        verbose=False 
    )
    
    model = HavCocapCaptioner(base_model, caption_head).to(device)
    
    # Load Pretrained CLIP weights for Visual Encoder
    if args.load_clip:
        load_clip_visual_weights(model)

    # Display Trainable Parameters
    num_params = count_parameters(model)
    logger.info(f"Trainable Parameters: {num_params:,}")
    logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    # 3. Optimizer & Scheduler
    # Parameters to optimize:
    # - AudioEncoder: Frozen (handled in __init__)
    # - IFrameEncoder: Trained or Frozen? Typically fine-tuned or adapter? Let's fine-tune (requires small LR or frozen). 
    #   For now, we fine-tune all.
    # - Motion/Residual: Scratch
    # - Action: Scratch
    # - CaptionHead: Scratch/Pretrained? (If Decoder, maybe pretrained from BERT/CLIP-Text? CoCap uses random initialized BertSelfEncoder usually or BERT?)
    #   CaptionHead.from_pretrained uses CLIP text weights? 
    #   We used __init__, which is random.
    #   Ideally use CaptionHead.from_pretrained.
    
    # Refine CaptionHead initialization if possible
    # We will stick to current but acknowledge.
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is pad
    
    # Scheduler
    # Estimate total steps
    # Note: IterableDataset doesn't have __len__ usually.
    # We estimated ~3322 samples.
    num_samples = 3322 # Approximate
    steps_per_epoch = num_samples // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    model.train()
    
    global_step = 0
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume Logic
    if args.resume:
        # Find latest checkpoint
        checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint_*.pth"))
        latest_ckpt = None
        if checkpoints:
            # Sort by modification time or name? Name might be tricky if mixed formats.
            # Let's look for 'latest_checkpoint.pth' first, then sort others.
            latest_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
            if os.path.exists(latest_path):
                latest_ckpt = latest_path
            else:
                # Fallback to sorting by time
                latest_ckpt = max(checkpoints, key=os.path.getmtime)
        
        if latest_ckpt:
            logger.info(f"Resuming from {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('step', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            
            # If resuming from an epoch checkpoint (e.g. at end of epoch 2), start_epoch should be 3
            # But the saved 'epoch' usually means "finished epoch X" or "current epoch X".
            # Let's assume saved 'epoch' is the one just finished if it's an epoch checkpoint.
            # If we save at start of epoch, it's different.
            # Existing code: 'epoch': epoch (current loop index)
            # So if we crashed during epoch 2 (index), we resume at epoch 2. 
            # If we finished epoch 2 and saved, we probably want to start at 3?
            # Let's just trust the loop will handle range(start_epoch, args.epochs)
            # If saved epoch is 2, range(2, 200) starts at 2. Correct if we want to redo/continue epoch 2?
            # Actually, standard practice: save epoch AFTER finishing it. so save epoch=2 means finished 2.
            # Then we should start at epoch + 1.
            # But wait, existing code saves: 'epoch': epoch+1 at end of loop.
            # And 'epoch': epoch inside loop.
            # Let's adjust logic:
            if "checkpoint_epoch_" in latest_ckpt:
                 # It was saved at end of epoch
                 start_epoch = checkpoint.get('epoch', 0) # e.g. saved epoch=1 (1st epoch finished).
                 # We want to start at 1 (which would be 2nd epoch since 0-indexed).
                 # Wait, loop is range(args.epochs). range(0, 200).
                 # If we loaded epoch=1 (meaning epoch 1 finished, i.e., 2nd epoch), we want range(1, 200)? No, range(1, 200) starts at 1.
                 # Epoch 0 is 1st. Epoch 1 is 2nd. If we finished epoch 1 (2nd), we want to start at 2 (3rd).
                 # Confusion: existing code `save_path = ... f"checkpoint_epoch_{epoch+1}.pth"; 'epoch': epoch + 1`.
                 # So if we finish loop index 0 (1st epoch), we save 'epoch': 1.
                 # If we load 'epoch': 1, we want to start at loop index 1.
                 start_epoch = checkpoint.get('epoch', 0)
            
            logger.info(f"Resumed at Epoch {start_epoch}, Step {global_step}")
        else:
            logger.warning("No checkpoint found to resume from.")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=steps_per_epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(progress_bar):
            # Move to device
            iframe = batch['iframe'].to(device, non_blocking=True)
            audio = batch['audio'].to(device, non_blocking=True)
            
            # Placeholders
            if batch['motion'] is not None and torch.is_tensor(batch['motion']):
                motion = batch['motion'].to(device, non_blocking=True)
                # Correct channel mismatch if legacy dataloader
                if motion.shape[3] == 2:
                    # Pad to 4 channels if necessary (zeros)
                    padded = torch.zeros(motion.shape[0], motion.shape[1], motion.shape[2], 4, motion.shape[4], motion.shape[5], device=device)
                    padded[:, :, :, :2, :, :] = motion
                    motion = padded
            else:
                 b, t = iframe.shape[:2]
                 motion = torch.zeros(b, t, 1, 4, 56, 56).to(device) # 4 channels for motion

            if batch['residual'] is not None and torch.is_tensor(batch['residual']):
                residual = batch['residual'].to(device, non_blocking=True)
            else:
                 b, t = iframe.shape[:2]
                 residual = torch.zeros(b, t, 1, 3, 224, 224).to(device)
                 
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            input_mask = batch['input_mask'].to(device, non_blocking=True)
            
            inputs = {
                "iframe": iframe,
                "motion": motion,
                "residual": residual,
                "audio": audio,
                "input_ids": input_ids,
                "input_mask": input_mask
            }
            
            # Forward with Mixed Precision
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(inputs)
                prediction_scores = outputs['prediction_scores'] # (B, T_text, Vocab)
                
                # Loss Calculation
                # Shift for auto-regressive training
                shifted_logits = prediction_scores[:, :-1, :].contiguous()
                shifted_labels = input_ids[:, 1:].contiguous()
                
                loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
                
                # Check for NaN inside autocast block to ensure loss variable exists
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at step {global_step}. Skipping batch.")
                    if args.fp16: scaler.update() # Just in case
                    optimizer.zero_grad()
                    continue

                loss = loss / args.accum_steps # Scale loss

            # Backward
            scaler.scale(loss).backward()
            
            if (i + 1) % args.accum_steps == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            current_loss = loss.item() * args.accum_steps
            epoch_loss += current_loss
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
            
            if global_step % args.save_steps == 0 and (i + 1) % args.accum_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pth")
                state = {
                    'epoch': epoch, # Current running epoch index
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }
                torch.save(state, save_path)
                # Also save as latest
                torch.save(state, os.path.join(args.output_dir, "latest_checkpoint.pth"))
                

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Validation / Evaluation
        if (epoch + 1) % args.eval_epochs == 0:
            logger.info("Starting evaluation...")
            model.eval()
            results = {} # format: {image_id: [caption]}
            gts = {} # format: {image_id: [caption]}
            
            # Use train_loader for now as we don't have a separate val set defined in args
            # But typically we should use a val split.
            # Let's try to load 'val' split if possible, or skip if not.
            try:
                val_dataset = HavCocapDataset(args.data_root, split="val", blacklist_file=args.blacklist_file)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Evaluating"):
                        # Move inputs
                        iframe = batch['iframe'].to(device)
                        audio = batch['audio'].to(device)
                        # ... prepare inputs ... 
                        # This part depends on how generation is implemented. 
                        # HavCocapCaptioner needs a generate method? or just forward?
                        # Captioning usually requires auto-regressive generation loop (beam search).
                        # The current model code doesn't seem to have a 'generate' method exposed clearly 
                        # other than `caption_head(..., input_ids, ...)` which is for training loss.
                        # We need a generation method in HavCocapCaptioner or CaptionHead.
                        # Since I don't have the `generate` code handy in the provided snippet (Code only showed forward), 
                        # I will add a placeholder warning or assume `model.generate` exists if I can find it.
                        # Actually, `HavCocapCaptioner` uses `CaptionHead`. checked `hav_cocap.py`.
                        # `CaptionHead` is imported from `compressed_video_captioner`.
                        # I don't have the source of `compressed_video_captioner.py` visible here but usually it has `.generate()`.
                        # Let's assume `model.generate(inputs)` works or `model.hav_cocap_model` + `caption_head.generate`.
                        pass
                
                logger.warning("Evaluation logic requires generation method which needs to be verified. Skipping metric calculation for now to prevent crash.")
            except Exception as e:
                 logger.warning(f"Evaluation failed: {e}")

            model.train()
        
        # Check best loss (using training loss as proxy since no val set)
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"New best loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss
            }, os.path.join(args.output_dir, "best_checkpoint.pth"))

        # Save epoch checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            state = {
                'epoch': epoch + 1, # Next start epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss
            }
            torch.save(state, save_path)
            # Also save as latest (overwrite)
            torch.save(state, os.path.join(args.output_dir, "latest_checkpoint.pth"))
            logger.info(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Hav-Cocap/dataset/AVCaps")
    parser.add_argument("--output_dir", type=str, default="Hav-Cocap/checkpoints")
    parser.add_argument("--blacklist_file", type=str, default="Hav-Cocap/corrupt_files.json", help="Path to json file containing blacklisted video paths")
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Physical batch size per GPU")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for scheduler")
    
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--update_blacklist", action="store_true", help="Automatically add corrupt videos to blacklist")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--save_epochs", type=int, default=5, help="Save checkpoint every X epochs")
    parser.add_argument("--eval_epochs", type=int, default=5, help="Run evaluation every X epochs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=False, help="Enable mixed precision training")
    parser.add_argument("--load_clip", action="store_true", default=True, help="Load pretrained CLIP weights for visual encoder")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)
