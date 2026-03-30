import hydra
from omegaconf import DictConfig
from havcocap_new.modeling.lm_cocap import CoCapLM
from havcocap_new.data.datasets.compressed_video.dataset_charades import CharadesCaptioningDataset
import torch

@hydra.main(config_path="configs", config_name="config")
def debug_batch(cfg: DictConfig):
    # Instantiate dataset
    dataset = CharadesCaptioningDataset(
        video_root="dataset/Charades/Charades_filtered_240",
        metadata="dataset/Charades/charades_filter_captioning.json",
        video_reader="read_frames_compressed_domain",
        max_frames=16,
        video_size=[224, 224],
        max_words=77,
        unfold_sentences=True,
        cv_config=cfg.dataset.cv_config,
        split="train"
    )
    
    item = dataset[0]
    
    # Manually collate into a batch of size 1
    batch = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).cpu()
        else:
            batch[k] = [v]
            
    # Remove audio if not using
    if "audio" in batch and "use_audio" not in cfg.model:
        batch.pop("audio", None)
    
    model = hydra.utils.instantiate(cfg.model)
    model.cpu()
    model.train()
    
    print("Running forward pass...")
    try:
        outputs = model.model(batch)
        print("Forward pass successful!")
        loss = model.loss(batch, outputs)
        print("Loss calculation successful!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    sys.argv = ["debug_cpu.py", "--config-name", "exp/train/charades_captioning"]
    debug_batch()
