import hydra
from havcocap_new.data.datasets.compressed_video.dataset_charades import CharadesCaptioningDataset
from havcocap_new.data.datasets.compressed_video.video_text_base import CVConfig

print("Init CVConfig...")
config = CVConfig(num_gop=16, num_mv=59, num_res=59, with_residual=True, use_pre_extract=False, sample="rand")

print("Init Dataset...")
ds = CharadesCaptioningDataset(
    video_root="./dataset/Charades/Charades_filtered_240",
    metadata="./dataset/Charades/charades_filter_captioning.json",
    split="val",
    max_words=77,
    max_frames=16,
    unfold_sentences=True,
    video_size=(224, 224),
    video_reader="read_frames_compressed_domain",
    cv_config=config,
)

print(f"Dataset size: {len(ds)}")
print("Testing first item...")
for i in range(min(5, len(ds))):
    print(f"Reading item {i}...")
    item = ds[i]
    print(f"Success! I-frames shape: {item['video'][0].shape}")
