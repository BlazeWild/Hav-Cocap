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

print("Video ID for item 0:", ds.sentences[0][0])
print("Video ID for item 1:", ds.sentences[1][0])
