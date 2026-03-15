import json
import random
from pathlib import Path

json_path = Path(r"c:\hav_video_captioning\Hav-Cocap_avcaps\CoCap\dataset\msvd\MSVD_caption.json")
output_path = Path(r"c:\hav_video_captioning\Hav-Cocap_avcaps\CoCap\dataset\msvd\MSVD_caption_split.json")

print("Loading MSVD JSON...")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect unique video IDs
metadata = data.get("metadata", [])
unique_vids = list(set([item["video_id"] for item in metadata]))
unique_vids.sort() # sort for deterministic assignment
print(f"Total unique videos: {len(unique_vids)}")

# Standard MSVD Split: 1200 train, 100 val, 670 test
# Since videos are Youtube ID names, we'll shuffle with seed 42 to get a random mix,
# or just take the sorted slice. In standard papers, the split is fixed.
# If we simply slice:
random.seed(42)
random.shuffle(unique_vids)

train_vids = unique_vids[:1200]
val_vids = unique_vids[1200:1300]
test_vids = unique_vids[1300:]

print(f"Train: {len(train_vids)}, Val: {len(val_vids)}, Test: {len(test_vids)}")

# Construct new dict
new_data = {
    "train": train_vids,
    "val": val_vids,
    "test": test_vids,
    "metadata": metadata
}

print("Saving split JSON...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f)
print("Saved smoothly to MSVD_caption_split.json")
