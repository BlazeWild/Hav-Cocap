#!/usr/bin/env python3
"""Build a unified MSVD caption JSON from split files.

Input files (default):
  - dataset/msvd/captions/msvd_train.json
  - dataset/msvd/captions/msvd_val.json
  - dataset/msvd/captions/msvd_test.json

Output file (default):
  - dataset/msvd/MSVD_caption_combined.json

Output schema mirrors MSVD_caption.json, with an added `val` split key:
{
  "metadata": [{"video_id": str, "sentence": str}, ...],
  "train": [video_id, ...],
  "val": [video_id, ...],
  "test": [video_id, ...]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_split(split_items: List[dict], split_name: str):
    video_ids: List[str] = []
    metadata: List[dict] = []

    for item in split_items:
        vid = item["video_id"]
        video_ids.append(vid)
        captions = item.get("caption", [])
        for sent in captions:
            metadata.append({"video_id": vid, "sentence": sent})

    # preserve first-seen order while removing duplicates
    seen = set()
    ordered_unique_ids = []
    for vid in video_ids:
        if vid not in seen:
            seen.add(vid)
            ordered_unique_ids.append(vid)

    print(f"{split_name}: videos={len(ordered_unique_ids)}, captions={len(metadata)}")
    return ordered_unique_ids, metadata


def main():
    parser = argparse.ArgumentParser(description="Combine MSVD train/val/test caption JSONs")
    parser.add_argument(
        "--captions_dir",
        type=Path,
        default=Path("/teamspace/studios/this_studio/Hav-Cocap/Cocap_NEW/dataset/msvd/captions"),
        help="Directory containing msvd_train.json, msvd_val.json, msvd_test.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/teamspace/studios/this_studio/Hav-Cocap/Cocap_NEW/dataset/msvd/MSVD_caption_combined.json"),
        help="Path to write combined output JSON",
    )
    args = parser.parse_args()

    split_files: Dict[str, Path] = {
        "train": args.captions_dir / "msvd_train.json",
        "val": args.captions_dir / "msvd_val.json",
        "test": args.captions_dir / "msvd_test.json",
    }

    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} file: {path}")

    combined = {
        "metadata": [],
        "train": [],
        "val": [],
        "test": [],
    }

    for split in ("train", "val", "test"):
        items = load_json(split_files[split])
        ids, split_meta = normalize_split(items, split)
        combined[split] = ids
        combined["metadata"].extend(split_meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)

    print(f"Wrote combined file: {args.output}")
    print(f"total videos={len(set(combined['train']) | set(combined['val']) | set(combined['test']))}")
    print(f"total captions={len(combined['metadata'])}")


if __name__ == "__main__":
    main()
