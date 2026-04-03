#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def load_captions(caption_json_path: Path, video_id: str):
    with caption_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "metadata" in data and isinstance(data["metadata"], list):
            data = data["metadata"]
        elif video_id in data and isinstance(data[video_id], list):
            # Alternate format: {"video_id": ["cap1", "cap2", ...]}
            return [str(x) for x in data[video_id]]
        else:
            data = []

    captions = []
    for x in data:
        if isinstance(x, dict) and x.get("video_id") == video_id:
            sent = x.get("sentence")
            if isinstance(sent, str):
                captions.append(sent)
    return captions


def get_frame_types(video_path: Path):
    # Try ffprobe JSON output first
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pict_type",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")

    obj = json.loads(proc.stdout)
    frames = obj.get("frames", [])
    frame_types = [f.get("pict_type", "") for f in frames]
    if not frame_types:
        raise RuntimeError("No frame types found from ffprobe output")
    return frame_types


def read_all_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded from video")
    return frames


def flow_to_color(prev_rgb, curr_rgb):
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return mv_rgb


def save_rgb_jpg(path: Path, rgb_img):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def asset_name(gop_idx: int, frame_idx: int, tag: str):
    return f"g{gop_idx:02d}_f{frame_idx:05d}_{tag}.jpg"


def main():
    parser = argparse.ArgumentParser(description="Extract GOP assets and captions for one MSVD video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--captions", required=True, help="Path to MSVD_caption_combined.json")
    parser.add_argument("--out-dir", required=True, help="Output folder under tests")
    args = parser.parse_args()

    video_path = Path(args.video)
    captions_path = Path(args.captions)
    out_dir = Path(args.out_dir)

    video_id = video_path.stem
    out_video_dir = out_dir / video_id
    out_video_dir.mkdir(parents=True, exist_ok=True)

    frame_types = get_frame_types(video_path)
    frames = read_all_frames(video_path)

    n = min(len(frame_types), len(frames))
    frame_types = frame_types[:n]
    frames = frames[:n]

    i_indices = [i for i, t in enumerate(frame_types) if t == "I"]
    if not i_indices:
        i_indices = [0]

    gops = []
    for g, start in enumerate(i_indices):
        end = i_indices[g + 1] - 1 if g + 1 < len(i_indices) else n - 1
        gops.append((start, end))

    # Save captions
    caps = load_captions(captions_path, video_id)
    with (out_video_dir / "captions.txt").open("w", encoding="utf-8") as f:
        for i, c in enumerate(caps, 1):
            f.write(f"{i:02d}. {c}\n")

    # Save GOP summary metadata
    metadata = {
        "video": str(video_path),
        "video_id": video_id,
        "num_frames": n,
        "num_gops": len(gops),
        "gops": [{"gop": i + 1, "start": s, "end": e} for i, (s, e) in enumerate(gops)],
    }
    with (out_video_dir / "gop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Rule requested by user:
    # - GOP 1: save I frame(s), P frames, motion visualization, residual image,
    #          and P frame with residual subtracted for all P frames
    # - GOP 2..last: save only I frame

    for gop_idx, (start, end) in enumerate(gops, 1):
        gop_dir = out_video_dir / f"gop_{gop_idx:02d}"

        # Always save I frame at GOP start
        save_rgb_jpg(
            gop_dir / "i_frames" / asset_name(gop_idx, start, "I"),
            frames[start],
        )

        if gop_idx == 1:
            for fi in range(start + 1, end + 1):
                if frame_types[fi] != "P":
                    continue

                # P frame
                save_rgb_jpg(
                    gop_dir / "p_frames" / asset_name(gop_idx, fi, "P"),
                    frames[fi],
                )

                # Residual proxy: absolute difference between current and previous decoded frame
                residual = cv2.absdiff(frames[fi], frames[fi - 1])
                save_rgb_jpg(
                    gop_dir / "residuals" / asset_name(gop_idx, fi, "RESIDUAL"),
                    residual,
                )

                # P frame with residual subtracted (user requested)
                p_minus_residual = cv2.subtract(frames[fi], residual)
                save_rgb_jpg(
                    gop_dir / "p_minus_residuals" / asset_name(gop_idx, fi, "P_MINUS_RESIDUAL"),
                    p_minus_residual,
                )

                # Motion proxy: dense optical-flow color map between previous and current frame
                mv = flow_to_color(frames[fi - 1], frames[fi])
                save_rgb_jpg(
                    gop_dir / "motion_vectors" / asset_name(gop_idx, fi, "MV"),
                    mv,
                )

    print(f"Done. Output written to: {out_video_dir}")
    print(f"Captions found: {len(caps)}")
    print(f"GOPs found: {len(gops)}")


if __name__ == "__main__":
    main()
