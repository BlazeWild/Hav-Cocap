import os
import argparse
import av
import glob
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress av logging
av.logging.set_level(av.logging.ERROR)

def check_video(video_path):
    """
    Tries to open and decode a video file.
    Returns True if valid, False if corrupt.
    """
    try:
        with av.open(video_path) as container:
            # Try decoding the first video frame
            if not container.streams.video:
                return False
            
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                return True # If we can decode one frame, we assume it's roughly okay for now
            return False # No frames?
    except Exception as e:
        return False

import json

# ... (imports)

def main():
    parser = argparse.ArgumentParser(description="Scan dataset for corrupt video files and optionally delete them.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--delete", action="store_true", help="Delete corrupt files automatically")
    parser.add_argument("--output", type=str, default="corrupt_files.json", help="Path to save the list of corrupt files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        logger.error(f"Dataset root {args.data_root} does not exist.")
        return

    logger.info(f"Scanning {args.data_root} for mp4 files...")
    video_files = glob.glob(os.path.join(args.data_root, "**", "*.mp4"), recursive=True)
    logger.info(f"Found {len(video_files)} video files.")
    
    corrupt_files = []
    
    for video_path in tqdm(video_files, desc="Checking videos"):
        if not check_video(video_path):
            # Normalize path separators for consistency
            corrupt_files.append(os.path.normpath(video_path))
    
    if not corrupt_files:
        logger.info("\nNo corrupt files found! Dataset looks good.")
        return

    logger.info(f"\nFound {len(corrupt_files)} corrupt files:")
    for f in corrupt_files:
        logger.info(f" - {f}")

    # Save to file
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(corrupt_files, f, indent=4)
            logger.info(f"\nSaved list of corrupt files to {args.output}")
        except Exception as e:
            logger.error(f"Failed to save output file: {e}")

    if args.delete:
         # ... (existing delete logic)
        confirm = input(f"\nAre you sure you want to PERMANENTLY DELETE these {len(corrupt_files)} files? (y/n): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for f in corrupt_files:
                try:
                    os.remove(f)
                    deleted_count += 1
                    logger.info(f"Deleted: {f}")
                except Exception as e:
                    logger.error(f"Failed to delete {f}: {e}")
            logger.info(f"\nDeleted {deleted_count} files.")
        else:
            logger.info("Deletion cancelled.")
    else:
        logger.info("\nUse --delete flag to delete these files.")

if __name__ == "__main__":
    main()
