import os
import csv

def get_video_stats(dataset_dir, csv_files):
    # 1. Get all video IDs from the directory
    try:
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mp4')]
    except Exception as e:
        print(f"Error listing directory: {e}")
        return

    video_ids_in_dir = {os.path.splitext(f)[0] for f in video_files}
    print(f"Found {len(video_ids_in_dir)} video files in {dataset_dir}")

    # 2. Build mapping of ID -> length from CSVs
    id_to_length = {}
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}")
            continue
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v_id = row['id']
                length = float(row['length'])
                id_to_length[v_id] = length

    print(f"Mapped {len(id_to_length)} video lengths from CSV files.")

    # 3. Analyze lengths of videos actually present
    lengths = []
    missing_ids = []
    for vid in video_ids_in_dir:
        if vid in id_to_length:
            lengths.append(id_to_length[vid])
        else:
            missing_ids.append(vid)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} videos found in directory but missing from CSVs.")
    
    total_count = len(lengths)
    if total_count == 0:
        print("No video lengths found to analyze.")
        return

    # 4. Categorize
    buckets = [
        (0, 10, "0-10s"),
        (10, 15, "10-15s"),
        (15, 30, "15-30s"),
        (30, 40, "30-40s"),
        (40, 50, "40-50s"),
        (50, float('inf'), "50+s")
    ]
    
    results = {}
    for start, end, label in buckets:
        count = sum(1 for l in lengths if start <= l < end)
        percentage = (count / total_count) * 100
        results[label] = (count, percentage)

    print("\nVideo Duration Analysis Results:")
    print("-" * 40)
    print(f"{'Range':<10} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 40)
    for label in ["0-10s", "10-15s", "15-30s", "30-40s", "40-50s", "50+s"]:
        count, pct = results[label]
        print(f"{label:<10} | {count:<8} | {pct:>9.2f}%")
    print("-" * 40)
    print(f"Total analyzed: {total_count}")

if __name__ == "__main__":
    BASE_DIR = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/Hav-Cocap/Hav-Cocap/dataset"
    DATASET_DIR = os.path.join(BASE_DIR, "Charades_v1_480")
    CSV_PATH = os.path.join(BASE_DIR, "Charades")
    csv_files = [
        os.path.join(CSV_PATH, "Charades_v1_train.csv"),
        os.path.join(CSV_PATH, "Charades_v1_test.csv")
    ]
    get_video_stats(DATASET_DIR, csv_files)
