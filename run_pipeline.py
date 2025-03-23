import os
import argparse

# Import our backblaze functions from our package
from backblaze_sdk import download_file, upload_json

# Import your YOLO detection functions (assumed to be in the same folder)
from save_yolo_left import save_yolo_left
from save_yolo_right import save_yolo_right

# Import the merge function from ENTRY_YOLO_merge.py (make sure it is defined there)
import ENTRY_YOLO_merge as merge_module

def run_pipeline(match_id: str, device: str = "cpu"):
    # --- 1. Download Videos from Backblaze ---
    print("Downloading left video...")
    result_left = download_file(match_id, "left_video.mp4", local_path="left_video.mp4")
    if "error" in result_left:
        print(f"Error downloading left video: {result_left['error']}")
        return

    print("Downloading right video...")
    result_right = download_file(match_id, "right_video.mp4", local_path="right_video.mp4")
    if "error" in result_right:
        print(f"Error downloading right video: {result_right['error']}")
        return

    # --- 2. Run YOLO detections ---
    # These functions are assumed to read local files "left_video.mp4" and "right_video.mp4"
    # and produce "left5shifted.json" and "right5.json" respectively in the current directory.
    print("Running YOLO detection on left video...")
    save_yolo_left(device=device)
    print("Running YOLO detection on right video...")
    save_yolo_right(device=device)

    # --- 3. Merge the outputs ---
    # This function is expected to combine the two detection JSON files (along with any other necessary files)
    # and produce the final merged outputs: right_intersections.json, right_non_intersections.json,
    # left_intersections.json, and left_non_intersections.json in the current directory.
    print("Running merge step...")
    merge_module.run_merge()

    # --- 4. Upload Final JSONs Back to Backblaze ---
    final_jsons = [
        "right_intersections.json",
        "right_non_intersections.json",
        "left_intersections.json",
        "left_non_intersections.json"
    ]
    for fname in final_jsons:
        print(f"Uploading {fname}...")
        upload_result = upload_json(match_id, fname, fname)
        if "error" in upload_result:
            print(f"Error uploading {fname}: {upload_result['error']}")
        else:
            print(f"{fname} uploaded to: {upload_result.get('final_file_url', 'unknown URL')}")

    print("✅ Pipeline complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True, help="Match folder ID in Backblaze B2")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device for YOLO inference")
    args = parser.parse_args()

    run_pipeline(args.match_id, args.device)
