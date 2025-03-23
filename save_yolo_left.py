import os
import json
from ultralytics import YOLO

def save_yolo_left():
    """
    Perform object detection on the left-side video using YOLO
    and save the output to left5shifted.json.
    """
    # Use the current working directory as the base
    base_dir = os.getcwd()

    # Define file names in the current directory (adjust the names as needed)
    video_filename = 'left_video.mp4'   # Your left video file
    model_filename = 'model.pt'           # Your YOLO model weights file (same as for right)
    output_filename = 'left5shifted.json' # Output JSON file for left detections

    # Build full paths (all in the current directory)
    video_path = os.path.join(base_dir, video_filename)
    model_path = os.path.join(base_dir, model_filename)
    output_json = os.path.join(base_dir, output_filename)

    # Initialize YOLO model
    model = YOLO(model_path)

    # Choose device option: 0 for GPU if available, else "cpu"
    device_option = 0  # Change to "cpu" if needed

    # Run prediction (streaming mode)
    results = model.predict(video_path, verbose=True, save=False, stream=True,
                            save_txt=False, save_conf=False, device=device_option)

    detection_data = []
    for frame_idx, result in enumerate(results):
        frame_data = {"frame_index": frame_idx, "objects": []}
        for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            bbox = box.tolist()  # [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            frame_data["objects"].append({
                "class_id": int(cls_id.item()),
                "confidence": float(conf.item()),
                "bbox": list(map(float, bbox)),
                "center": [float(center_x), float(center_y)]
            })
        detection_data.append(frame_data)

    # Save the detection data to the output JSON file
    with open(output_json, 'w') as json_file:
        json.dump(detection_data, json_file, indent=4)
    print(f"YOLO detection data saved to {output_json}")

if __name__ == "__main__":
    save_yolo_left()
