import cv2
import os
import threading
import logging
import time
import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 摄像头的 RTSP URLs
urls = [
    "rtsp://admin:wuhan027@172.16.40.23:554/Streaming/Channels/101",
    "rtsp://admin:wuhan027@172.16.40.24:554/Streaming/Channels/101",
    "rtsp://admin:wuhan027@172.16.40.25:554/Streaming/Channels/101",
]

# 主输出目录
base_dir = r'C:\automatic_labeling\automatic_labeling\data'

# Create timestamped directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp_dir = os.path.join(base_dir, timestamp)

# Create camera-specific folders
camera_folders = {
    "172.16.40.23:554": "camera_23",
    "172.16.40.24:554": "camera_24",
    "172.16.40.25:554": "camera_25"
}

# Create directories
for camera_ip, camera_folder in camera_folders.items():
    camera_dir = os.path.join(timestamp_dir, camera_folder)
    os.makedirs(os.path.join(camera_dir, 'json_file'), exist_ok=True)
    os.makedirs(os.path.join(camera_dir, 'pic_file'), exist_ok=True)
    os.makedirs(os.path.join(camera_dir, 'label_file'), exist_ok=True)

def load_yolo_model(model_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def perform_inference(model, image):
    with torch.amp.autocast('cuda'):
        results = model(image)
    detections = []
    for *box, conf, cls in results.xyxy[0]:
        x_min, y_min, x_max, y_max = map(int, box)
        conf = float(conf)
        cls = int(cls)
        label = model.names[cls]
        box = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        detections.append({
            'label': label,
            'box': box,
            'confidence': conf
        })
    return detections

def capture_frames(url, initial_frame_num, model):
    camera_ip = url.split("@")[1].split("/")[0]
    camera_folder = camera_folders[camera_ip]
    pic_output_dir = os.path.join(timestamp_dir, camera_folder, 'pic_file')
    json_output_dir = os.path.join(timestamp_dir, camera_folder, 'json_file')
    labeled_image_output_dir = os.path.join(timestamp_dir, camera_folder, 'label_file')

    cap = cv2.VideoCapture(url)
    frame_num = initial_frame_num
    retry_attempts = 5
    retry_delay = 5

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Failed to read frame from {url}")
                retry_attempts -= 1
                if retry_attempts == 0:
                    logging.error(f"Max retry attempts reached for {url}. Exiting...")
                    break
                logging.info(f"Retrying in {retry_delay} seconds for {url}...")
                time.sleep(retry_delay)
                continue

            frame_num += 1

            # 每100帧保存一张图片
            if frame_num % 100 == 0:
                filename = f'frame_{frame_num // 100}_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.jpg'
                filepath = os.path.join(pic_output_dir, filename)
                cv2.imwrite(filepath, frame)
                logging.info(f'Saved {filename} from {url}')

                # Perform initial inference
                initial_detections = perform_inference(model, frame)

                # Save initial inference results to JSON
                output_json = {
                    'version': "5.5.0",
                    'flags': {},
                    'shapes': [{
                        'label': det['label'],
                        'points': det['box'],
                        'group_id': None,
                        'description': "",
                        'shape_type': "polygon",
                        'flags': {}
                    } for det in initial_detections],
                    'imagePath': filename,
                    'imageData': None,
                    'imageHeight': frame.shape[0],
                    'imageWidth': frame.shape[1]
                }
                json_output_path = os.path.join(json_output_dir, Path(filename).stem + '_detections.json')
                with open(json_output_path, 'w') as f:
                    json.dump(output_json, f, indent=4)

                # Save labeled image (without manual adjustments)
                labeled_image_path = os.path.join(labeled_image_output_dir, Path(filename).stem + '_labeled.jpg')
                for det in initial_detections:
                    points = det['box']
                    label = det['label']
                    conf = det['confidence']
                    cv2.polylines(frame, [np.array(points, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.imwrite(labeled_image_path, frame)

                logging.info(f"Detections saved to {json_output_path}")
                logging.info(f"Labeled image saved to {labeled_image_path}")

    except Exception as e:
        logging.error(f"An error occurred with {url}: {e}")

    finally:
        cap.release()
        logging.info(f"Released video capture for {url}")

# Load the YOLOv5 model once
model_path = r"C:\automatic_labeling\automatic_labeling\model\best_OEM.pt"  # Update path to your model
model = load_yolo_model(model_path)

# 创建并启动线程
threads = []
initial_frame_num = 0

for url in urls:
    thread = threading.Thread(target=capture_frames, args=(url, initial_frame_num, model))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

logging.info("Finished capturing from all URLs")
