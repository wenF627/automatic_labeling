import cv2
import os
import threading
import logging
import time
import torch
import json
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
output_base_dir = 'C:/data/pic_file/'  # Update path to your folder
json_save_base_folder = 'C:/data/json_file/'  # Update path to save JSON files
labeled_image_save_base_folder = 'C:/data/label_file/'  # Update path to save labeled images

# 创建带有时间戳的子目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_base_dir = os.path.join(output_base_dir, timestamp)
json_save_folder = os.path.join(json_save_base_folder, timestamp)
labeled_image_save_folder = os.path.join(labeled_image_save_base_folder, timestamp)

# 创建主输出目录（带时间戳）
os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(json_save_folder, exist_ok=True)
os.makedirs(labeled_image_save_folder, exist_ok=True)


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
        label = model.names[cls]  # Get the class name
        detections.append({
            'box': [x_min, y_min, x_max, y_max],
            'confidence': conf,
            'class': label
        })
    return detections


def create_output_dirs(url):
    """Create directories based on the URL inside the main timestamped directory."""
    # 使用URL的部分作为文件夹名称
    url_identifier = url.split("//")[1].replace(":", "_").replace("/", "_")

    pic_output_dir = os.path.join(output_base_dir, url_identifier)
    json_output_dir = os.path.join(json_save_folder, url_identifier)
    labeled_image_output_dir = os.path.join(labeled_image_save_folder, url_identifier)

    os.makedirs(pic_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(labeled_image_output_dir, exist_ok=True)

    return pic_output_dir, json_output_dir, labeled_image_output_dir


def capture_frames(url, initial_frame_num, model):
    pic_output_dir, json_output_dir, labeled_image_output_dir = create_output_dirs(url)
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
                    'image_path': filepath,
                    'detections': initial_detections
                }
                json_output_path = os.path.join(json_output_dir, Path(filename).stem + '_detections.json')
                with open(json_output_path, 'w') as f:
                    json.dump(output_json, f, indent=4)

                # Save labeled image (without manual adjustments)
                labeled_image_path = os.path.join(labeled_image_output_dir, Path(filename).stem + '_labeled.jpg')
                for det in initial_detections:
                    x_min, y_min, x_max, y_max = det['box']
                    label = det['class']
                    conf = det['confidence']
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
                cv2.imwrite(labeled_image_path, frame)

                logging.info(f"Detections saved to {json_output_path}")
                logging.info(f"Labeled image saved to {labeled_image_path}")

    except Exception as e:
        logging.error(f"An error occurred with {url}: {e}")

    finally:
        cap.release()
        logging.info(f"Released video capture for {url}")


# Load the YOLOv5 model once
model_path = 'C:/Users/ychen/Desktop/automatic_labeling/automatic_labeling/model/best_OEM.pt'  # Update path to your model
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
