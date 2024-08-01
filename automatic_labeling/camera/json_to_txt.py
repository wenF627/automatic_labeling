import json
import os
from pathlib import Path

def calculate_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return centroid_x, centroid_y

def calculate_width_height(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width, height

def normalize_coordinates(points, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in points]

def json_to_txt(json_folder, txt_folder):
    os.makedirs(txt_folder, exist_ok=True)

    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            img_width = data['imageWidth']
            img_height = data['imageHeight']
            shapes = data['shapes']
            image_path = data['imagePath']
            image_name = Path(image_path).stem
            txt_path = os.path.join(txt_folder, image_name + '.txt')

            with open(txt_path, 'w') as txt_file:
                for shape in shapes:
                    if 'points' in shape:
                        class_id = 0
                        points = shape['points']
                        normalized_points = normalize_coordinates(points, img_width, img_height)
                        center_x, center_y = calculate_centroid(normalized_points)
                        width, height = calculate_width_height(normalized_points)
                        points_str = ' '.join(f"{x:.6f} {y:.6f}" for x, y in normalized_points)
                        txt_file.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {points_str}\n")

            print(f"Converted {json_file} to {txt_path}")

if __name__ == "__main__":
    json_folder = input("Enter the path to the JSON files folder: ").strip()
    txt_folder = os.path.join(json_folder + '_txt_file')
    json_to_txt(json_folder, txt_folder)
