import cv2
import json
import os
from pathlib import Path
import numpy as np

def display_annotations(image, annotations):
    display_image = image.copy()
    for idx, det in enumerate(annotations):
        if 'points' in det:
            points = det['points']
            cv2.polylines(display_image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
            for point in points:
                cv2.circle(display_image, tuple(point), 5, (0, 0, 255), -1)
            centroid = np.mean(points, axis=0).astype(int)
            label = det['class']
            conf = det['confidence']
            cv2.putText(display_image, f'{label} {conf:.2f}', tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(display_image, f'{idx}', tuple(centroid + np.array([0, 20])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display index
        elif 'box' in det:
            x_min, y_min, x_max, y_max = det['box']
            label = det['class']
            conf = det['confidence']
            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_image, f'{label} {conf:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(display_image, f'{idx}', (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display index
    return display_image

def select_polygon(image, display_image):
    points = []
    img = display_image
    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow("Select Polygon", img)

    cv2.namedWindow("Select Polygon")
    cv2.setMouseCallback("Select Polygon", draw_polygon)

    
    cv2.imshow("Select Polygon", img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to confirm the polygon
            break
        elif key == ord("s"):
            img = display_image
            cv2.imshow("Select Polygon", img)
        elif key == ord("n"):
            img = image
            cv2.imshow("Select Polygon", img)
        

    cv2.destroyWindow("Select Polygon")
    return points


def manual_labeling(image, initial_detections):
    annotations = initial_detections.copy()  # Start with the initial detections
    while True:
        # Display current annotations with indexes
        display_image = display_annotations(image, annotations)
        #display_image_r = cv2.resize(display_image, (960, 540))
        #image_r = cv2.resize(image, (960,540))
        cv2.imshow("Manual Labeling", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):  # Add a new bounding polygon
            print("Adding new bounding polygon...")
            image_copy = image.copy()
            display_image_copy = display_image.copy()
            points = select_polygon(image_copy, display_image_copy)
            if len(points) < 3:
                print("A polygon must have at least 3 points.")
                continue

            label = input("Enter label for the bounding polygon: ")
            conf = float(input("Enter confidence for the bounding polygon (0 to 1): "))

            annotations.append({
                'points': points,
                'confidence': conf,
                'class': label
            })

        elif key == ord("d"):  # Delete an existing bounding polygon
            print("Deleting existing bounding polygon...")
            try:
                idx = int(input(f"Enter index of bounding polygon to delete (0 to {len(annotations) - 1}): "))
                if 0 <= idx < len(annotations):
                    annotations.pop(idx)
                else:
                    print(f"Invalid index: {idx}")
            except ValueError:
                print("Invalid input, please enter a valid index number.")

# ??? how to drag to modify, needs to be fixed????
        elif key == ord("m"):  # Modify an existing bounding polygon
            print("Modifying existing bounding polygon...")
            try:
                idx = int(input(f"Enter index of bounding polygon to modify (0 to {len(annotations) - 1}): "))
                if 0 <= idx < len(annotations):
                    original_points = annotations[idx].get('points')
                    if original_points:
                        image_copy = image.copy()
                        for point in original_points:
                            cv2.circle(image_copy, tuple(point), 5, (0, 0, 255), -1)
                        points = select_polygon(image_copy)
                        if len(points) < 3:
                            print("A polygon must have at least 3 points.")
                            continue
                        annotations[idx]['points'] = points
                        label = input("Enter new label for the bounding polygon: ")
                        conf = float(input("Enter new confidence for the bounding polygon (0 to 1): "))
                        annotations[idx]['class'] = label
                        annotations[idx]['confidence'] = conf
                    else:
                        print("This annotation is not a polygon.")
                else:
                    print(f"Invalid index: {idx}")
            except ValueError:
                print("Invalid input, please enter a valid index number.")

        elif key == ord("q"):  # Quit the manual labeling
            print("Quitting manual labeling...")
            break

    cv2.destroyAllWindows()
    return annotations, image

def main():
    base_json_folder = 'C:/data/json_file/'
    base_labeled_image_save_folder = 'C:/data/label_file/'

    # Iterate through timestamp folders
    for timestamp_folder in os.listdir(base_json_folder):
        json_folder = os.path.join(base_json_folder, timestamp_folder)
        labeled_image_save_folder = os.path.join(base_labeled_image_save_folder, timestamp_folder)

        if not os.path.isdir(json_folder):
            continue

        # Iterate through camera folders inside the timestamp folder
        for camera_folder in os.listdir(json_folder):
            camera_json_folder = os.path.join(json_folder, camera_folder)
            camera_labeled_image_folder = os.path.join(labeled_image_save_folder, camera_folder)

            if not os.path.isdir(camera_json_folder):
                continue

            json_files = [f for f in os.listdir(camera_json_folder) if f.endswith('_detections.json')]

            for json_file in json_files:
                json_path = os.path.join(camera_json_folder, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)

                image_path = data['image_path']
                initial_detections = data['detections']

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to load image at {image_path}")
                    continue

                # Perform manual labeling
                annotations, labeled_image = manual_labeling(image, initial_detections)

                # Update the original JSON file with the modified annotations
                data['detections'] = annotations
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)

                # Ensure labeled image save folder exists
                os.makedirs(camera_labeled_image_folder, exist_ok=True)

                # Save labeled image
                labeled_image_path = os.path.join(camera_labeled_image_folder, Path(image_path).stem + '_manual_labeled.jpg')
                cv2.imwrite(labeled_image_path, labeled_image)

                print(f"Detections saved to {json_path}")
                print(f"Labeled image saved to {labeled_image_path}")

if __name__ == "__main__":
    main()
