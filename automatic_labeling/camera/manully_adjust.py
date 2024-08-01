import cv2
import json
import os
from pathlib import Path
import numpy as np

def display_annotations(image, annotations, zoom=1.0, alpha=0.4):
    point_radius = max(1, int(2 / zoom))
    line_thickness = max(1, int(2 / zoom))

    display_image = image.copy()
    overlay = image.copy()
    for idx, det in enumerate(annotations):
        if 'points' in det:
            points = det['points']
            points_array = np.array(points)
            cv2.fillPoly(overlay, [points_array], color=(0, 255, 0))
            cv2.polylines(display_image, [points_array], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
            for point in points:
                cv2.circle(display_image, tuple(point), point_radius, (0, 0, 255), -1)
            centroid = np.mean(points, axis=0).astype(int)
            label = det['label']
            cv2.putText(display_image, f'{label}', tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(display_image, f'{idx}', tuple(centroid + np.array([0, 20])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display index
        elif 'box' in det:
            x_min, y_min, x_max, y_max = det['box']
            label = det['label']
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), line_thickness)
            cv2.putText(display_image, f'{label}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(display_image, f'{idx}', (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display index

    cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)
    return display_image

def select_polygon(image, display_image, percentage):
    points = []
    img = display_image.copy()
    point_radius = min(1, int(2 / percentage))
    line_thickness = min(1, int(2 / percentage))
    dragging = False

    def draw_polygon(event, x, y, flags, param):
        nonlocal dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = False
            param["start_pos"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            if not dragging:
                print((x, y))
                points.append((x, y))
                cv2.circle(img, (x, y), point_radius, (0, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(img, points[-2], points[-1], (0, 255, 0), line_thickness)
                cv2.imshow("Select Polygon", img)
            dragging = False

    cv2.namedWindow("Select Polygon")
    cv2.setMouseCallback("Select Polygon", draw_polygon, param={"start_pos": None})

    while True:
        cv2.imshow("Select Polygon", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to confirm the polygon
            break
        elif key == ord("s"):
            img = display_image.copy()
            for i, point in enumerate(points):
                cv2.circle(img, point, point_radius, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(img, points[i - 1], points[i], (0, 255, 0), line_thickness)
            cv2.imshow("Select Polygon", img)
        elif key == ord("n"):
            img = image.copy()
            for i, point in enumerate(points):
                cv2.circle(img, point, point_radius, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(img, points[i - 1], points[i], (0, 255, 0), line_thickness)
            cv2.imshow("Select Polygon", img)
        elif key == 27:  # Press 'ESC' to exit the program
            print("Exiting program...")
            cv2.destroyAllWindows()
            os._exit(0)
        elif key == 8:  # Press 'Backspace' to remove the last point
            if points:
                points.pop()
                img = display_image.copy()
                for i, point in enumerate(points):
                    cv2.circle(img, point, point_radius, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(img, points[i - 1], points[i], (0, 255, 0), line_thickness)
                cv2.imshow("Select Polygon", img)

    cv2.destroyWindow("Select Polygon")

    # Convert points back to the original size x y pos
    for i, x in enumerate(points):
        points[i] = (int(x[0] / percentage), int(x[1] / percentage))

    return points

def save_annotations(json_path, annotations):
    with open(json_path, 'r') as f:
        data = json.load(f)

    data['shapes'] = annotations

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def manual_labeling(image, json_path, initial_detections, labeled_image_save_folder, image_name):
    annotations = initial_detections.copy()  # Start with the initial detections

    def nothing(x):
        pass

    width = image.shape[1]  # width of image
    height = image.shape[0]  # height of image
    cv2.namedWindow("Manual Labeling")  # create a window for display
    cv2.moveWindow("Manual Labeling", 0, 0)  # move window to start from top left angle
    cv2.createTrackbar("Window_size", "Manual Labeling", 65, 100,
                       nothing)  # create a Trackbar for controlling window size initial percentage is 65%

    while True:
        p = cv2.getTrackbarPos("Window_size", "Manual Labeling") / 100
        cv2.resizeWindow("Manual Labeling", int(p * width), int(p * height))
        display_image = display_annotations(image, annotations, zoom=p)  # Display current annotations with indexes
        display_image_r = cv2.resize(display_image, (int(p * width), int(p * height)))  # resize our image for display
        image_r = cv2.resize(image, (int(p * width), int(p * height)))  # resize our unannotated image for display

        cv2.imshow("Manual Labeling", display_image_r)
        cv2.setMouseCallback("Manual Labeling",
                             lambda *args: None)  # Disable mouse callback when not in polygon selection mode
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):  # Add a new bounding polygon
            print("Adding new bounding polygon...")
            image_copy = image_r.copy()
            display_image_copy = display_image_r.copy()
            points = select_polygon(image_copy, display_image_copy, p)
            if len(points) < 3:
                print("A polygon must have at least 3 points.")
                continue

            label = input("Enter label for the bounding polygon: ")

            annotations.append({
                'label': label,
                'points': points,
                'group_id': None,
                'description': "",
                'shape_type': "polygon",
                'flags': {}
            })
            save_annotations(json_path, annotations)  # Automatically save changes

        elif key == ord("d"):  # Delete an existing bounding polygon
            print("Deleting existing bounding polygon...")
            try:
                idx = int(input(f"Enter index of bounding polygon to delete (0 to {len(annotations) - 1}): "))
                if 0 <= idx < len(annotations):
                    annotations.pop(idx)
                    save_annotations(json_path, annotations)  # Automatically save changes
                else:
                    print(f"Invalid index: {idx}")
            except ValueError:
                print("Invalid input, please enter a valid index number.")

        elif key == ord("s"):  # Save the current labeled image
            print("Saving labeled image...")
            labeled_image_path = os.path.join(labeled_image_save_folder, f"{image_name}_labeled.jpg")
            cv2.imwrite(labeled_image_path, display_image)
            print(f"Labeled image saved to {labeled_image_path}")
            break

        elif key == ord("q"):  # Quit the manual labeling
            print("Quitting manual labeling...")
            break
        elif key == 27:  # Press 'ESC' to exit the program
            print("Exiting program...")
            cv2.destroyAllWindows()
            os._exit(0)

    cv2.destroyAllWindows()
    return annotations, image

def main():
    image_dir = input("Enter the path to the image directory: ").strip()
    json_dir = input("Enter the path to the JSON directory (or press Enter to skip): ").strip()
    labeled_image_save_folder = os.path.join(image_dir, 'labeled_images')

    if not os.path.isdir(image_dir):
        print(f"The image directory path is invalid or does not exist: {image_dir}")
        return

    if json_dir and not os.path.isdir(json_dir):
        print(f"The JSON directory path is invalid or does not exist: {json_dir}")
        return

    json_files = {}
    if json_dir:
        json_files = {Path(f).stem.replace('_detections', ''): os.path.join(json_dir, f) for f in os.listdir(json_dir)
                      if f.lower().endswith('_detections.json')}

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            stem = Path(image_file).stem
            json_path = json_files.get(stem)
            initial_detections = []

            if json_path:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                initial_detections = data.get('shapes', [])
            else:
                print(f"No JSON file found for image: {image_file}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image at {image_path}")
                continue

            # Perform manual labeling
            annotations, labeled_image = manual_labeling(image, json_path, initial_detections, labeled_image_save_folder, stem)

            # Ensure labeled image save folder exists
            os.makedirs(labeled_image_save_folder, exist_ok=True)

            print(f"Detections saved to {json_path}")

if __name__ == "__main__":
    main()
