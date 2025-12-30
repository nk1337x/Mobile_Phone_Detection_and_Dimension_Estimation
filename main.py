import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

KNOWN_DISTANCE = 35.0
KNOWN_PHONE_WIDTH = 7.53
KNOWN_PHONE_HEIGHT = 16.25
BBOX_CORRECTION_FACTOR = 0.92

focal_length = None
calibration_samples = []
measurement_history = []
MAX_HISTORY = 5

def calculate_focal_length(measured_distance, real_width, width_in_pixels):
    return (width_in_pixels * measured_distance) / real_width

def calculate_distance(focal_length, real_width, width_in_pixels):
    if width_in_pixels == 0:
        return 0
    return (real_width * focal_length) / width_in_pixels

def calculate_dimension(distance, pixel_size, focal_length):
    if focal_length == 0:
        return 0
    return (pixel_size * distance) / focal_length

print("=== Phone Dimension Estimator ===")
print("1. Hold phone at 35cm and press 'c' to calibrate (5 times)")
print("2. Press 'q' to quit | 'r' to reset\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)  

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == 67 and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width_px = (x2 - x1) * BBOX_CORRECTION_FACTOR
                height_px = (y2 - y1) * BBOX_CORRECTION_FACTOR
                
                padding_x = int((x2 - x1) * (1 - BBOX_CORRECTION_FACTOR) / 2)
                padding_y = int((y2 - y1) * (1 - BBOX_CORRECTION_FACTOR) / 2)
                x1_adj = x1 + padding_x
                y1_adj = y1 + padding_y
                x2_adj = x2 - padding_x
                y2_adj = y2 - padding_y

                cv2.rectangle(frame, (x1_adj, y1_adj), (x2_adj, y2_adj), (0, 255, 0), 2)

                if focal_length is None:
                    label = f"CALIBRATION MODE - Press 'c'"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Calculate distance using both width and height, then average
                    distance_from_width = calculate_distance(focal_length, KNOWN_PHONE_WIDTH, width_px)
                    distance_from_height = calculate_distance(focal_length, KNOWN_PHONE_HEIGHT, height_px)
                    distance = (distance_from_width + distance_from_height) / 2
                    
                    # Calculate both dimensions dynamically from the averaged distance
                    width_cm = calculate_dimension(distance, width_px, focal_length)
                    height_cm = calculate_dimension(distance, height_px, focal_length)
                    
                    measurement_history.append((width_cm, height_cm, distance))
                    if len(measurement_history) > MAX_HISTORY:
                        measurement_history.pop(0)
                    
                    avg_width = np.mean([m[0] for m in measurement_history])
                    avg_height = np.mean([m[1] for m in measurement_history])
                    avg_distance = np.mean([m[2] for m in measurement_history])

                    label = f"W: {avg_width:.2f}cm H: {avg_height:.2f}cm"
                    cv2.putText(frame, label, (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    label2 = f"Distance: {avg_distance:.1f}cm"
                    cv2.putText(frame, label2, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    status = "Calibrated" if focal_length else "Not Calibrated"
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if calibration_samples:
        cv2.putText(frame, f"Samples: {len(calibration_samples)}/5", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Phone Detection + Dimensions", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 67:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width_px = (x2 - x1) * BBOX_CORRECTION_FACTOR  # Apply correction during calibration
                    
                    fl_sample = calculate_focal_length(KNOWN_DISTANCE, KNOWN_PHONE_WIDTH, width_px)
                    calibration_samples.append(fl_sample)
                    
                    print(f"Sample {len(calibration_samples)}: {fl_sample:.2f}px")
                    
                    if len(calibration_samples) >= 5:
                        focal_length = np.mean(calibration_samples)
                        print(f"Calibration complete! Focal length: {focal_length:.2f}px\n")
                    break
    elif key == ord('r'):
        focal_length = None
        calibration_samples = []
        measurement_history = []
        print("Reset")

cap.release()
cv2.destroyAllWindows()
