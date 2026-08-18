from ultralytics import YOLO
from sort.sort import Sort
import numpy as np
from datetime import datetime
from playsound import playsound
import threading
import csv
import os
from database import search_plate
from utils import clean_plate
import easyocr
import cv2
import time

# Load model
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/license_plate_detector.pt")
saved_plates = set()
# OCR Reader
reader = easyocr.Reader(['en'], gpu=False)
# Open video
cap = cv2.VideoCapture("videos/sample.mp4")

# COCO vehicle classes
vehicle_classes = [2, 3, 5, 7]

log_file = "logs/detections.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date",
            "Time",
            "Plate",
            "Owner",
            "Vehicle",
            "Status"
        ])

detected_plates = set()
alerted_plates = set()
total_detected = 0

os.makedirs("logs", exist_ok=True)


if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Plate",
            "Owner",
            "Vehicle",
            "Status",
            "Date",
            "Time",
            "Image"
        ])

def play_alert():
    playsound("sounds/alert.wav")        
prev_time = 0
fps = 0
tracker = Sort()

while True:

    ret, frame = cap.read()

    if not ret:
        break
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time 

    # ---------------- Vehicle Detection ----------------
    vehicle_results = vehicle_model(frame)
    detections = []

    for result in vehicle_results:

        for box in result.boxes:

            cls = int(box.cls[0])

            if cls in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_id = None

                
                detections.append([
                x1,
                y1,
                x2,
                y2,
                float(box.conf[0])
               ])

    tracks = tracker.update(np.array(detections))

    for track in tracks:

        x1, y1, x2, y2, track_id = track

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        track_id = int(track_id)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Vehicle ID : {track_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )


    # ---------------- Plate Detection ----------------
    plate_results = plate_model(frame)

    for result in plate_results:

        for box in result.boxes:

            px1, py1, px2, py2 = map(int, box.xyxy[0])
            vehicle_id = None

            for track in tracks:

                tx1, ty1, tx2, ty2, track_id = track

                tx1 = int(tx1)
                ty1 = int(ty1)
                tx2 = int(tx2)
                ty2 = int(ty2)

                # Check if the plate lies inside the vehicle box
                if (
                    px1 >= tx1 and
                    py1 >= ty1 and
                    px2 <= tx2 and
                    py2 <= ty2
                ):
                    vehicle_id = int(track_id)
                    break

            # Draw plate box
            cv2.rectangle(frame,
                        (px1, py1),
                        (px2, py2),
                        (0,0,255),
                        2)


            cv2.putText(frame,
                        "Plate",
                        (px1, py1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2)

            # Crop the plate
            plate_crop = frame[py1:py2, px1:px2]

            cv2.imshow("Plate Crop", plate_crop)

            cv2.imwrite("output/plate.jpg", plate_crop)

            # OCR
            # Convert to grayscale
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # Enlarge image
            gray = cv2.resize(gray, None, fx=2, fy=2)

            # Remove noise
            gray = cv2.GaussianBlur(gray, (3,3), 0)

            # Binary threshold
            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            cv2.imshow("Processed Plate", thresh)
            cv2.imwrite("output/processed_plate.jpg", thresh)
            ocr_result = reader.readtext(thresh)


            for detection in ocr_result:

                text = detection[1]
                confidence = detection[2]

                # Clean OCR text
                plate = clean_plate(text)

                cv2.putText(
                    frame,
                    plate,
                    (px1, py1-35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )

                # Search in database
                record = search_plate(plate)

                print("=" * 50)
                print("Detected Plate :", plate)
                print("Confidence     :", round(confidence, 2))
                 
                if record is not None:

                    owner = record["owner"]
                    vehicle_name = record["vehicle"]
                    status = record["status"]
                    folder = "evidence/clear"

                    cv2.putText(
                        frame,
                        f"Owner : {owner}",
                        (px1, py2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Vehicle : {vehicle_name}",
                        (px1, py2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2
                    )


                    if status.upper() == "WANTED":
                        folder = "evidence/wanted"

                    os.makedirs(folder, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{folder}/{plate}_{timestamp}.jpg"

                    if plate not in saved_plates:
                        cv2.imwrite(filename, frame)
                        saved_plates.add(plate)
                        today = datetime.now()

                        with open(log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                plate,
                                owner,
                                vehicle_name,
                                status,
                                today.strftime("%d-%m-%Y"),
                                today.strftime("%H:%M:%S"),
                                filename
                            ])

                    if plate not in detected_plates:
                        
                        detected_plates.add(plate)
                        total_detected += 1

                        now = datetime.now()

                        date = now.strftime("%Y-%m-%d")
                        log_time = now.strftime("%H:%M:%S")

                        with open(log_file, "a", newline="") as f:

                            writer = csv.writer(f)

                            writer.writerow([
                                date,
                                log_time,
                                plate,
                                owner,
                                vehicle_name,
                                status
                            ])

                        print(f"Logged -> {plate}")
                    

                    # Green = Clear
                    color = (0,255,0)

                    if status.upper() != "CLEAR":
                        color = (0,0,255)

                    # Background panel
                    cv2.rectangle(
                        frame,
                        (px1, py1-95),
                        (px1+260, py1),
                        (40,40,40),
                        -1
                    )

                    # Plate
                    cv2.putText(
                        frame,
                        f"Plate : {plate}",
                        (px1+5, py1-70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # Owner
                    cv2.putText(
                        frame,
                        f"Owner : {owner}",
                        (px1+5, py1-45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2
                    )

                    # Vehicle
                    cv2.putText(
                        frame,
                        f"Vehicle : {vehicle_name}",
                        (px1+5, py1-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2
                    )

                                        # Status
                    if status.upper() == "CLEAR":

                        cv2.putText(
                            frame,
                            "STATUS : CLEAR",
                            (px1+5, py1+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2
                        )

                    else:

                        if plate not in alerted_plates:
                            threading.Thread(target=play_alert, daemon=True).start()
                            alerted_plates.add(plate)

                        cv2.putText(
                            frame,
                            "STATUS : WANTED",
                            (px1+5, py1+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2
                        )

                        # 🚨 Alert Banner
                        cv2.rectangle(
                            frame,
                            (0, 0),
                            (frame.shape[1], 70),
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                            frame,
                            "ALERT! WANTED VEHICLE DETECTED",
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            3
                        )

                else:

                    cv2.putText(
                        frame,
                        "NOT REGISTERED",
                        (px1, py1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,255),
                        2
                    )
                print("=" * 50)



# ------------------------------------
# Dashboard
# ------------------------------------

    cv2.rectangle(frame, (10, 10), (330, 210), (35, 35, 35), -1)


    cv2.putText(
        frame,
        "ALPR MONITOR",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles Detected : {total_detected}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Current Plate : {plate if 'plate' in locals() else '--'}",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        (20,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0,255,0),
        2
    )

    cv2.putText(
    frame,
    f"FPS : {int(fps)}",
    (20, 175),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 255),
    2
    )

    # Show final frame
    cv2.imshow("ALPR System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
