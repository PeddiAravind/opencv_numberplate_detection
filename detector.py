from ultralytics import YOLO


class VehicleDetector:

    def __init__(self, model_path="yolov8n.pt"):

        self.model = YOLO(model_path)

        # COCO vehicle classes
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                if cls in self.vehicle_classes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    confidence = float(box.conf[0])

                    detections.append([
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        cls
                    ])

        return detections