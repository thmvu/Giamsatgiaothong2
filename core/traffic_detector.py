"""
Traffic Light Detector
=======================
Sử dụng model phathienden.pt để detect đèn giao thông.
Classes: {0: 'green', 1: 'off', 2: 'red', 3: 'yellow'}
"""

from ultralytics import YOLO
from typing import List, Tuple

# Class IDs
GREEN = 0
OFF = 1
RED = 2
YELLOW = 3
LIGHT_IDS = [GREEN, RED, YELLOW]  # Bỏ OFF


class TrafficDetector:
    """
    Expert 2: Phát hiện đèn giao thông.
    Model nhỏ, chuyên biệt → nhanh, chính xác.
    """

    def __init__(self, model_path: str = "phathienden.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, conf: float = 0.3):
        """
        Detect trạng thái đèn + trả về boxes để vẽ.

        Returns:
            state: 'red' | 'green' | 'yellow' | 'unknown'
            light_boxes: list[{state, conf, bbox}]
        """
        results = self.model.predict(
            frame, classes=LIGHT_IDS, conf=conf, verbose=False
        )

        state = "unknown"
        best_conf = 0
        light_boxes = []

        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i])
                c = float(res.boxes.conf[i])
                x1, y1, x2, y2 = res.boxes.xyxy[i].int().cpu().tolist()

                if cls_id == RED:
                    s = "red"
                elif cls_id == GREEN:
                    s = "green"
                elif cls_id == YELLOW:
                    s = "yellow"
                else:
                    continue

                light_boxes.append({"state": s, "conf": c, "bbox": (x1, y1, x2, y2)})

                if c > best_conf:
                    best_conf = c
                    state = s

        return state, light_boxes
