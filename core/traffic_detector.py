"""
Traffic Light Detector
====================================
Sử dụng model phathienden.pt để detect đèn giao thông (chỉ đèn).
Vạch kẻ đường được xử lý riêng bởi model batvachkeduongseg.pt trong giai đoạn Calibration.
"""

from ultralytics import YOLO
from typing import List, Tuple, Optional


# Class IDs cho model phathienden.pt (chỉ đèn giao thông)
GREEN_LIGHT = 0
RED_LIGHT = 1
YELLOW_LIGHT = 2

LIGHT_IDS = [GREEN_LIGHT, RED_LIGHT, YELLOW_LIGHT]


class TrafficDetector:
    """
    Phát hiện đèn giao thông bằng model phathienden.pt.
    Chỉ detect đèn (đỏ/xanh/vàng), không detect vạch dừng.
    Vạch dừng được xử lý bởi batvachkeduongseg.pt trong Calibration Phase.
    """

    def __init__(self, model_path: str = "phathienden.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, conf: float = 0.3):
        """
        Detect trạng thái đèn giao thông.

        Returns:
            state: 'red' | 'green' | 'yellow' | 'unknown'
            light_boxes: list[{state, conf, bbox}]  — các đèn phát hiện được
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

                if cls_id == RED_LIGHT:
                    s = "red"
                elif cls_id == GREEN_LIGHT:
                    s = "green"
                elif cls_id == YELLOW_LIGHT:
                    s = "yellow"
                else:
                    continue

                light_boxes.append({"state": s, "conf": c, "bbox": (x1, y1, x2, y2)})
                if c > best_conf:
                    best_conf = c
                    state = s

        return state, light_boxes
