"""
Traffic Light & Stop Line Detector
====================================
Sử dụng model phathiendenvadung.pt để detect đèn giao thông VÀ vạch dừng.
Classes: {0: 'green_light', 1: 'red_light', 2: 'stop_line', 3: 'yellow_light'}
"""

from ultralytics import YOLO
from typing import List, Tuple, Optional


# Class IDs cho model phathiendenvadung.pt
GREEN_LIGHT = 0
RED_LIGHT = 1
STOP_LINE = 2
YELLOW_LIGHT = 3

LIGHT_IDS = [GREEN_LIGHT, RED_LIGHT, YELLOW_LIGHT]
ALL_IDS = [GREEN_LIGHT, RED_LIGHT, STOP_LINE, YELLOW_LIGHT]


class TrafficDetector:
    """
    Phát hiện đèn giao thông VÀ vạch dừng bằng model phathiendenvadung.pt.
    Model detect cùng lúc cả đèn và vạch dừng → không cần vẽ thủ công.
    """

    def __init__(self, model_path: str = "phathiendenvadung.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, conf: float = 0.3):
        """
        Detect trạng thái đèn + vạch dừng + trả về boxes để vẽ.

        Returns:
            state: 'red' | 'green' | 'yellow' | 'unknown'
            light_boxes: list[{state, conf, bbox}]  — các đèn phát hiện được
            stop_line_pts: ((x1,y1), (x2,y2)) | None  — 2 điểm trái-phải vạch dừng
            stop_line_bbox: (x1,y1,x2,y2) | None  — bbox gốc của vạch dừng
        """
        results = self.model.predict(
            frame, classes=ALL_IDS, conf=conf, verbose=False
        )

        state = "unknown"
        best_conf = 0
        light_boxes = []
        stop_line_pts = None
        stop_line_bbox = None
        stop_line_conf = 0

        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i])
                c = float(res.boxes.conf[i])
                x1, y1, x2, y2 = res.boxes.xyxy[i].int().cpu().tolist()

                if cls_id == STOP_LINE:
                    # Vạch dừng: lấy bbox có confidence cao nhất
                    if c > stop_line_conf:
                        stop_line_conf = c
                        stop_line_bbox = (x1, y1, x2, y2)
                        # Tính 2 điểm: trái-giữa và phải-giữa của bbox
                        mid_y = (y1 + y2) // 2
                        stop_line_pts = ((x1, mid_y), (x2, mid_y))
                elif cls_id == RED_LIGHT:
                    s = "red"
                    light_boxes.append({"state": s, "conf": c, "bbox": (x1, y1, x2, y2)})
                    if c > best_conf:
                        best_conf = c
                        state = s
                elif cls_id == GREEN_LIGHT:
                    s = "green"
                    light_boxes.append({"state": s, "conf": c, "bbox": (x1, y1, x2, y2)})
                    if c > best_conf:
                        best_conf = c
                        state = s
                elif cls_id == YELLOW_LIGHT:
                    s = "yellow"
                    light_boxes.append({"state": s, "conf": c, "bbox": (x1, y1, x2, y2)})
                    if c > best_conf:
                        best_conf = c
                        state = s

        return state, light_boxes, stop_line_pts, stop_line_bbox
