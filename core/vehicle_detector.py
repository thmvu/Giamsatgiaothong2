"""
Vehicle Detector Module
=======================
Sử dụng YOLO11m gốc (COCO) để detect + track phương tiện.
COCO classes: bicycle(1), car(2), motorcycle(3), bus(5), truck(7), traffic_light(9)
"""

from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional

# COCO class IDs
COCO_CLASSES = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic_light": 9,
}

VEHICLE_IDS = {1, 2, 3, 5, 7}
MOTORBIKE_IDS = {3}
DETECT_IDS = list(VEHICLE_IDS) + [9]  # Xe + đèn

VN_NAMES = {
    1: "Xe đạp", 2: "Ô tô", 3: "Xe máy",
    5: "Xe buýt", 7: "Xe tải", 9: "Đèn giao thông",
}


@dataclass
class DetectedObject:
    """Một object được detect trong frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def center_y(self):
        return (self.y1 + self.y2) // 2

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def is_vehicle(self):
        return self.cls_id in VEHICLE_IDS

    @property
    def is_motorbike(self):
        return self.cls_id in MOTORBIKE_IDS

    @property
    def is_traffic_light(self):
        return self.cls_id == 9

    @property
    def vn_name(self):
        return VN_NAMES.get(self.cls_id, self.class_name)


class VehicleDetector:
    """
    Expert 1: Phát hiện + theo dõi phương tiện bằng YOLO gốc.
    Sử dụng ByteTrack để gán ID liên tục cho mỗi phương tiện.
    """

    def __init__(self, model_path: str = "yolo11m.pt"):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame, conf: float = 0.3) -> List[DetectedObject]:
        """
        Chạy detect + track trên 1 frame.
        Returns list of DetectedObject (cả xe lẫn đèn giao thông).
        """
        results = self.model.track(
            frame,
            classes=DETECT_IDS,
            conf=conf,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        objects = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes.xyxy.int().cpu().tolist()
            clss = res.boxes.cls.int().cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(boxes)

            for i in range(len(boxes)):
                obj = DetectedObject(
                    x1=boxes[i][0], y1=boxes[i][1],
                    x2=boxes[i][2], y2=boxes[i][3],
                    cls_id=clss[i],
                    class_name=self.model.names[clss[i]],
                    confidence=confs[i],
                    track_id=ids[i]
                )
                objects.append(obj)

        return objects
