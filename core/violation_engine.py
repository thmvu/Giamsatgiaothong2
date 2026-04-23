"""
Violation Detection Engine
===========================
Bộ não chính — kết hợp tất cả Experts để phát hiện vi phạm.

Logic vượt đèn đỏ:
  - Stop line được model phathiendenvadung.pt tự phát hiện
  - Đèn XANH → phương tiện được phép đi qua
  - Đèn ĐỎ  → nếu TÂM phương tiện vượt qua stop_line → VI PHẠM
  - Chỉ đánh dấu vi phạm 1 lần duy nhất mỗi xe (sổ đen)
"""

import cv2
import os
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
from .vehicle_detector import DetectedObject


@dataclass
class Violation:
    """Một bản ghi vi phạm."""
    time_sec: float
    frame_number: int
    violation_type: str       # "Không đội mũ bảo hiểm" | "Vượt đèn đỏ"
    vehicle_type: str
    track_id: int
    plate: str = ""
    evidence_path: str = ""


class ViolationState:
    """Quản lý trạng thái vi phạm xuyên suốt video."""

    def __init__(self):
        self.helmet_violated_ids: Set[int] = set()
        self.redlight_violated_ids: Set[int] = set()
        self.prev_bbox_cache: Dict[int, list] = {}  # track_id → bbox frame trước
        self.violations: List[Violation] = []

    def is_helmet_violated(self, track_id: int) -> bool:
        return track_id in self.helmet_violated_ids

    def is_redlight_violated(self, track_id: int) -> bool:
        return track_id in self.redlight_violated_ids

    def add_helmet_violation(self, obj: DetectedObject, frame_number: int,
                              fps: float, evidence_dir: str, frame=None):
        """Ghi nhận vi phạm mũ bảo hiểm."""
        self.helmet_violated_ids.add(obj.track_id)

        ev_path = ""
        if frame is not None and evidence_dir:
            ev_path = os.path.join(evidence_dir,
                f"helmet_ID{obj.track_id}_f{frame_number}.jpg")
            cv2.imwrite(ev_path, frame)

        self.violations.append(Violation(
            time_sec=round(frame_number / max(fps, 1), 2),
            frame_number=frame_number,
            violation_type="Không đội mũ bảo hiểm",
            vehicle_type=obj.vn_name,
            track_id=obj.track_id,
            evidence_path=ev_path
        ))

    def check_redlight_crossing(self, obj: DetectedObject,
                                 stop_line_pts: Optional[Tuple],
                                 traffic_state: str, frame_number: int,
                                 fps: float, evidence_dir: str, frame=None) -> bool:
        """
        Kiểm tra xe có vượt đèn đỏ không.

        Logic:
        - stop_line_pts = ((x1,y1), (x2,y2)) từ model detect
        - So sánh cross product giữa tâm xe frame trước và hiện tại
        - Sign change + đèn ĐỎ → VI PHẠM

        Returns: True nếu vi phạm
        """
        if obj.track_id is None or stop_line_pts is None:
            return False

        # Đã nằm trong sổ đen
        if obj.track_id in self.redlight_violated_ids:
            return True

        curr_bbox = [obj.x1, obj.y1, obj.x2, obj.y2]
        prev_bbox = self.prev_bbox_cache.get(obj.track_id)

        # Lưu bbox cho frame sau
        self.prev_bbox_cache[obj.track_id] = curr_bbox

        # Đèn XANH hoặc VÀNG → cho phép đi, không xét
        if traffic_state != "red":
            return False

        # Đèn ĐỎ → kiểm tra tâm xe có vượt qua stop_line không
        if prev_bbox is not None:
            from utils.violation import has_crossed_line
            crossed = has_crossed_line(prev_bbox, curr_bbox, stop_line_pts)

            if crossed:
                self.redlight_violated_ids.add(obj.track_id)

                ev_path = ""
                if frame is not None and evidence_dir:
                    ev_path = os.path.join(evidence_dir,
                        f"redlight_ID{obj.track_id}_f{frame_number}.jpg")
                    cv2.imwrite(ev_path, frame)

                self.violations.append(Violation(
                    time_sec=round(frame_number / max(fps, 1), 2),
                    frame_number=frame_number,
                    violation_type="Vượt đèn đỏ",
                    vehicle_type=obj.vn_name,
                    track_id=obj.track_id,
                    evidence_path=ev_path
                ))
                return True

        return False

    @property
    def total_violations(self):
        return len(self.helmet_violated_ids) + len(self.redlight_violated_ids)
