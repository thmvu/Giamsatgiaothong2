"""
Violation Detection Engine
===========================
Bộ não chính — kết hợp tất cả Experts để phát hiện vi phạm.

Logic vượt đèn đỏ:
  - Stop line kết nối với trạng thái đèn
  - Đèn XANH → phương tiện được phép đi qua
  - Đèn ĐỎ  → nếu TÂM phương tiện đi qua stop_line → VI PHẠM
  - Chỉ đánh dấu vi phạm 1 lần duy nhất mỗi xe (sổ đen)
"""

import cv2
import os
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
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
        self.vehicle_prev_center_y: Dict[int, int] = {}  # track_id → center_y frame trước
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

    def check_redlight_crossing(self, obj: DetectedObject, stop_line_y: int,
                                 traffic_state: str, frame_number: int,
                                 fps: float, evidence_dir: str, frame=None) -> bool:
        """
        Kiểm tra xe có vượt đèn đỏ không.

        Logic:
        - Lấy center_y của vehicle
        - So sánh với center_y ở frame trước
        - Nếu đèn ĐỎ + center_y vượt qua stop_line → VI PHẠM
        - Nếu đèn XANH → cho phép đi qua, không phạt

        Returns: True nếu vi phạm
        """
        if obj.track_id is None or stop_line_y is None:
            return False

        # Đã nằm trong sổ đen
        if obj.track_id in self.redlight_violated_ids:
            return True

        current_center_y = obj.center_y
        prev_center_y = self.vehicle_prev_center_y.get(obj.track_id)

        # Lưu vị trí cho frame sau
        self.vehicle_prev_center_y[obj.track_id] = current_center_y

        # Đèn XANH hoặc VÀNG → cho phép đi, không xét
        if traffic_state != "red":
            return False

        # Đèn ĐỎ → kiểm tra tâm xe có vượt qua stop_line không
        if prev_center_y is not None:
            # Tâm xe đi từ TRÊN stop_line xuống DƯỚI stop_line
            if prev_center_y <= stop_line_y < current_center_y:
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
