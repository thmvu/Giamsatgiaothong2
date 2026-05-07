"""
Violation Detection Logic
==========================
Phát hiện xe vi phạm vượt đèn đỏ.

Logic (Proximity + Side):
  - Dùng ĐIỂM GIỮA CẠNH DƯỚI bbox (cx, y2 = bánh xe) để kiểm tra.
  - Stop line được biểu diễn bằng đoạn thẳng ngang:
      ((x_left, stop_y), (x_right, stop_y))
    Trong đó stop_y = Y_MIN của polygon - offset_up (nâng vạch lên).
  - Điều kiện vi phạm:
      1. dist(bottom_center, stop_line) < threshold   → xe đang sát/qua vạch
      2. dy > 0  (bottom_center.y > stop_y)           → bánh xe đã VƯỢT qua vạch
      3. Đèn ĐỎ
  - Chỉ ghi nhận vi phạm 1 lần/xe (violation_memory).
  - Không cần prev_bbox → robust với frame skip, xe xuất hiện muộn.

Ghi chú toạ độ ảnh:
  - Y tăng từ trên xuống dưới.
  - dy = cy - stop_y  → dy > 0 khi bánh xe ở DƯỚI (đã qua) stop line.
"""

import cv2
import numpy as np
import os
import csv
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích hình học
# ─────────────────────────────────────────────────────────────────────────────

def point_line_distance(px, py, x1, y1, x2, y2):
    """
    Tính khoảng cách vuông góc từ điểm (px, py) đến đoạn thẳng (x1,y1)-(x2,y2).
    Trả về (dist, dy):
        dist : khoảng cách (px)
        dy   : py - yy (dương = điểm nằm DƯỚI đoạn thẳng, tức đã vượt qua)
    """
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    len_sq = C * C + D * D
    if len_sq == 0:
        # Đoạn thẳng có độ dài 0
        dx, dy_v = px - x1, py - y1
        return (dx * dx + dy_v * dy_v) ** 0.5, dy_v

    param = (A * C + B * D) / len_sq
    param = max(0.0, min(1.0, param))   # clamp về [0, 1]

    xx = x1 + param * C
    yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return (dx * dx + dy * dy) ** 0.5, dy


def get_stop_line_from_polygon(polygon_pts, offset_up=20):
    """
    Rút trích stop line ngang từ polygon vạch dừng.

    Trả về ((x_left, stop_y), (x_right, stop_y)) hoặc None.
        stop_y = Y_MIN(polygon) - offset_up   (nâng lên trên offset_up px)
    """
    if polygon_pts is None or len(polygon_pts) < 2:
        return None

    x_vals = [int(p[0]) if hasattr(p, '__len__') else int(p[0]) for p in polygon_pts]
    y_vals = [int(p[1]) if hasattr(p, '__len__') else int(p[1]) for p in polygon_pts]

    x_left  = min(x_vals)
    x_right = max(x_vals)
    stop_y  = max(0, min(y_vals) - offset_up)   # nâng lên offset_up px

    return ((x_left, stop_y), (x_right, stop_y))


# ─────────────────────────────────────────────────────────────────────────────
# Logic vi phạm chính
# ─────────────────────────────────────────────────────────────────────────────

def check_redlight_violation(vehicle_id, vehicle_bbox, stop_line_pts,
                              light_status, violation_memory, threshold=40):
    """
    Kiểm tra xe có vi phạm vượt đèn đỏ không (proximity + side).

    Tham số:
        vehicle_id      : track_id của xe
        vehicle_bbox    : [x1, y1, x2, y2]
        stop_line_pts   : ((x1,y1),(x2,y2)) — đoạn thẳng stop line
        light_status    : "red" / "green" / "yellow" / "unknown"
        violation_memory: dict {vehicle_id: {'saved': bool}} — dùng chung toàn session
        threshold       : khoảng cách tối đa (px) để tính là "đang ở vạch"

    Trả về:
        True  → vi phạm (mới hoặc đã ghi nhận trước đó)
        False → không vi phạm
    """
    # Nếu đã vi phạm trước đó → giữ trạng thái True
    if vehicle_id in violation_memory:
        return True

    # Chỉ check khi đèn đỏ
    if light_status != "red" or stop_line_pts is None:
        return False

    if vehicle_bbox is None:
        return False

    bbox = vehicle_bbox.tolist() if hasattr(vehicle_bbox, 'tolist') else list(vehicle_bbox)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Điểm kiểm tra = giữa cạnh dưới (bánh xe)
    cx = (x1 + x2) // 2
    cy = y2

    (lx1, ly1), (lx2, ly2) = stop_line_pts
    dist, dy = point_line_distance(cx, cy, lx1, ly1, lx2, ly2)

    # Điều kiện: sát vạch (dist < threshold) VÀ đã vượt qua (dy > 0)
    if dist < threshold and dy > 0:
        violation_memory[vehicle_id] = {'saved': False}
        return True

    return False


def save_violation_evidence(vehicle_id, frame, frame_number, x1, y1, x2, y2,
                             stop_line_pts, violation_memory, evidence_dir,
                             fps=25.0, cls_name="", vn_names=None):
    """
    Lưu ảnh bằng chứng + CSV khi phát hiện vi phạm lần đầu.
    Chỉ lưu 1 lần/xe.

    Trả về: đường dẫn file ảnh nếu lưu thành công, None nếu không.
    """
    mem = violation_memory.get(vehicle_id)
    if mem is None or mem.get('saved', False):
        return None   # chưa vi phạm hoặc đã lưu rồi

    try:
        os.makedirs(evidence_dir, exist_ok=True)

        ev_path = os.path.join(evidence_dir, f"redlight_ID{vehicle_id}_f{frame_number}.jpg")
        ev_frame = frame.copy()

        # Vẽ bbox đỏ
        cv2.rectangle(ev_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Vẽ stop line trên ảnh bằng chứng
        if stop_line_pts is not None:
            (lx1, ly1), (lx2, ly2) = stop_line_pts
            cv2.line(ev_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)

        label = f"VUOT DEN DO ID{vehicle_id}"
        cv2.putText(ev_frame, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(ev_path, ev_frame)

        # Ghi CSV
        csv_path = os.path.join(evidence_dir, "violations.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["vehicle_id", "frame_number", "class", "datetime"])
            writer.writerow([vehicle_id, frame_number, cls_name,
                             datetime.now().isoformat()])

        violation_memory[vehicle_id]['saved'] = True
        print(f"[INFO] Đã lưu bằng chứng: {ev_path}")
        return ev_path

    except Exception as e:
        print(f"[ERROR] Không lưu được bằng chứng ID{vehicle_id}: {e}")
        return None


def draw_stop_line_on_frame(frame, stop_line_pts, color=(0, 200, 255), thickness=2):
    """Vẽ stop line lên frame (dùng trong ảnh bằng chứng / calibration viz)."""
    if stop_line_pts is None or frame is None:
        return frame
    (lx1, ly1), (lx2, ly2) = stop_line_pts
    cv2.line(frame, (lx1, ly1), (lx2, ly2), color, thickness)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Dọn violation_memory: xoá xe không còn track
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_violation_memory(violation_memory, current_vehicle_ids):
    """Xoá các xe không còn được track để tiết kiệm bộ nhớ."""
    expired = [vid for vid in violation_memory if vid not in current_vehicle_ids]
    for vid in expired:
        del violation_memory[vid]


# ─────────────────────────────────────────────────────────────────────────────
# Hàm legacy (giữ để không break import cũ)
# ─────────────────────────────────────────────────────────────────────────────

def _cross_product(p1, p2, point):
    return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
            (p2[1] - p1[1]) * (point[0] - p1[0]))


def _get_center(bbox):
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def is_below_line(bbox, stop_line_pts):
    """[LEGACY]"""
    if stop_line_pts is None:
        return False
    p1, p2 = stop_line_pts
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    center = _get_center(bbox)
    return _cross_product(p1, p2, center) < 0


def has_crossed_line(prev_bbox, curr_bbox, stop_line_pts):
    """[LEGACY]"""
    if stop_line_pts is None or prev_bbox is None or curr_bbox is None:
        return False
    p1, p2 = stop_line_pts
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    prev_cross = _cross_product(p1, p2, _get_center(prev_bbox))
    curr_cross  = _cross_product(p1, p2, _get_center(curr_bbox))
    if prev_cross == 0 or curr_cross == 0:
        return False
    return (prev_cross > 0) != (curr_cross > 0)


def is_overlapping_polygon(bbox, polygon_pts):
    """[LEGACY]"""
    if polygon_pts is None or len(polygon_pts) < 3:
        return False
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    points_to_check = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        ((x1+x2)//2, (y1+y2)//2), ((x1+x2)//2, y2)
    ]
    poly = np.array(polygon_pts, np.int32)
    for pt in points_to_check:
        if cv2.pointPolygonTest(poly, pt, measureDist=False) >= 0:
            return True
    return False
