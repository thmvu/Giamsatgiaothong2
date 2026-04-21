"""
Drawing Utilities
==================
Hàm vẽ bounding box, label, stop line lên frame.
"""

import cv2


def draw_box(frame, bbox, label, color=(0, 255, 0), thickness=2):
    """
    Vẽ bounding box + label lên frame.

    Args:
        frame: ảnh BGR
        bbox: (x1, y1, x2, y2) hoặc tensor
        label: text hiển thị
        color: màu BGR
        thickness: độ dày viền
    """
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Vẽ khung
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Vẽ nền cho label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_stop_line(frame, stop_line_y):
    """Vẽ vạch dừng ngang khung hình."""
    w = frame.shape[1]
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (255, 0, 255), 3)
    cv2.putText(frame, "STOP LINE", (10, stop_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def draw_light_status(frame, light_state):
    """Vẽ HUD trạng thái đèn góc trên trái."""
    info = {
        "red": ("DEN DO", (0, 0, 255)),
        "green": ("DEN XANH", (0, 255, 0)),
        "yellow": ("DEN VANG", (0, 255, 255)),
        "unknown": ("--", (128, 128, 128)),
    }
    text, color = info.get(light_state, ("--", (128, 128, 128)))
    cv2.rectangle(frame, (5, 5), (230, 42), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (230, 42), color, 2)
    cv2.putText(frame, f"Den: {text}", (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
