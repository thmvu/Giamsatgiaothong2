"""
Violation Detection Logic
==========================
Phát hiện xe vi phạm vượt đèn đỏ.

Nguyên lý:
  - Dùng cross product để xác định xe ở phía nào của stop line
  - Khi sign(cross) thay đổi giữa 2 frame → xe VỪA vượt qua
  - Chỉ tính vi phạm khi đang đèn ĐỎ
  - Chỉ tính khi xe đi từ trên xuống (y tăng) → tránh false positive khi lùi
"""


def _cross_product(p1, p2, point):
    """
    Tích có hướng (cross product) của vector p1→p2 với p1→point.
    Kết quả > 0 hoặc < 0 cho biết point đang ở phía nào của đường.
    """
    return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
            (p2[1] - p1[1]) * (point[0] - p1[0]))


def _get_center_bottom(bbox):
    """Lấy điểm tâm-dưới của bbox — đại diện vị trí xe trên mặt đường."""
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (int((x1 + x2) / 2), int(y2))


def has_crossed_line(prev_bbox, curr_bbox, stop_line_pts):
    """
    Kiểm tra xe có VỪA vượt qua stop line trong frame này không.

    Logic:
      - Tính cross product của tâm-dưới xe với stop line ở frame trước (prev)
        và frame hiện tại (curr)
      - Nếu dấu cross product thay đổi → xe vừa vượt qua
      - Chỉ tính khi xe đang di chuyển XUỐNG (y tăng) → không phạt xe lùi

    Args:
        prev_bbox: (x1,y1,x2,y2) của xe ở frame TRƯỚC
        curr_bbox: (x1,y1,x2,y2) của xe ở frame HIỆN TẠI
        stop_line_pts: ((x1,y1), (x2,y2)) — 2 điểm xác định vạch dừng

    Returns:
        True nếu xe vừa vượt qua stop line
    """
    if stop_line_pts is None or prev_bbox is None or curr_bbox is None:
        return False

    p1, p2 = stop_line_pts

    prev_pt = _get_center_bottom(prev_bbox)
    curr_pt = _get_center_bottom(curr_bbox)

    # Cross product ở 2 frame
    prev_cross = _cross_product(p1, p2, prev_pt)
    curr_cross = _cross_product(p1, p2, curr_pt)

    # Bỏ qua nếu nằm ngay trên đường (cross = 0)
    if prev_cross == 0 or curr_cross == 0:
        return False

    # Dấu thay đổi → xe vừa vượt qua đường
    sign_changed = (prev_cross > 0) != (curr_cross > 0)

    # Xe phải đang đi XUỐNG (y tăng) để tránh false positive
    moving_down = curr_pt[1] > prev_pt[1]

    return sign_changed and moving_down
