"""
Violation Detection Logic
==========================
Phát hiện xe vi phạm vượt đèn đỏ.

Logic:
  - Dùng TÂM xe (center) để đại diện vị trí
  - So sánh cross product giữa frame trước và frame hiện tại
  - Sign thay đổi = xe vừa vượt qua stop line
  - Nếu không có prev_bbox → bỏ qua (chờ frame tiếp)
"""


def _cross_product(p1, p2, point):
    """Cross product (p1→p2) × (p1→point). Xác định point ở bên nào đường."""
    return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
            (p2[1] - p1[1]) * (point[0] - p1[0]))


def _get_center(bbox):
    """Tâm bbox: ((x1+x2)/2, (y1+y2)/2)"""
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def is_below_line(bbox, stop_line_pts):
    """
    Kiểm tra TÂM xe có đang ở DƯỚI stop line không.
    Dùng cross product:
      - Giả sử user vẽ stop line từ TRÁI sang PHẢI (P1.x < P2.x)
      - Cross < 0 → dưới đường
    Nếu user vẽ ngược (P1.x > P2.x), ta tự swap.
    """
    if stop_line_pts is None:
        return False

    p1, p2 = stop_line_pts
    # Đảm bảo P1 bên trái, P2 bên phải
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    center = _get_center(bbox)
    cross = _cross_product(p1, p2, center)
    return cross < 0


def has_crossed_line(prev_bbox, curr_bbox, stop_line_pts):
    """
    Kiểm tra TÂM xe có VỪA vượt qua stop line không.
    Sign change = vượt qua.
    """
    if stop_line_pts is None or prev_bbox is None or curr_bbox is None:
        return False

    p1, p2 = stop_line_pts
    # Swap nếu cần
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    prev_center = _get_center(prev_bbox)
    curr_center = _get_center(curr_bbox)

    prev_cross = _cross_product(p1, p2, prev_center)
    curr_cross = _cross_product(p1, p2, curr_center)

    if prev_cross == 0 or curr_cross == 0:
        return False

    # Dấu thay đổi = vượt qua
    sign_changed = (prev_cross > 0) != (curr_cross > 0)

    return sign_changed
