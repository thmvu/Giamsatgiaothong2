"""
Violation Detection Logic
==========================
Phát hiện xe vi phạm vượt đèn đỏ bằng cách kiểm tra
xe có vượt qua ĐƯỜNG THẲNG xác định bởi 2 điểm stop_line không.

Nguyên lý hình học:
  - Lấy tích có hướng (cross product) để xác định xe đang ở phía nào đường thẳng
  - Frame trước: xe ở phía trên/ngang → Frame hiện tại: xe ở phía dưới → VI PHẠM
  - Đèn XANH → bỏ qua, cho qua
  - Đèn ĐỎ + xe vượt đường → VI PHẠM
"""


def _cross_product_side(p1, p2, point):
    """
    Tính phía của 'point' so với đường thẳng qua p1→p2.
    Dùng cross product: (p2-p1) × (point-p1)
    > 0 → bên trái
    < 0 → bên phải
    = 0 → trên đường
    """
    return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
            (p2[1] - p1[1]) * (point[0] - p1[0]))


def point_below_line(p1, p2, point):
    """
    Kiểm tra 'point' có ở phía DƯỚI đường p1→p2 không.
    "Dưới" = phía xe di chuyển từ trên xuống (y tăng dần trong ảnh).
    Giả sử p1 là điểm bên trái, p2 là điểm bên phải.
    """
    cross = _cross_product_side(p1, p2, point)
    # Trong tọa độ ảnh (Y tăng xuống dưới):
    # cross < 0 → điểm ở phía phải đường → tức là phía dưới nếu đường gần nằm ngang
    # Cụ thể: với đường từ trái sang phải, phía dưới = cross < 0
    return cross < 0


def check_violation(light_state, vehicle_bbox, stop_line_pts):
    """
    Kiểm tra xe có vượt đèn đỏ không.

    Args:
        light_state: 'red' | 'green' | 'yellow' | 'unknown'
        vehicle_bbox: (x1, y1, x2, y2)
        stop_line_pts: ((x1,y1), (x2,y2)) — 2 điểm xác định vạch dừng
                       hoặc None nếu chưa đặt

    Returns:
        True nếu VI PHẠM
    """
    if light_state != "red":
        return False

    if stop_line_pts is None:
        return False

    # Tâm dưới của xe (center-bottom) — điểm đại diện vị trí xe trên mặt đường
    if hasattr(vehicle_bbox, 'tolist'):
        vehicle_bbox = vehicle_bbox.tolist()

    x1, y1, x2, y2 = vehicle_bbox
    center_x = int((x1 + x2) / 2)
    bottom_y = int(y2)
    vehicle_point = (center_x, bottom_y)

    p1, p2 = stop_line_pts

    # Xe ở phía dưới đường stop line = đã vượt qua
    return point_below_line(p1, p2, vehicle_point)


def has_crossed_line(prev_bbox, curr_bbox, stop_line_pts):
    """
    Kiểm tra xe có VỪA vượt qua stop line trong frame này không.
    (So sánh vị trí frame trước vs frame hiện tại)

    Returns:
        True nếu xe vừa vượt qua (transition: trên → dưới)
    """
    if stop_line_pts is None or prev_bbox is None:
        return False

    p1, p2 = stop_line_pts

    def center_bottom(bbox):
        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    prev_pt = center_bottom(prev_bbox)
    curr_pt = center_bottom(curr_bbox)

    was_above = not point_below_line(p1, p2, prev_pt)
    is_below = point_below_line(p1, p2, curr_pt)

    return was_above and is_below
