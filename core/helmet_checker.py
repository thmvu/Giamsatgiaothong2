"""
Helmet Checker Module
=====================
Sử dụng model phathienmu1.pt để kiểm tra mũ bảo hiểm
trên ảnh crop xe máy.
Classes: With Helmet, Without Helmet
"""

from ultralytics import YOLO
import numpy as np


class HelmetChecker:
    """
    Expert 3: Kiểm tra mũ bảo hiểm.
    Nhận ảnh crop xe máy → trả về có vi phạm hay không.
    """

    def __init__(self, model_path: str = "models/phathienmu1.pt"):
        self.model = YOLO(model_path)

    def check(self, crop_img: np.ndarray, conf: float = 0.4) -> bool:
        """
        Kiểm tra ảnh crop xe máy có người không đội mũ không.

        Args:
            crop_img: Ảnh crop (BGR) của xe máy
            conf: Ngưỡng confidence

        Returns:
            True nếu VI PHẠM (Without Helmet), False nếu OK
        """
        if crop_img is None or crop_img.size == 0:
            return False
        if crop_img.shape[0] < 15 or crop_img.shape[1] < 15:
            return False

        results = self.model.predict(crop_img, conf=conf, verbose=False)

        for res in results:
            if res.boxes is None:
                continue
            for b in res.boxes:
                label = res.names[int(b.cls[0])]
                # Hỗ trợ nhiều tên class phổ biến từ các bộ dataset khác nhau
                if label.lower() in ('without helmet', 'without_helmet', 'no_helmet', 'nohelmet', 'no helmet'):
                    return True

        return False
