"""
License Plate Reader Module
============================
Sử dụng EasyOCR để nhận dạng biển số xe.
Crop phần dưới phương tiện → tiền xử lý → OCR.
"""

import cv2
import numpy as np
import re

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class PlateReader:
    """
    Expert 4: Nhận dạng biển số.
    OCR chạy 1 lần duy nhất mỗi xe (cache kết quả theo track_id).
    """

    def __init__(self, languages=None, gpu=False):
        if not EASYOCR_AVAILABLE:
            self.reader = None
            return
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self._cache = {}  # track_id → plate_text

    @property
    def available(self):
        return self.reader is not None

    def read_plate(self, vehicle_crop: np.ndarray, track_id: int = None) -> str:
        """
        Đọc biển số từ ảnh crop phương tiện.
        Crop phần dưới 40% (vùng biển số) → OCR.

        Args:
            vehicle_crop: Ảnh crop (BGR) của phương tiện
            track_id: ID tracking (dùng để cache, tránh OCR trùng)

        Returns:
            Chuỗi biển số (hoặc "" nếu không đọc được)
        """
        if not self.available:
            return ""

        # Đã cache → trả về luôn
        if track_id is not None and track_id in self._cache:
            return self._cache[track_id]

        if vehicle_crop is None or vehicle_crop.size == 0:
            return ""

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return ""

        # Crop vùng biển số (phần dưới 40%)
        plate_region = vehicle_crop[int(h * 0.6):, :]

        # Tiền xử lý
        processed = self._preprocess(plate_region)

        try:
            ocr_results = self.reader.readtext(processed, detail=1)
            best_text = ""
            best_conf = 0

            for (bbox, text, prob) in ocr_results:
                cleaned = self._clean_text(text)
                if len(cleaned) >= 4 and prob > best_conf:
                    best_text = cleaned
                    best_conf = prob

            # Cache kết quả
            if track_id is not None:
                self._cache[track_id] = best_text

            return best_text

        except Exception:
            if track_id is not None:
                self._cache[track_id] = ""
            return ""

    def get_cached_plate(self, track_id: int) -> str:
        """Lấy biển số đã cache theo track_id."""
        return self._cache.get(track_id, "")

    @property
    def cache(self):
        return dict(self._cache)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Tăng contrast + binarize cho OCR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _clean_text(self, text: str) -> str:
        """Loại ký tự nhiễu, giữ chữ/số."""
        cleaned = re.sub(r'[^A-Za-z0-9\-.]', '', text)
        return cleaned.upper()
