"""
License Plate Reader Module (RapidOCR + YOLO Detection)
========================================================
Pipeline 2 giai đoạn — "Combo Hủy Diệt" Biển Số:

  Giai đoạn 1 — LicensePlateDetector:
      YOLO (license_plate_detector.pt) → detect BBox biển số trong ảnh crop xe.
      Chính xác hơn nhiều so với crop thô phần dưới xe.

  Giai đoạn 2 — PlateReader (RapidOCR):
      RapidOCR = đúng model PaddleOCR (det + cls + rec) chạy qua ONNX Runtime.
      - Độ chính xác ngang PaddleOCR gốc.
      - Không cần paddlepaddle (tránh dependency hell).
      - Hỗ trợ biển số Việt Nam 1 dòng & 2 dòng.
      - Fallback sang EasyOCR nếu RapidOCR chưa cài.

Logic "Khoe Khéo":
  - CHỈ kích hoạt khi xe bị phát hiện VI PHẠM → tiết kiệm GPU đáng kể!
  - Cache kết quả theo track_id → chỉ OCR 1 lần/xe.

CÁCH CHẠY (venv Python 3.10):
    .\venv_paddle\Scripts\streamlit run app.py
"""

import cv2
import numpy as np
import re
from ultralytics import YOLO

try:
    # pyrefly: ignore [missing-import]
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Giai đoạn 1: YOLO License Plate Detector
# ─────────────────────────────────────────────────────────────────────────────

class LicensePlateDetector:
    """
    Phát hiện vùng biển số trong ảnh crop xe bằng YOLO.
    Dùng models/license_plate_detector.pt.
    Chính xác hơn crop thô: detect đúng bbox dù xe nghiêng/xa/khác góc.
    """

    def __init__(self, model_path: str = "models/license_plate_detector.pt", conf: float = 0.20, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.device = device
        print(f"[OK] LicensePlateDetector loaded: {model_path} [{device.upper()}]")

    def detect(self, vehicle_crop: np.ndarray) -> list:
        """
        Detect tất cả biển số trong ảnh crop xe.

        Returns:
            List[dict]: {'bbox': [x1,y1,x2,y2], 'conf': float}
            Sắp xếp theo conf giảm dần.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return []

        results = self.model.predict(vehicle_crop, conf=self.conf, verbose=False, device=self.device)
        plates = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
                conf_val = float(r.boxes.conf[i])
                plates.append({'bbox': [x1, y1, x2, y2], 'conf': conf_val})

        plates.sort(key=lambda p: p['conf'], reverse=True)
        return plates

    def crop_best_plate(self, vehicle_crop: np.ndarray, padding: int = 6):
        """
        Detect và trả về crop biển số có confidence cao nhất.

        Returns:
            (plate_crop_bgr, bbox) hoặc (None, None) nếu không tìm thấy.
        """
        plates = self.detect(vehicle_crop)
        if not plates:
            return None, None

        best = plates[0]
        x1, y1, x2, y2 = best['bbox']
        h, w = vehicle_crop.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None, None
        return plate_crop, best['bbox']


# ─────────────────────────────────────────────────────────────────────────────
# Giai đoạn 2: PlateReader (RapidOCR primary / EasyOCR fallback)
# ─────────────────────────────────────────────────────────────────────────────

class PlateReader:
    """
    Nhận dạng text biển số xe.

    Primary  : RapidOCR (model PaddleOCR chạy qua ONNX Runtime — chất lượng cao,
               không cần PaddlePaddle, hỗ trợ Python 3.10 & venv_paddle).
    Fallback : EasyOCR (Python 3.13 hệ thống) — nếu RapidOCR chưa cài.

    Cache kết quả theo track_id — chỉ OCR 1 lần/xe.
    """

    def __init__(self, use_gpu: bool = True):
        self._cache: dict = {}
        self._engine = "none"

        # ── Thử RapidOCR trước (model PaddleOCR + ONNX) ──
        if RAPIDOCR_AVAILABLE:
            try:
                # RapidOCR tự động dùng ONNX Runtime
                # GPU: cài thêm onnxruntime-gpu và đặt use_cuda=True
                self._ocr = RapidOCR()
                self._engine = "rapidocr"
                print("[OK] RapidOCR (PaddleOCR model + ONNX) khởi tạo thành công 🚀")
                return
            except Exception as e:
                print(f"[WARNING] RapidOCR init thất bại: {e} → thử EasyOCR")

        # ── Fallback: EasyOCR ──
        if EASYOCR_AVAILABLE:
            try:
                self._ocr = easyocr.Reader(['en'], gpu=use_gpu)
                self._engine = "easyocr"
                mode = "GPU 🚀" if use_gpu else "CPU"
                print(f"[OK] EasyOCR (fallback) khởi tạo — {mode} mode")
                return
            except Exception as e:
                print(f"[WARNING] EasyOCR init thất bại: {e}")

        self._ocr = None
        print("[ERROR] Không có OCR engine nào khả dụng!")

    @property
    def available(self) -> bool:
        return self._ocr is not None

    @property
    def engine(self) -> str:
        return self._engine

    def read_plate(self, plate_crop: np.ndarray, track_id: int = None, min_score: float = 0.5) -> str:
        """
        Đọc text biển số từ ảnh crop biển số.

        Args:
            plate_crop : Ảnh BGR của biển số (đã crop bởi LicensePlateDetector)
            track_id   : Dùng để cache — không bao giờ OCR lại cùng xe
            min_score  : Ngưỡng score OCR tối thiểu (từ sidebar slider)

        Returns:
            Chuỗi biển số VD: "51F-12345" hoặc ""
        """
        if not self.available:
            return ""

        # Cache hit — không OCR lại
        if track_id is not None and track_id in self._cache:
            return self._cache[track_id]

        if plate_crop is None or plate_crop.size == 0:
            return ""
        h, w = plate_crop.shape[:2]
        if h < 10 or w < 20:
            return ""

        processed = self._preprocess(plate_crop)

        try:
            if self._engine == "rapidocr":
                plate_text = self._read_rapid(processed, min_score)
            else:
                plate_text = self._read_easyocr(processed, min_score)
        except Exception as e:
            print(f"[ERROR] OCR ({self._engine}): {e}")
            plate_text = ""

        if track_id is not None:
            self._cache[track_id] = plate_text
        return plate_text

    def _read_rapid(self, img: np.ndarray, min_score: float = 0.5) -> str:
        """Đọc bằng RapidOCR — hỗ trợ biển 1 & 2 dòng VN."""
        result, elapse = self._ocr(img)
        if not result:
            return ""

        texts = []
        for item in result:
            # item = [bbox_pts, text, score]
            if len(item) >= 3:
                raw_text = str(item[1])
                score = float(item[2]) if item[2] is not None else 0.0
                cleaned = self._clean_text(raw_text)
                if len(cleaned) >= 3 and score > min_score:
                    texts.append((cleaned, score))

        if not texts:
            return ""
        # Biển 2 dòng: ghép dòng dài nhất lên trước
        texts.sort(key=lambda t: len(t[0]), reverse=True)
        return '-'.join(t[0] for t in texts[:2])

    def _read_easyocr(self, img: np.ndarray, min_score: float = 0.5) -> str:
        """Đọc bằng EasyOCR (fallback)."""
        results = self._ocr.readtext(img, detail=1)
        best_text, best_conf = "", 0.0
        for (_, text, prob) in results:
            cleaned = self._clean_text(text)
            if len(cleaned) >= 3 and prob > min_score and prob > best_conf:
                best_text = cleaned
                best_conf = prob
        return best_text

    def get_cached_plate(self, track_id: int) -> str:
        """Lấy biển số đã cache theo track_id."""
        return self._cache.get(track_id, "")

    def invalidate(self, track_id: int):
        """Xóa cache của 1 xe."""
        self._cache.pop(track_id, None)

    @property
    def cache(self) -> dict:
        return dict(self._cache)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh biển số:
          1. Scale 2x → OCR nhạy hơn với biển nhỏ
          2. CLAHE → tăng contrast tự thích nghi
          3. Sharpen → nét ký tự hơn
        """
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def _clean_text(self, text: str) -> str:
        """Chỉ giữ chữ/số/dấu chấm/gạch ngang."""
        return re.sub(r'[^A-Za-z0-9\-.]', '', text).upper()
