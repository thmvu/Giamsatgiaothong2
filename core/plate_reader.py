"""
License Plate Reader Module (RapidOCR + YOLO Detection)
========================================================
Pipeline 2 giai đoạn — "Combo Hủy Diệt" Biển Số:

  Giai đoạn 1 — LicensePlateDetector:
      YOLO / ONNX detect BBox biển số trong ảnh crop xe.
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
    .\\venv_paddle\\Scripts\\streamlit run app.py
"""

import os
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

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
    Hỗ trợ cả .pt và .onnx (best.onnx 1024px).
    Chính xác hơn crop thô: detect đúng bbox dù xe nghiêng/xa/khác góc.

    imgsz: Kích thước inference (640 = nhanh, 1024 = chính xác hơn với best.onnx)
           Có thể điều chỉnh qua sidebar mà không cần train lại.
    """

    def __init__(self, model_path: str = "models/biensoxe.onnx",
                 conf: float = 0.20, device: str = "cpu", imgsz: int = 640):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Khong tim thay model bien so: {model_path}")
        resolved_path = model_path
        self.model_path = resolved_path
        self.is_exported = os.path.splitext(resolved_path)[1].lower() != ".pt"
        self.model = YOLO(resolved_path, task="detect")
        if not self.is_exported:
            self.model.to(device)
        self.conf = conf
        self.device = device
        self.predict_device = "cpu" if self.is_exported else device
        self.imgsz = imgsz   # Dynamic: 640 (nhanh) hoặc 1024 (chính xác, dùng với best.onnx)
        print(f"[OK] LicensePlateDetector loaded: {resolved_path} [{device.upper()}] imgsz={imgsz}")

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

        # imgsz dynamic: 640 (default/MX110) hoặc 1024 (best.onnx, máy đủ mạnh)
        results = self.model.predict(
            vehicle_crop,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.predict_device,
        )
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

    def crop_best_plate(self, vehicle_crop: np.ndarray, padding_pct: float = 0.10):
        """
        Detect và trả về crop biển số có confidence cao nhất.
        padding_pct: Mở rộng bbox thêm X% mỗi chiều (default 10%).
                     PaddleOCR/RapidOCR nhận diện tốt hơn khi có padding rộng.

        Returns:
            (plate_crop_bgr, bbox) hoặc (None, None) nếu không tìm thấy.
        """
        plates = self.detect(vehicle_crop)
        if not plates:
            return None, None

        best = plates[0]
        x1, y1, x2, y2 = best['bbox']
        h, w = vehicle_crop.shape[:2]

        # Padding 10% kích thước bbox mỗi chiều (thay vì fixed 6px)
        # PaddleOCR cực kỳ thích padding rộng → nhận diện ký tự rìa tốt hơn
        pad_x = int((x2 - x1) * padding_pct)
        pad_y = int((y2 - y1) * padding_pct)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

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

    def __init__(self, use_gpu: bool = False):
        self._cache: dict = {}        # track_id -> plate_text (chỉ lưu khi có kết quả)
        self._retry_count: dict = {}  # track_id -> số lần OCR thất bại
        self._MAX_RETRIES = 15        # sau 15 lần thất bại mới dừng thử
        self._engine = "none"

        # ── Thử RapidOCR trước (model PaddleOCR + ONNX) ──
        if RAPIDOCR_AVAILABLE:
            # Thử bật GPU trước (cần onnxruntime-gpu)
            if use_gpu:
                try:
                    self._ocr = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
                    self._engine = "rapidocr"
                    print("[OK] RapidOCR (PaddleOCR model + ONNX) — GPU mode 🚀")
                    return
                except Exception as e:
                    print(f"[INFO] RapidOCR GPU không khả dụng ({e}) → fallback CPU")
            # Fallback CPU (không cần onnxruntime-gpu)
            try:
                self._ocr = RapidOCR()
                self._engine = "rapidocr"
                print("[OK] RapidOCR (PaddleOCR model + ONNX) — CPU mode")
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

        Cache logic:
          - Nếu đã đọc được biển số trước đó → trả cách luôn (chỉ OCR 1 lần/xe).
          - Nếu OCR thất bại → KHÔNG cache → frame sau thử lại (tối đa MAX_RETRIES lần).
          - Sau MAX_RETRIES lần thất bại → dừng thử để tiết kiệm CPU.
        """
        if not self.available:
            return ""

        # Cache hit với kết quả thực — không OCR lại
        if track_id is not None and track_id in self._cache:
            return self._cache[track_id]

        # Đã vượt số lần retry cho phép — dừng thử
        if track_id is not None and self._retry_count.get(track_id, 0) >= self._MAX_RETRIES:
            return ""

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
            if plate_text:
                # Đọc được → cache kết quả, xóa retry counter
                self._cache[track_id] = plate_text
                self._retry_count.pop(track_id, None)
                print(f"[OCR] ✅ Đọc biển số ID{track_id}: [{plate_text}] (engine: {self._engine})")
            else:
                # Thất bại → tăng retry counter, KHÔNG cache
                self._retry_count[track_id] = self._retry_count.get(track_id, 0) + 1
                retries = self._retry_count[track_id]
                if retries % 3 == 1:  # log mỗi 3 lần
                    print(f"[OCR] ⚠️  OCR thất bại ID{track_id} (lần {retries}/{self._MAX_RETRIES})")
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
                bbox_pts = item[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                raw_text = str(item[1])
                score = float(item[2]) if item[2] is not None else 0.0
                cleaned = self._clean_text(raw_text)
                if len(cleaned) >= 2 and score > min_score:
                    # Lấy tọa độ Y trung bình của bbox → sort dòng trên trước
                    try:
                        y_center = float(np.mean([pt[1] for pt in bbox_pts]))
                    except Exception:
                        y_center = 0.0
                    texts.append((cleaned, score, y_center))

        if not texts:
            return ""
        # Biển VN: dòng trên (Y nhỏ hơn) = mã tỉnh "51F", dòng dưới = số "12345"
        # → sort theo Y tăng dần (trên trước) thay vì độ dài
        texts.sort(key=lambda t: t[2])
        return '-'.join(t[0] for t in texts[:2])

    def _read_easyocr(self, img: np.ndarray, min_score: float = 0.5) -> str:
        """Đọc bằng EasyOCR (fallback)."""
        results = self._ocr.readtext(img, detail=1)
        best_text, best_conf = "", 0.0
        for (_, text, prob) in results:
            cleaned = self._clean_text(text)
            if len(cleaned) >= 2 and prob > min_score and prob > best_conf:
                best_text = cleaned
                best_conf = prob
        return best_text

    def get_cached_plate(self, track_id: int) -> str:
        """Lấy biển số đã cache theo track_id. Chỉ trả kết quả nếu có text."""
        return self._cache.get(track_id, "")

    def invalidate(self, track_id: int):
        """Xóa cache của 1 xe."""
        self._cache.pop(track_id, None)

    @property
    def cache(self) -> dict:
        return dict(self._cache)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh biển số nâng cao (đặc trị biển số siêu nhỏ/ở xa):
          1. Khử nhiễu giữ cạnh bằng Bilateral Filter để làm mịn nhiễu hạt vỡ màu của ảnh gốc nhỏ
          2. Siêu phân giải thích ứng bằng nội suy Cubic chất lượng cao
          3. CLAHE tăng tương phản thích nghi trên không gian màu LAB (clipLimit=4.0)
          4. Unsharp Masking (làm nét chuyên nghiệp) giúp ký tự sắc cạnh mà không bị viền nhiễu hạt
        """
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        
        # Bước 1: Khử nhiễu giữ cạnh trước khi phóng to (rất hiệu quả để dọn dẹp các đốm hạt nén JPEG)
        if h < 50:
            # Tham số d=5, sigmaColor=50, sigmaSpace=50 là tối ưu để mịn da biển số nhưng giữ nguyên cạnh chữ
            img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
            
        # Bước 2: Siêu phân giải thích ứng động
        scale_h = 110.0 / h
        scale_w = 330.0 / w
        scale = max(2.5, scale_h, scale_w)
        scale = min(scale, 6.0)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Bước 3: CLAHE tăng tương phản kênh sáng L (LAB)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        
        # Bước 4: Unsharp Masking làm sắc nét cạnh chữ thay cho bộ lọc nhân chập thô dễ gây vỡ hạt
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(img, 1.8, blur, -0.8, 0)
        
        # Bước 5: Thêm viền đệm trắng bao quanh (Margin Padding)
        # Đây là bí quyết để mô hình DBNet định vị dòng chữ hoàn hảo, tránh hiện tượng chữ chạm sát mép bị mô hình bỏ qua
        bordered = cv2.copyMakeBorder(sharpened, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        return bordered

    def _clean_text(self, text: str) -> str:
        """Chỉ giữ chữ/số/dấu chấm/gạch ngang."""
        return re.sub(r'[^A-Za-z0-9\-.]', '', text).upper()
