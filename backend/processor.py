"""
VideoProcessor — AI Pipeline (không Streamlit)
==============================================
Tách hoàn toàn logic AI khỏi UI, dùng cho FastAPI WebSocket.
"""

import os
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2, gc, sys, numpy as np, torch
from dataclasses import dataclass, field
from typing import Callable, Optional

# Thêm thư mục gốc vào sys.path để import utils/, core/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultralytics import YOLO, SAM
from utils.drawing import draw_box, draw_light_status, draw_polygon
from utils.violation import (
    check_redlight_violation, save_violation_evidence,
    get_stop_line_from_polygon, cleanup_violation_memory,
)
from core.plate_reader import LicensePlateDetector, PlateReader


@dataclass
class ProcessorConfig:
    conf_light: float = 0.5
    conf_vehicle: float = 0.4
    conf_helmet: float = 0.4
    conf_lp: float = 0.15
    conf_ocr: float = 0.1
    conf_stop_line: float = 0.3
    process_every_n: int = 2
    traffic_interval: int = 5
    display_every_n: int = 3
    display_width: int = 640
    yolo_imgsz: int = 640
    lp_imgsz: int = 1024
    stop_line_extend_left: int = 150
    stop_line_extend_right: int = 150
    stop_line_offset_up: int = 30
    min_width_pct: float = 8.0
    check_redlight: bool = True
    check_helmet: bool = True
    check_plate: bool = True
    show_all: bool = True
    stop_line_model_path: str = "models/phathienvachmoi.onnx"
    lp_model_path: str = "models/biensoxe.onnx"
    evidence_dir: str = "evidence"
    jpeg_quality: int = 75


class VideoProcessor:
    VEHICLE_CLASSES = [2, 3, 5, 7]
    MOTORBIKE_CLASS = 3
    VN_NAMES = {2: "Ô tô", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải"}

    def __init__(self, config: ProcessorConfig):
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._stopped = False

    def _resolve_existing_path(self, preferred_path: str, *fallbacks: str) -> str:
        for path in (preferred_path, *fallbacks):
            if path and os.path.exists(path):
                return path
        return preferred_path

    def _is_exported_model(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() != ".pt"

    def _predict_device(self, path: str):
        if self._is_exported_model(path):
            return "cpu"
        return self.device

    # ── Model loading ─────────────────────────────────────────────────────────

    def _yolo(self, path, task: str = "detect"):
        m = YOLO(path, task=task)
        if not self._is_exported_model(path):
            m.to(self.device)
        return m

    def load_models(self):
        cfg = self.cfg
        self.light_model   = self._yolo("models/phathienden.pt")
        self.vehicle_model = self._yolo("models/yolo11m.pt")
        self.helmet_model  = self._yolo("models/phathienmu1.pt")
        lp_path = cfg.lp_model_path
        if not os.path.exists(lp_path):
            raise FileNotFoundError(f"Khong tim thay model bien so: {lp_path}")
        self.lp_detector  = LicensePlateDetector(lp_path, conf=cfg.conf_lp, device=self.device, imgsz=cfg.lp_imgsz)
        self.plate_reader = PlateReader(use_gpu=False)
        print(f"[OK] Models loaded — device={self.device.upper()}")

    def stop(self):
        self._stopped = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode(self, frame) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        return bytes(buf) if ok else b""

    def _check_red_hsv(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        mid_y = y1 + (y2 - y1) // 2
        crop = frame[y1:mid_y, x1:x2]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) + \
            cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        return cv2.countNonZero(m) / max(crop.shape[0] * crop.shape[1], 1) > 0.05

    # ── Calibration ───────────────────────────────────────────────────────────

    def _extract_stop_polygon(self, frame, yolo_line, sam2, light_bbox=None):
        cfg = self.cfg
        h, w = frame.shape[:2]
        roi_top = int(h * 0.4)
        roi = frame[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_enh = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        results = yolo_line.predict(
            roi_enh,
            conf=cfg.conf_stop_line,
            imgsz=1024,
            verbose=False,
            device=self._predict_device(cfg.stop_line_model_path),
        )
        boxes = []
        for r in results:
            if r.boxes is None: continue
            for i in range(len(r.boxes)):
                bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                bw, bh = bx2 - bx1, max(by2 - by1, 1)
                if bw / bh < 4.0 or bw < roi_w * cfg.min_width_pct / 100: continue
                boxes.append(([bx1, by1 + roi_top, bx2, by2 + roi_top], float(r.boxes.conf[i]), bw))

        if not boxes:
            return None, None

        def score(item):
            bb, c, bw = item
            cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            ws = bw / w
            if light_bbox:
                lx, ly, lx2, ly2 = light_bbox
                if cy <= ly2: return -1.0
                dist = ((cx-(lx+lx2)/2)**2 + (cy-ly2)**2)**0.5
                ds = 1.0 - min(dist / (w**2+h**2)**0.5, 1.0)
            else:
                ds = 0.5
            return 0.6*ws + 0.4*ds

        best = max(boxes, key=score)
        if score(best) < 0: return None, None

        bx1, by1, bx2, by2 = map(int, best[0])
        ext_left  = cfg.stop_line_extend_left
        ext_right = cfg.stop_line_extend_right

        try:
            sam_res = sam2.predict(frame, bboxes=[best[0]], verbose=False, device="cpu")
            if sam_res and sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
                mask = sam_res[0].masks.data[0].cpu().numpy()
                m8 = (mask * 255).astype(np.uint8)
                if m8.shape[:2] != (h, w):
                    m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_NEAREST)
                clip = np.zeros_like(m8)
                clip_x1 = max(bx1 - ext_left, 0)
                clip_x2 = min(bx2 + ext_right, w)
                clip[max(by1,0):min(by2,h), clip_x1:clip_x2] = \
                    m8[max(by1,0):min(by2,h), clip_x1:clip_x2]
                cnts, _ = cv2.findContours(clip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    lg = max(cnts, key=cv2.contourArea)
                    eps = 0.008 * cv2.arcLength(lg, True)
                    poly = np.array(cv2.approxPolyDP(lg, eps, True).reshape(-1, 2), np.int32)
                    mid_x = (bx1 + bx2) / 2
                    for pt in poly:
                        if pt[0] < mid_x:
                            pt[0] = max(0, pt[0] - ext_left)
                        else:
                            pt[0] = min(w, pt[0] + ext_right)
                    return poly, int(max(p[1] for p in poly))
        except Exception as e:
            print(f"[SAM2] {e}")

        bx1e = max(0, bx1 - ext_left)
        bx2e = min(w, bx2 + ext_right)
        return np.array([[bx1e,by1],[bx2e,by1],[bx2e,by2],[bx1e,by2]], np.int32), by2

    # ── Main process ──────────────────────────────────────────────────────────

    def _try_read_plate(self, frame, x1, y1, x2, y2, track_id, conf_lp, conf_ocr,
                        evidence_dir="evidence", frame_count=0, log_detect=True):
        """
        OCR biển số xe — trả về (plate_text, bbox_in_frame_or_None, crop_path_or_None).
        
        Luồng:
          1. Kiểm tra cache trước
          2. Crop vùng xe → YOLO detect bbox biển số
          3. Log vị trí biển số (trong xe + trong frame gốc)
          4. Lưu crop biển số vào evidence/plates/
          5. OCR → trả về text + bbox + path crop
        """
        if not self.lp_detector or not track_id:
            return "", None, None

        # Dừng ngay nếu đã bị cancel
        if self._stopped:
            return "", None, None

        self.lp_detector.conf = conf_lp

        # Cache hit — không detect lại
        cached = self.plate_reader.get_cached_plate(track_id)
        if cached:
            return cached, None, None

        vc = frame[y1:y2, x1:x2]
        if vc.size == 0 or min(vc.shape[:2]) <= 20:
            return "", None, None

        # ── Detect bbox biển số bằng YOLO ─────────────────────────────
        plates = self.lp_detector.detect(vc)
        plate_crop = None
        bbox_in_frame = None
        crop_path = None

        if plates:
            best = plates[0]
            px1, py1, px2, py2 = best['bbox']
            conf_val = best['conf']

            # Vị trí biển số trong xe (relative to vehicle crop)
            veh_h, veh_w = vc.shape[:2]

            # Chuyển sang tọa độ frame gốc
            fpx1 = x1 + px1
            fpy1 = y1 + py1
            fpx2 = x1 + px2
            fpy2 = y1 + py2
            bbox_in_frame = [fpx1, fpy1, fpx2, fpy2]

            if log_detect:
                print(f"[BIEN SO] 🔍 YOLO detect biển số ID{track_id} conf={conf_val:.2f}"
                      f" | trong xe: [{px1},{py1},{px2},{py2}] (xe {veh_w}x{veh_h}px)"
                      f" | trong frame: [{fpx1},{fpy1},{fpx2},{fpy2}]"
                      f" | kích thước biển: {px2-px1}x{py2-py1}px")

            # Crop với padding 15% để tránh mất rìa ký tự cho biển số nhỏ
            pad_x = max(8, int((px2 - px1) * 0.15))
            pad_y = max(4, int((py2 - py1) * 0.15))
            cx1 = max(0, px1 - pad_x)
            cy1 = max(0, py1 - pad_y)
            cx2 = min(veh_w, px2 + pad_x)
            cy2 = min(veh_h, py2 + pad_y)
            plate_crop = vc[cy1:cy2, cx1:cx2]

            # Lưu crop biển số (lưu ảnh thô gốc + ảnh chính đã qua xử lý làm nét siêu phân giải)
            if plate_crop is not None and plate_crop.size > 0:
                plates_dir = os.path.join(evidence_dir, "plates")
                os.makedirs(plates_dir, exist_ok=True)
                
                # 1. Lưu ảnh gốc thô làm bằng chứng kỹ thuật
                raw_path = os.path.join(plates_dir, f"plate_ID{track_id}_f{frame_count}_raw.jpg")
                cv2.imwrite(raw_path, plate_crop)
                
                # 2. Tạo ảnh tiền xử lý làm nét siêu phân giải chất lượng cao
                processed_plate = self.plate_reader._preprocess(plate_crop)
                
                # 3. Lưu ảnh đã làm nét làm ảnh chính để người dùng xem trực quan
                crop_path = os.path.join(plates_dir, f"plate_ID{track_id}_f{frame_count}.jpg")
                cv2.imwrite(crop_path, processed_plate)
                
                if log_detect:
                    print(f"[BIEN SO] 💾 Crop biển số lưu: {crop_path} (đã làm nét siêu phân giải) | Ảnh thô: {raw_path}")

            # OCR
            if plate_crop is not None and plate_crop.size > 0:
                text = self.plate_reader.read_plate(plate_crop, track_id, conf_ocr)
            else:
                text = ""
        else:
            # Fallback: crop 35% dưới xe
            fb = vc[int(vc.shape[0] * 0.65):, :]
            if fb.size > 0:
                if log_detect:
                    fb_y = y1 + int(vc.shape[0] * 0.65)
                    print(f"[BIEN SO] ⚠️  Không detect được bbox biển số ID{track_id}"
                          f" → fallback crop dưới xe: y=[{fb_y}..{y2}] trong frame")
                text = self.plate_reader.read_plate(fb, track_id, conf_ocr)
                plate_crop = fb
            else:
                text = ""

        # Log kết quả OCR
        if text and log_detect:
            engine = getattr(self.plate_reader, '_engine', 'unknown')
            engine_label = {
                'rapidocr': 'RapidOCR (ONNX PaddleOCR)',
                'easyocr': 'EasyOCR (fallback)',
            }.get(engine, engine)
            print(f"[BIEN SO] 🔢 OCR kết quả ID{track_id}: [{text}]"
                  f" | engine: {engine_label}")

        return text, bbox_in_frame, crop_path


    def process(self, video_path: str,
                on_frame: Callable[[bytes, dict], None],
                on_violation: Optional[Callable[[dict], None]] = None,
                on_calib: Optional[Callable[[bytes], None]] = None,
                on_done: Optional[Callable[[dict], None]] = None,
                on_progress: Optional[Callable[[float, str], None]] = None,
                on_plate_update: Optional[Callable[[dict], None]] = None):
        """Blocking call — run in ThreadPoolExecutor."""

        def prog(pct, msg):
            if on_progress: on_progress(pct, msg)

        cfg = self.cfg
        os.makedirs(cfg.evidence_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # ── Calibration ──
        prog(0.1, "🔍 Phân tích frame đầu tiên...")
        ret, first = cap.read()
        if not ret:
            if on_done: on_done({"error": "Không đọc được video"})
            return

        frame_count = 1

        prog(0.3, "🚦 Tìm đèn giao thông...")
        light_bbox_c = None
        for r in self.light_model(first, conf=0.3, verbose=False):
            if r.boxes is not None and len(r.boxes) > 0:
                bc = max(range(len(r.boxes)), key=lambda i: float(r.boxes.conf[i]))
                light_bbox_c = r.boxes.xyxy[bc].tolist()

        prog(0.5, "🎯 Load SAM2...")
        stop_line_model_path = cfg.stop_line_model_path
        if not os.path.exists(stop_line_model_path):
            raise FileNotFoundError(f"Khong tim thay model vach dung: {stop_line_model_path}")
        yolo_line = self._yolo(stop_line_model_path)
        sam2 = SAM("models/sam2_b.pt"); sam2.to("cpu")

        stop_poly = None
        calib_frames = [first]
        for _ in range(4):
            r2, f2 = cap.read()
            if not r2: break
            calib_frames.append(f2); frame_count += 1

        for idx, cf in enumerate(calib_frames):
            if self._stopped: break
            prog(0.5 + 0.2*idx/5, f"🎯 Frame {idx+1}/{len(calib_frames)}...")
            poly, _ = self._extract_stop_polygon(cf, yolo_line, sam2, light_bbox_c)
            if poly is not None:
                stop_poly = poly
                if on_calib:
                    viz = cf.copy()
                    draw_polygon(viz, poly, (255, 0, 255), 2, alpha=0.4)
                    on_calib(self._encode(viz))
                break

        del yolo_line, sam2
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        stop_line = get_stop_line_from_polygon(stop_poly, offset_up=cfg.stop_line_offset_up) if stop_poly is not None else None
        if stop_line is not None:
            (sx1, sy), (sx2, _) = stop_line
            print(f"[VACH DUNG] ✅ Đã phát hiện vạch dừng — Y={sy}px  X=[{sx1}..{sx2}]  rộng={sx2-sx1}px")
            prog(0.8, "✅ Đã phát hiện vạch dừng — bắt đầu theo dõi...")
        else:
            print("[VACH DUNG] ⚠️  Không tìm thấy vạch dừng — tiếp tục không có vạch dừng")
            prog(0.8, "⚠️ Không tìm thấy vạch dừng — theo dõi không có vạch...")

        # ── Detection loop ──
        current_light   = "unknown"
        redlight_memory = {}
        helmet_ids      = set()
        plate_cache     = {}
        violations_log  = []
        processed       = 0

        while cap.isOpened() and not self._stopped:
            ret, frame = cap.read()
            if not ret: break
            if self._stopped: break
            frame_count += 1
            if frame_count % cfg.process_every_n != 0: continue
            processed += 1

            # Đèn
            if processed % cfg.traffic_interval == 1 or cfg.traffic_interval == 1:
                for r in self.light_model(frame, conf=cfg.conf_light, imgsz=cfg.yolo_imgsz, verbose=False):
                    if r.boxes is None: continue
                    bc2 = 0
                    for i in range(len(r.boxes)):
                        cv2_val = float(r.boxes.conf[i])
                        if cv2_val > bc2:
                            bc2 = cv2_val
                            x1,y1,x2,y2 = map(int, r.boxes.xyxy[i].tolist())
                            name = self.light_model.names[int(r.boxes.cls[i])]
                            is_red = self._check_red_hsv(frame, (x1,y1,x2,y2))
                            if is_red: current_light = "red"
                            elif "green" in name.lower(): current_light = "green"
                            elif "yellow" in name.lower(): current_light = "yellow"
                            col = (0,0,255) if is_red else (0,255,0) if current_light=="green" else (0,255,255)
                            draw_box(frame, [x1,y1,x2,y2], f"Light({current_light}) {cv2_val:.2f}", col)

            # Xe
            if self._stopped: break
            veh_res = self.vehicle_model.track(
                frame, conf=cfg.conf_vehicle, classes=self.VEHICLE_CLASSES,
                persist=True, tracker="bytetrack.yaml", imgsz=cfg.yolo_imgsz, verbose=False
            )
            cur_ids = set()
            for r in veh_res:
                if r.boxes is None or len(r.boxes) == 0: continue
                # Chuyển đổi toàn bộ tensor sang danh sách CPU một lần duy nhất để tối ưu hiệu năng tối đa (không nghẽn CUDA sync)
                boxes_list = r.boxes.xyxy.int().cpu().tolist()
                clss_list = r.boxes.cls.int().cpu().tolist()
                ids_list = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes_list)

                for i in range(len(boxes_list)):
                    bbox = boxes_list[i]
                    x1, y1, x2, y2 = bbox
                    cls_id = clss_list[i]
                    track_id = ids_list[i]
                    if track_id: cur_ids.add(track_id)
                    cls_name = self.VN_NAMES.get(cls_id, self.vehicle_model.names[cls_id])

                    is_rl = is_hm = False
                    plate_text = ""

                    # ── Bước 1: Kiểm tra đèn đỏ ─────────────────────────────
                    is_rl_new = False   # True = vi phạm MỚI frame này (chưa lưu trước đó)
                    if cfg.check_redlight and stop_line and track_id:
                        was_violated = track_id in redlight_memory
                        is_rl = check_redlight_violation(track_id, bbox, stop_line, current_light, redlight_memory, 40)
                        if is_rl and not was_violated:
                            is_rl_new = True   # mới phát hiện frame này

                    # ── Bước 2: Kiểm tra mũ bảo hiểm ────────────────────────
                    is_hm_new = False
                    if cfg.check_helmet and cls_id == self.MOTORBIKE_CLASS:
                        if track_id and track_id in helmet_ids:
                            is_hm = True
                        else:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0 and min(crop.shape[:2]) > 15:
                                for hr in self.helmet_model.predict(crop, conf=cfg.conf_helmet, imgsz=320, verbose=False):
                                    for b in hr.boxes:
                                        lbl = hr.names[int(b.cls[0])].lower()
                                        if lbl in ('without helmet','without_helmet','no_helmet','nohelmet','no helmet'):
                                            is_hm = True
                                            is_hm_new = True
                                            if track_id: helmet_ids.add(track_id)
                                            break

                    # ── Bước 3: OCR biển số — chạy cho xe vi phạm hoặc tất cả xe nếu bật show_all ──
                    plate_bbox_in_frame = None
                    plate_crop_path = None
                    should_read_plate = (not self._stopped) and cfg.check_plate and track_id and ((is_rl or is_hm) or cfg.show_all)
                    if should_read_plate:
                        plate_text, plate_bbox_in_frame, plate_crop_path = self._try_read_plate(
                            frame, x1, y1, x2, y2, track_id,
                            cfg.conf_lp, cfg.conf_ocr,
                            evidence_dir=cfg.evidence_dir,
                            frame_count=frame_count,
                            log_detect=True,
                        )
                        if plate_text:
                            plate_cache[track_id] = plate_text

                    # ── Bước 4: Lưu bằng chứng + gọi on_violation (kèm plate) ─
                    if is_rl_new:
                        ev = save_violation_evidence(track_id, frame, frame_count, x1,y1,x2,y2,
                                                     stop_line, redlight_memory, cfg.evidence_dir, fps, cls_name)
                        if ev:
                            vio = {"time": round(frame_count/fps,2), "frame": frame_count,
                                   "type": "Vượt đèn đỏ", "vehicle": cls_name,
                                   "track_id": track_id, "plate": plate_text, "evidence": ev}
                            violations_log.append(vio)
                            if on_violation: on_violation(vio)
                            print(f"[VI PHAM] 🔴 VUOT DEN DO — ID{track_id} ({cls_name}) | t={round(frame_count/fps,2)}s"
                                  f"{f' | BS: [{plate_text}]' if plate_text else ' | BS: (chưa đọc được)'}")

                    if is_hm_new:
                        ev2 = os.path.join(cfg.evidence_dir, f"helmet_ID{track_id}_f{frame_count}.jpg")
                        cv2.imwrite(ev2, frame)
                        vio2 = {"time": round(frame_count/fps,2), "frame": frame_count,
                                "type": "Không đội mũ", "vehicle": cls_name,
                                "track_id": track_id, "plate": plate_text, "evidence": ev2}
                        violations_log.append(vio2)
                        if on_violation: on_violation(vio2)
                        print(f"[VI PHAM] 🪖 KHONG MU BAO HIEM — ID{track_id} ({cls_name}) | t={round(frame_count/fps,2)}s"
                              f"{f' | BS: [{plate_text}]' if plate_text else ' | BS: (chưa đọc được)'}")

                    # ── Bước 5: Cập nhật biển số muộn cho các vi phạm đã ghi nhận trước đó ──
                    if cfg.check_plate and (is_rl or is_hm) and plate_text and track_id:
                        updated = False
                        for v in violations_log:
                            if v["track_id"] == track_id and not v["plate"]:
                                v["plate"] = plate_text
                                updated = True
                                # Thông báo frontend cập nhật biển số
                                if on_plate_update:
                                    on_plate_update({"track_id": track_id, "plate": plate_text,
                                                     "frame": frame_count, "time": round(frame_count/fps,2),
                                                     "bbox": plate_bbox_in_frame, "crop_path": plate_crop_path})
                        if updated:
                            print(f"[BIEN SO]  🔢 Cập nhật biển số muộn ID{track_id}: [{plate_text}]")

                    # ── Bước 6: Vẽ lên frame ──────────────────────────────────
                    if is_rl or is_hm or cfg.show_all:
                        if not plate_text and track_id:
                            plate_text = plate_cache.get(track_id, "")

                        is_violating = is_rl or is_hm
                        col = (0, 0, 255) if is_violating else (0, 255, 0)
                        thickness = 3 if is_violating else 2

                        parts = []
                        if is_rl: parts.append("VUOT DEN DO")
                        if is_hm: parts.append("KHONG MU")

                        draw_lbl = f"ID{track_id} {cls_name}"
                        if parts:
                            draw_lbl += f": {' | '.join(parts)}"
                        if plate_text:
                            draw_lbl += f" [{plate_text}]"

                        draw_box(frame, bbox, draw_lbl, col, thickness)



            # Không dọn dẹp redlight_memory quá gắt để tránh trùng lặp vi phạm khi mất track tạm thời
            # cleanup_violation_memory(redlight_memory, cur_ids)
            draw_light_status(frame, current_light)

            if cfg.display_every_n > 1 and frame_count % cfg.display_every_n != 0:
                continue

            out_frame = frame
            if cfg.display_width and out_frame.shape[1] > cfg.display_width:
                scale = cfg.display_width / out_frame.shape[1]
                out_frame = cv2.resize(
                    out_frame,
                    (cfg.display_width, int(out_frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            stats = {
                "frame": frame_count,
                "total": total,
                "progress": round(frame_count / max(total, 1), 3),
                "light": current_light,
                "violations": len(violations_log),
                "stop_line_ok": stop_line is not None,
                "fps_target": fps,
            }
            on_frame(self._encode(out_frame), stats)

        cap.release()
        if self._stopped:
            print("[PROCESSOR] ⏹️ Đã dừng theo yêu cầu người dùng.")
            return
        summary = {
            "total_frames": frame_count,
            "violations": violations_log,
            "redlight_count": sum(1 for v in violations_log if v["type"] == "Vượt đèn đỏ"),
            "helmet_count": len(helmet_ids),
        }
        if on_done: on_done(summary)
