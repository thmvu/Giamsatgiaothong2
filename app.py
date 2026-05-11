"""
 HỆ THỐNG AI GIÁM SÁT GIAO THÔNG (One-time Initialization + YOLO-BBox + SAM)
=====================================
Kiến trúc "Ma Thuật" — Promptable Segmentation:
  - Giai đoạn 1 (Instant Calibration - Frame đầu tiên):
    1. Detect đèn giao thông (models/phathienden.pt) → lấy vị trí đèn.
    2. YOLO-BBox (models/vachkeduongbbox1.pt) → tìm BBox vạch dừng gần đèn nhất.
    3. SAM (models/sam2_b.pt) → nhận BBox, segment chính xác từng milimet vạch sơn.
    4. Fallback: OpenCV (adaptiveThreshold + contour) nếu YOLO/SAM fail.
    → Lưu Polygon vĩnh viễn (STATIC_STOP_POLYGON).
    → Giải phóng YOLO-BBox + SAM khỏi bộ nhớ (chỉ chạy 1 lần duy nhất!).
  - Giai đoạn 2 (Real-time Detection — siêu nhanh):
      + Model 1: models/phathienden.pt  → Phát hiện đèn giao thông, xác thực HSV chống nhiễu.
      + Model 2: models/yolo11m.pt      → Phát hiện phương tiện (car/motorcycle/bus/truck)
      + Model 3: models/phathienmu.pt   → Kiểm tra mũ bảo hiểm (xe máy)
      (KHÔNG chạy SAM hay YOLO-vạch ở giai đoạn này → tối ưu hiệu năng!)

Logic vi phạm:
  - Đèn ĐỎ (xác thực qua HSV) + xe đè lên STATIC_STOP_POLYGON -> VI PHẠM vượt đèn đỏ
  - Xe máy + Without Helmet → VI PHẠM không đội mũ
"""

import streamlit as st
import cv2
import tempfile
import os
import csv
import json
import numpy as np
import gc
import torch
from datetime import datetime
from ultralytics import YOLO, SAM, FastSAM

# Import utils
from utils.drawing import draw_box, draw_stop_line, draw_light_status, draw_polygon
from utils.violation import (
    check_redlight_violation, save_violation_evidence,
    get_stop_line_from_polygon, draw_stop_line_on_frame,
    cleanup_violation_memory,
    # legacy
    has_crossed_line, is_below_line, is_overlapping_polygon
)

# Import License Plate pipeline
from core.plate_reader import LicensePlateDetector, PlateReader

# === COCO CLASS IDs (yolo11m.pt) ===
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
MOTORBIKE_CLASS = 3

VN_NAMES = {2: "Ô tô", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải"}

# ===== GIAO DIỆN =====
st.set_page_config(page_title="AI Traffic Monitor", page_icon="🚦", layout="wide")
st.title("🚦 Hệ thống AI Giám sát Giao thông (ITS Pro)")
st.caption("One-time Calibration (YOLO-BBox + SAM) + Real-time Detection")


# ===== DEVICE SETUP =====
# SAM2 (~160MB weights) → CPU để tránh OOM với VRAM 2GB
# Các model YOLO nhỏ hơn → GPU nếu có
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_DEVICE = "cpu"   # SAM2 luôn chạy CPU để tiết kiệm VRAM
print(f"🖥️ Inference device: YOLO={DEVICE.upper()}, SAM={SAM_DEVICE.upper()}")

# ===== LOAD MODEL (cached) =====
@st.cache_resource
def load_model(path):
    model = YOLO(path)
    model.to(DEVICE)
    return model

# Model đèn giao thông (chỉ detect đèn đỏ/xanh/vàng)
light_model = load_model("models/phathienden.pt")          # Model đèn (chỉ đèn)
vehicle_model = load_model("models/yolo11m.pt")            # Model xe: COCO
helmet_model = load_model("models/phathienmu1.pt")         # Model mũ (v2)

# === CALIBRATION MODELS (sẽ giải phóng sau frame đầu tiên) ===
# Không dùng @st.cache_resource vì sẽ xóa khỏi RAM sau khi calibrate xong
@st.cache_resource
def load_calibration_models():
    """Load YOLO-BBox vạch (GPU) + SAM2 (CPU) cho calibration. Sẽ giải phóng sau."""
    yolo_line = YOLO("models/vachkeduongbbox1.pt")   # YOLO detect BBox vạch → GPU
    yolo_line.to(DEVICE)
    sam2 = SAM("models/sam2_b.pt")                  # SAM2 → CPU (tiết kiệm VRAM)
    sam2.to(SAM_DEVICE)
    return yolo_line, sam2

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Cài đặt")
conf_light = st.sidebar.slider("Confidence: Đèn", 0.1, 0.9, 0.5, 0.05)
conf_vehicle = st.sidebar.slider("Confidence: Xe", 0.1, 0.9, 0.4, 0.05)
conf_helmet = st.sidebar.slider("Confidence: Mũ", 0.1, 0.9, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🚧 Cài đặt Vạch Dừng (SAM)")
st.sidebar.caption("Điều chỉnh cho giai đoạn Calibration (Frame 1)")
conf_stop_line = st.sidebar.slider("Confidence: Vạch dừng (YOLO-BBox)", 0.1, 0.9, 0.3, 0.05,
    help="Ngưỡng tin cậy cho model detect vạch kẻ đường")
min_width_pct = st.sidebar.slider(
    "📐 Min width vạch (% frame)", 1, 40, 8, 1,
    help="Vạch phải rộng ít nhất X% chiều ngang frame mới hợp lệ.\n"
         "Video cũ (640px): ~20% | Video mới HD (1280px+): hạ xuống 5-10%\n"
         "Mặc định 8% phù hợp cho cả hai loại video."
)
stop_line_extend_left = st.sidebar.slider("↔️ Mở rộng trái (px)", 0, 500, 150, 10,
    help="Kéo dài vạch dừng sang bên trái thêm bao nhiêu pixel")
stop_line_offset_up = st.sidebar.slider("⬆️ Dịch vạch lên (px)", 0, 100, 30, 5,
    help="Dịch ngưỡng phát hiện lên trên N pixel so với cạnh trên của vạch, tránh false-positive")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Tính năng")
check_redlight = st.sidebar.checkbox("🔴 Phát hiện vượt đèn đỏ", value=True)
check_helmet_enabled = st.sidebar.checkbox("🪖 Kiểm tra mũ bảo hiểm", value=True)
check_plate_enabled = st.sidebar.checkbox("🔢 Nhận dạng biển số (OCR)", value=True)

if check_plate_enabled:
    st.sidebar.markdown(" ")
    st.sidebar.caption("🔍 Chỉnh độ nhạy biển số — giảm nếu mất nhiều biển")
    conf_lp  = st.sidebar.slider(
        "YOLO Biển số — conf", 0.05, 0.50, 0.20, 0.05,
        help="Ngưỡng YOLO phát hiện vùng biển số trong ảnh xe.\nGiảm = nhạy hơn nhưng dễ false-positive\nTăng = chất lượng hơn nhưng có thể bỏ sót biển nhỏ"
    )
    conf_ocr = st.sidebar.slider(
        "OCR Score — ngưỡng chấp nhận", 0.30, 0.90, 0.50, 0.05,
        help="Chỉ giữ kết quả OCR có score cao hơn ngưỡng này.\nGiảm = chấp nhận cả kết quả mờ\nTăng = chỉ giữ kết quả rất chắc chắn"
    )
else:
    conf_lp  = 0.20
    conf_ocr = 0.50

st.sidebar.markdown("---")
st.sidebar.header("⚡ Hiệu năng")
process_every_n = st.sidebar.slider("Xử lý mỗi N frame (AI)", 1, 5, 1,
    help="AI chạy mỗi N frame. Tăng lên = nhanh hơn nhưng bỏ sót xe nhanh")
traffic_interval = st.sidebar.slider("Check đèn mỗi N frame", 1, 10, 3)
display_every_n = st.sidebar.slider("🖥️ Cập nhật UI mỗi N frame", 1, 10, 2,
    help="Chỉ vẽ lên màn hình mỗi N frame → giảm lag UI đáng kể.\n"
         "Tăng cao = nhanh hơn nhưng video giật hơn.\n"
         "Gợi ý: 2-3 cho video thường, 5-10 khi muốn tốc độ tối đa")
display_width = st.sidebar.select_slider(
    "📐 Độ rộng hiển thị (px)", options=[320, 480, 640, 800, 960, 1280], value=640,
    help="Resize frame trước khi gửi lên UI → encode nhanh hơn.\n"
         "Nhỏ hơn = nhanh hơn nhiều. Chất lượng AI không bị ảnh hưởng (AI chạy trên frame gốc)."
)
show_all = st.sidebar.checkbox("👁️ Hiện tất cả xe", value=True)

# ===== BIỂN SỐ — YOLO LP Detector + RapidOCR =====
# Thêm _v2 vào tên hàm để force Streamlit reload cache khi code thay đổi
@st.cache_resource
def load_plate_models_v2():
    """
    Load YOLO biển số + RapidOCR (PaddleOCR model qua ONNX).
    Cache vĩnh viễn trong session (chỉ load 1 lần).
    Nếu bị lỗi stale cache: dừng app → chạy lại run.bat.
    """
    lp_det = LicensePlateDetector("models/license_plate_detector.pt", conf=0.10, device=DEVICE)
    lp_ocr = PlateReader(use_gpu=True)
    return lp_det, lp_ocr

if check_plate_enabled:
    lp_detector, plate_reader_ocr = load_plate_models_v2()
else:
    lp_detector, plate_reader_ocr = None, None

# ===== HÀM HỖ TRỢ =====
# Hàm OpenCV Fallback đã được gỡ bỏ theo yêu cầu


def extract_stop_polygon_yolo(frame, yolo_line_model, sam2_model,
                              light_bbox=None, conf=0.05, extend_left=100,
                              min_width_pct: float = 8.0):
    """
    🎯 YOLO-BBox → SAM2 (với Y-clip) → Perspective-aware Polygon:
      1. Cắt ROI 60% dưới frame + CLAHE + YOLO predict.
      2. Lọc bbox: aspect ≥ 4 VÀ width ≥ min_width_pct% frame width.
         (min_width_pct mặc định 8% — phù hợp cả video 640p lẫn 1080p+)
      3. SAM2 nhận BBox tốt nhất → segment mask.
      4. CLIP mask vào đúng Y-strip của YOLO bbox → tránh flood-fill toàn mặt đường.
      5. Fit đường thẳng qua contour của mask → perspective polygon chính xác.
    """
    h_frame, w_frame = frame.shape[:2]

    # ── Bước 1: Tự động cắt ROI ──
    roi_top = int(h_frame * 0.4)
    roi_frame = frame[roi_top:, :]
    roi_h, roi_w = roi_frame.shape[:2]
    print(f"📌 ROI: cắt từ y={roi_top} xuống, kích thước={roi_w}x{roi_h}")
    print(f"📐 Filter vạch: aspect≥4, width≥{min_width_pct:.0f}% ({roi_w * min_width_pct / 100:.0f}px)")

    # ── Bước 2: CLAHE ──
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    roi_enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    # ── Bước 3: YOLO-BBox predict ──
    # imgsz=1024 phù hợp cho cả video nhỏ (640p) lẫn video lớn (1080p+)
    results = yolo_line_model.predict(roi_enhanced, conf=conf, imgsz=1024, verbose=False)

    raw_bboxes = []  # (full_frame_bbox, conf_val, bw)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
            conf_val = float(r.boxes.conf[i])
            bw = bx2 - bx1
            bh = max(by2 - by1, 1)
            aspect = bw / bh

            if aspect < 4.0:
                print(f"  ⏩ Bỏ box {i}: aspect={aspect:.1f} < 4.0")
                continue
            min_width_px = roi_w * (min_width_pct / 100.0)
            if bw < min_width_px:
                print(f"  ⏩ Bỏ box {i}: width={bw:.0f}px < {min_width_pct:.0f}% frame ({min_width_px:.0f}px)")
                continue

            fy1, fy2 = by1 + roi_top, by2 + roi_top
            raw_bboxes.append(([bx1, fy1, bx2, fy2], conf_val, bw))
            print(f"  ✅ Box {i}: conf={conf_val:.3f}, aspect={aspect:.1f}, "
                  f"width={bw:.0f}px ({bw/roi_w*100:.0f}% frame), "
                  f"full=[{bx1:.0f},{fy1:.0f},{bx2:.0f},{fy2:.0f}]")

    if not raw_bboxes:
        print(f"⚠️ Không có vạch nào qua bộ lọc (aspect≥4 + width≥{min_width_pct:.0f}%). "
              f"Thử giảm Confidence hoặc giảm 'Min width vạch' trong sidebar.")
        return None, None

    print(f"✅ {len(raw_bboxes)} vạch hợp lệ sau filter")

    # ── Bước 4: Chọn 1 bbox TỐT NHẤT (đơn giản, đáng tin) ──
    def score_bbox(item):
        bb, c, bw_val = item
        bb_cx = (bb[0] + bb[2]) / 2
        bb_cy = (bb[1] + bb[3]) / 2
        width_score = bw_val / w_frame   # rộng hơn → tốt hơn

        if light_bbox is not None:
            lx, ly, lx2, ly2 = light_bbox
            if bb_cy <= ly2:             # trên đèn → loại
                return -1.0
            dist = ((bb_cx - (lx+lx2)/2)**2 + (bb_cy - ly2)**2) ** 0.5
            dist_score = 1.0 - min(dist / ((w_frame**2+h_frame**2)**0.5), 1.0)
        else:
            dist_score = 0.5

        return 0.6 * width_score + 0.4 * dist_score

    best = max(raw_bboxes, key=score_bbox)
    if score_bbox(best) < 0:
        print("⚠️ Tất cả bbox đều nằm TRÊN đèn → không hợp lệ.")
        return None, None

    best_bb = best[0]   # [x1, y1, x2, y2] full-frame coords
    bx1, by1, bx2, by2 = map(int, best_bb)
    # ── Bước 5: SAM2 nhận đúng bbox đó → segment ──
    # Prompt = chính xác bbox YOLO detect được (không mở rộng X to đùng để tránh SAM2 ngáo)
    # Y-clip sau để ngăn flood-fill xuống đường
    try:
        sam_results = sam2_model.predict(frame, bboxes=[best_bb], verbose=False, device=SAM_DEVICE)

        if sam_results and sam_results[0].masks is not None \
                and len(sam_results[0].masks.data) > 0:
            mask = sam_results[0].masks.data[0].cpu().numpy()
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Resize về đúng kích thước frame
            if mask_uint8.shape[:2] != (h_frame, w_frame):
                mask_uint8 = cv2.resize(mask_uint8, (w_frame, h_frame),
                                        interpolation=cv2.INTER_NEAREST)

            # ★ STRICT CLIP: Chỉ giữ pixel trong dải Y của YOLO bbox (không padding)
            # → SAM2 chỉ được lan NGANG, không được lan DỌC ra ngoài vùng vạch
            clip_mask = np.zeros_like(mask_uint8)
            yc_top = max(by1, 0)
            yc_bot = min(by2, h_frame)
            # Clip thêm X: chỉ lấy phần mask nằm trong bbox X (mở rộng thêm extend_left phía trái)
            xc_left = max(bx1 - extend_left, 0)
            xc_right = min(bx2, w_frame)
            clip_mask[yc_top:yc_bot, xc_left:xc_right] = mask_uint8[yc_top:yc_bot, xc_left:xc_right]
            print(f"✂️ Strict clip mask: y=[{yc_top},{yc_bot}], x=[{xc_left},{xc_right}]")

            contours, _ = cv2.findContours(clip_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                epsilon = 0.008 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)
                poly = np.array(approx.reshape(-1, 2), np.int32)
                
                # Kéo các điểm nửa bên trái sang trái thêm extend_left px (từ sidebar)
                for pt in poly:
                    if pt[0] < (bx1 + bx2) / 2:
                        pt[0] = max(0, pt[0] - extend_left)
                
                stop_y = int(max(p[1] for p in poly))
                print(f"✅ SAM2 Polygon: {len(poly)} điểm, area={area:.0f}px², stop_y={stop_y}")
                return poly, stop_y
            else:
                print("⚠️ SAM2: Mask sau Y-clip không có contour.")
        else:
            print("⚠️ SAM2: Không tạo được mask.")
    except Exception as e:
        print(f"❌ SAM2 Error: {e}")

    # ── Fallback: YOLO bbox → rectangle polygon trực tiếp ──
    print("⚠️ Fallback → dùng YOLO bbox làm rectangle polygon")
    # Lan sang trái thêm 100px ở fallback
    bx1_ext = max(0, bx1 - extend_left)
    poly = np.array([[bx1_ext,by1],[bx2,by1],[bx2,by2],[bx1_ext,by2]], np.int32)
    stop_y = by2
    print(f"✅ Fallback polygon: [{bx1_ext},{by1},{bx2},{by2}], stop_y={stop_y}")
    return poly, stop_y


def check_red_light_hsv(frame, bbox):
    """ Kiểm tra xem hộp đèn có pixel màu đỏ ở nửa trên không. """
    x1, y1, x2, y2 = bbox
    # Cắt nửa trên của đèn
    mid_y = y1 + (y2 - y1) // 2
    crop = frame[y1:mid_y, x1:x2]
    
    if crop.size == 0: return False
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # 2 Dải màu đỏ trong HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    red_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    if total_pixels == 0: return False
    ratio = red_pixels / total_pixels
    return ratio > 0.05 # Ngưỡng 5% pixel đỏ

# ===== UPLOAD VIDEO =====
uploaded_file = st.file_uploader("📁 Tải video giao thông", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"📹 {uploaded_file.name} | {vid_w}x{vid_h} | {fps:.1f} FPS | "
            f"{total_frames} frames ({total_frames/max(fps,1):.1f}s)")

    evidence_dir = os.path.join(os.path.dirname(__file__), "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    if st.button("🚀 Bắt đầu Quét", type="primary", use_container_width=True):

        # --- Trạng thái ---
        current_light = "unknown"
        global_stop_polygon = None
        global_stop_line_pts = None  # ((x1,y1),(x2,y2)) — đoạn thẳng stop line ngang

        redlight_memory = {}         # {track_id: {'saved': bool}} — proximity violation
        helmet_violated_ids = set()  # Sổ đen mũ
        plate_cache = {}
        violations_log = []

        # --- UI ---
        col_vid, col_stat = st.columns([3, 1])
        with col_vid:
            st_frame = st.empty()
            st_calib_bg = st.empty()
        with col_stat:
            stats_ph = st.empty()
        progress = st.progress(0, text="Đang khởi tạo...")

        frame_count = 0
        processed_count = 0

        # =============================================
        # FRAME 1: INSTANT CALIBRATION (như repo GitHub)
        # Không dùng MOG2, không mất frame nào!
        # =============================================
        progress.progress(0.2, text="🔍 Đang phân tích frame đầu tiên...")
        
        ret, first_frame = cap.read()
        if not ret:
            st.error("❌ Không đọc được video!")
            st.stop()
        
        frame_count = 1
        st_frame.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Bước 1: Detect đèn giao thông trên frame đầu → lấy vị trí đèn
        progress.progress(0.4, text="🚦 Đang tìm đèn giao thông...")
        light_bbox_for_calib = None
        light_results = light_model(first_frame, conf=0.3, verbose=False)
        for r in light_results:
            if r.boxes is not None and len(r.boxes) > 0:
                best_c = 0
                for i in range(len(r.boxes)):
                    c = float(r.boxes.conf[i])
                    if c > best_c:
                        best_c = c
                        light_bbox_for_calib = r.boxes.xyxy[i].tolist()
        
        if light_bbox_for_calib:
            lx1, ly1, lx2, ly2 = map(int, light_bbox_for_calib)
            print(f"🚦 Đèn giao thông tìm thấy tại [{lx1},{ly1},{lx2},{ly2}]")
        else:
            print("⚠️ Không tìm thấy đèn trên frame 1 — vẫn tiếp tục detect vạch...")
        
        # =============================================
        # Bước 2: 🎯 YOLO-BBox tìm vạch dừng
        # =============================================
        progress.progress(0.5, text="🎯 Đang load YOLO-BBox + SAM2...")

        # Load calibration models (YOLO-BBox + SAM2)
        yolo_line_model, sam2_model = load_calibration_models()

        progress.progress(0.7, text="🎯 YOLO-BBox đang detect vạch dừng...")
        calib_method = ""

        # === Thử tối đa MAX_CALIB_FRAMES frame đầu tiên ===
        MAX_CALIB_FRAMES = 5
        calib_frames = [first_frame]

        for _ in range(MAX_CALIB_FRAMES - 1):
            ret_c, f_c = cap.read()
            if not ret_c:
                break
            calib_frames.append(f_c)
            frame_count += 1

        for attempt, calib_frame in enumerate(calib_frames):
            progress.progress(0.6 + 0.1 * attempt / MAX_CALIB_FRAMES,
                              text=f"🎯 Đang thử frame {attempt + 1}/{len(calib_frames)}...")
            print(f"\n🔄 Calibration attempt {attempt + 1}/{len(calib_frames)}...")

            result = extract_stop_polygon_yolo(
                calib_frame, yolo_line_model, sam2_model,
                light_bbox=light_bbox_for_calib, conf=conf_stop_line,
                extend_left=stop_line_extend_left,
                min_width_pct=min_width_pct,
            )

            if result[0] is not None:
                global_stop_polygon, stop_y = result
                calib_method = f"YOLO-BBox (frame {attempt + 1})"
                # Cập nhật first_frame hiển thị bằng frame tìm thấy vạch
                first_frame = calib_frame
                print(f"✅ Calibration thành công ở frame {attempt + 1}!")
                break

        if global_stop_polygon is None:
            print("❌ Không tìm thấy vạch sau khi thử tất cả các frame!")
        
        # === GIẢI PHÓNG CALIBRATION MODELS KHỎI BỘ NHỚ ===
        progress.progress(0.9, text="🧹 Giải phóng YOLO-BBox + SAM2 khỏi RAM/GPU...")
        load_calibration_models.clear()
        del yolo_line_model, sam2_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Đã giải phóng YOLO-BBox + SAM2 khỏi bộ nhớ!")
        
        # === HIỂN THỊ KẾT QUẢ CALIBRATION ===
        if global_stop_polygon is not None:
            # Vẽ minh họa
            viz = first_frame.copy()
            draw_polygon(viz, global_stop_polygon, (255, 0, 255), 2, alpha=0.4)
            min_y = min(p[1] for p in global_stop_polygon)
            min_x = min(p[0] for p in global_stop_polygon)
            cv2.putText(viz, f"STOP LINE (y={stop_y}) [{calib_method}]",
                        (int(min_x), int(min_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            if light_bbox_for_calib:
                lx1, ly1, lx2, ly2 = map(int, light_bbox_for_calib)
                cv2.rectangle(viz, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
                cv2.putText(viz, "LIGHT", (lx1, ly1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            st_calib_bg.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB),
                              caption=f"✅ Frame 1: Vạch dừng ({calib_method}) + Đèn", width=400)
        else:
            st.warning("⚠️ Không tìm thấy vạch dừng (SAM + OpenCV đều fail). "
                       "Hệ thống vẫn chạy nhưng không detect vượt đèn đỏ.")
            st_calib_bg.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                              caption="Frame 1 — không phát hiện vạch dừng", width=400)
        
        progress.progress(0, text="✅ Calibration xong! SAM đã giải phóng. Bắt đầu theo dõi...")

        # Tính stop line ngang từ polygon (nâng lên offset_up px)
        global_stop_line_pts = None
        if global_stop_polygon is not None:
            global_stop_line_pts = get_stop_line_from_polygon(
                global_stop_polygon, offset_up=stop_line_offset_up)
            print(f"🛑 global_stop_line_pts = {global_stop_line_pts}")

        # =============================================
        # VÒNG LẶP CHÍNH — REAL-TIME DETECTION
        # Bắt đầu từ frame 2 (chỉ mất 1 frame!)
        # =============================================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % process_every_n != 0: continue
            processed_count += 1
            progress.progress(min(frame_count / total_frames, 1.0), text=f"Frame {frame_count}/{total_frames}")

            # --- NHẬN DIỆN ĐÈN GIAO THÔNG (Mỗi N frame) ---
            if processed_count % traffic_interval == 1 or traffic_interval == 1:
                light_results = light_model(frame, conf=conf_light, verbose=False)
                
                best_light_conf = 0
                temp_light = "unknown"

                for r in light_results:
                    if r.boxes is None or len(r.boxes) == 0: continue
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i])
                        name = light_model.names[cls_id]
                            
                        conf_val = float(r.boxes.conf[i])
                        bbox_xyxy = r.boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = map(int, bbox_xyxy)

                        if conf_val > best_light_conf:
                            best_light_conf = conf_val
                            
                            # Xác thực bằng HSV Crop & Classify
                            is_red = check_red_light_hsv(frame, (x1, y1, x2, y2))
                            
                            if is_red:
                                temp_light = "red"
                            elif "green" in name.lower() or name == "xanh":
                                temp_light = "green"
                            elif "yellow" in name.lower() or name == "vang":
                                temp_light = "yellow"
                            else:
                                temp_light = "unknown"
                            
                            color = (0, 0, 255) if is_red else (0, 255, 0) if temp_light == "green" else (0, 255, 255)
                            draw_box(frame, [x1, y1, x2, y2], f"Light ({temp_light}) {conf_val:.2f}", color)
                            
                if temp_light != "unknown":
                    if temp_light != current_light:
                        print(f"🚦 Đèn đổi: {current_light} → {temp_light} (frame {frame_count})")
                    current_light = temp_light

            # --- NHẬN DIỆN PHƯƠNG TIỆN + TRACKING ---
            vehicle_results = vehicle_model.track(
                frame, conf=conf_vehicle, classes=VEHICLE_CLASSES,
                persist=True, tracker="bytetrack.yaml", verbose=False
            )

            for r in vehicle_results:
                if r.boxes is None or len(r.boxes) == 0: continue

                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].tolist()
                    x1, y1, x2, y2 = map(int, bbox)
                    cls_id = int(r.boxes.cls[i])
                    conf_val = float(r.boxes.conf[i])
                    class_name = vehicle_model.names[cls_id]
                    track_id = int(r.boxes.id[i]) if r.boxes.id is not None else None

                    is_redlight_vio = False
                    is_helmet_vio = False
                    plate_text = ""

                    # --- KIỂM TRA VƯỢT ĐÈN ĐỎ ---
                    # Logic proximity + side:
                    #   dist(bánh xe, stop_line) < threshold  VÀ  bánh xe đã qua vạch (dy>0)
                    #   → VI PHẠM (ghi nhận 1 lần/xe, không cần prev_bbox)
                    if check_redlight and global_stop_line_pts is not None and track_id is not None:
                        is_redlight_vio = check_redlight_violation(
                            track_id, bbox, global_stop_line_pts,
                            current_light, redlight_memory, threshold=40
                        )
                        if is_redlight_vio:
                            ev_path = save_violation_evidence(
                                track_id, frame, frame_count,
                                x1, y1, x2, y2,
                                global_stop_line_pts, redlight_memory,
                                evidence_dir, fps=fps,
                                cls_name=VN_NAMES.get(cls_id, class_name)
                            )
                            if ev_path:   # Mới lưu lần đầu → thêm vào log
                                violations_log.append({
                                    "time": round(frame_count / max(fps, 1), 2),
                                    "frame": frame_count, "type": "Vượt đèn đỏ",
                                    "vehicle": VN_NAMES.get(cls_id, class_name),
                                    "track_id": track_id, "plate": "", "evidence": ev_path
                                })

                    # --- KIỂM TRA MŨ BẢO HIỂM ---
                    if check_helmet_enabled and cls_id == MOTORBIKE_CLASS:
                        if track_id is not None and track_id in helmet_violated_ids:
                            is_helmet_vio = True
                        else:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0 and min(crop.shape[:2]) > 15:
                                h_res = helmet_model.predict(crop, conf=conf_helmet, verbose=False)
                                for hr in h_res:
                                    # Debug: in tên các class model detect được (chỉ in 1 lần đầu)
                                    if frame_count <= 30 and processed_count <= 5:
                                        detected_cls = [f"{hr.names[int(b.cls[0])]}({float(b.conf[0]):.2f})" for b in hr.boxes]
                                        if detected_cls:
                                            print(f"[HELMET-DBG] frame#{frame_count} ID{track_id}: {detected_cls}")
                                        else:
                                            print(f"[HELMET-DBG] frame#{frame_count} ID{track_id}: không detect được class nào (crop={crop.shape[1]}x{crop.shape[0]}px)")
                                    for b in hr.boxes:
                                        cls_name_h = hr.names[int(b.cls[0])]
                                        # Hỗ trợ nhiều tên class phổ biến
                                        if cls_name_h.lower() in ('without helmet', 'without_helmet', 'no_helmet', 'nohelmet', 'no helmet'):
                                            is_helmet_vio = True
                                            if track_id is not None:
                                                helmet_violated_ids.add(track_id)
                                            ev = os.path.join(evidence_dir, f"helmet_ID{track_id}_f{frame_count}.jpg")
                                            cv2.imwrite(ev, frame)
                                            if track_id not in [v['track_id'] for v in violations_log if v['type'] == 'Không đội mũ']:
                                                violations_log.append({
                                                    "time": round(frame_count / max(fps,1), 2),
                                                    "frame": frame_count, "type": "Không đội mũ",
                                                    "vehicle": VN_NAMES.get(cls_id, class_name),
                                                    "track_id": track_id, "plate": "", "evidence": ev
                                                })
                                            print(f"[HELMET-VIO] frame#{frame_count} ID{track_id}: '{cls_name_h}' → VI PHẠM KHÔNG ĐỘI MŨ!")
                                            break
                                    if is_helmet_vio: break

                    # --- BIỂN SỐ OCR ("Khoe Khéo": chỉ kích hoạt khi xe VI PHẠM!) ---
                    # Logic: YOLO detect xe → vi phạm? → YOLO detect biển số → crop → PaddleOCR
                    # Không OCR tất cả xe → tiết kiệm tài nguyên GPU đáng kể!
                    if check_plate_enabled and lp_detector and plate_reader_ocr and track_id is not None:
                        # Sync conf từ sidebar slider (thủ công) → YOLO LP detector
                        lp_detector.conf = conf_lp
                        # Bước 1: Lấy từ cache trước (không bao giờ OCR lại cùng xe)
                        cached_plate = plate_reader_ocr.get_cached_plate(track_id)
                        if cached_plate:
                            plate_text = cached_plate
                        elif is_redlight_vio or is_helmet_vio:
                            # Bước 2: CHỈ chạy khi xe đang vi phạm!
                            vehicle_crop = frame[y1:y2, x1:x2]
                            if vehicle_crop.size > 0 and min(vehicle_crop.shape[:2]) > 20:
                                # Bước 2a: YOLO license_plate_detector.pt → detect bbox biển số
                                plate_crop_img, plate_bbox = lp_detector.crop_best_plate(vehicle_crop)
                                if plate_crop_img is not None and plate_crop_img.size > 0:
                                    ph, pw = plate_crop_img.shape[:2]
                                    print(f"[LP-DETECT] ID{track_id} frame#{frame_count}: Tìm thấy biển số → crop={pw}x{ph}px")
                                    # Bước 2b: RapidOCR → đọc text biển số
                                    plate_text = plate_reader_ocr.read_plate(plate_crop_img, track_id, min_score=conf_ocr)
                                    if plate_text:
                                        print(f"[OCR-OK]    ID{track_id}: ✅ Biển số = '{plate_text}'")
                                    else:
                                        print(f"[OCR-FAIL]  ID{track_id}: ❌ Không đọc được text (biển mờ/nhỏ/góc xấu)")
                                else:
                                    print(f"[LP-DETECT] ID{track_id} frame#{frame_count}: Không thấy biển số → fallback crop 35%")
                                    # Fallback: crop thô phần dưới 35% xe nếu YOLO LP không thấy biển
                                    fallback = vehicle_crop[int(vehicle_crop.shape[0] * 0.65):, :]
                                    if fallback.size > 0:
                                        plate_text = plate_reader_ocr.read_plate(fallback, track_id, min_score=conf_ocr)
                                        if plate_text:
                                            print(f"[OCR-OK]    ID{track_id}: ✅ Biển số (fallback) = '{plate_text}'")
                                        else:
                                            print(f"[OCR-FAIL]  ID{track_id}: ❌ Fallback cũng không đọc được")
                            # Bước 3: Cập nhật cache + violations_log
                            if plate_text:
                                plate_cache[track_id] = plate_text
                                for v in violations_log:
                                    if v["track_id"] == track_id and not v["plate"]:
                                        v["plate"] = plate_text

                    # --- VẼ KẾT QUẢ ---
                    has_violation = is_redlight_vio or is_helmet_vio

                    if has_violation:
                        vio_parts = []
                        if is_redlight_vio: vio_parts.append("VUOT DEN DO")
                        if is_helmet_vio: vio_parts.append("KHONG MU")
                        label = f"ID{track_id} {class_name}: {' | '.join(vio_parts)}"
                        if plate_text: label += f" [{plate_text}]"
                        draw_box(frame, bbox, label, (0, 0, 255), 3)
                    # Xe bình thường: KHÔNG vẽ — video sạch hơn

            # --- Dọn violation_memory cho xe không còn track ---
            current_ids = set()
            for r in vehicle_results:
                if r.boxes is not None and r.boxes.id is not None:
                    for tid in r.boxes.id.tolist():
                        current_ids.add(int(tid))
            cleanup_violation_memory(redlight_memory, current_ids)

            # --- VẼ HUD (KHÔNG vẽ stop line trên video live) ---
            draw_light_status(frame, current_light)

            # Hiển thị — chỉ cập nhật UI mỗi display_every_n frame để giảm lag
            if frame_count % display_every_n == 0:
                disp = frame
                # Resize nhỏ lại nếu cần → encode PNG nhanh hơn rất nhiều
                if disp.shape[1] > display_width:
                    scale = display_width / disp.shape[1]
                    disp = cv2.resize(disp, (display_width, int(disp.shape[0] * scale)),
                                      interpolation=cv2.INTER_LINEAR)
                st_frame.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

            plates_found = sum(1 for v in plate_reader_ocr.cache.values() if v) if plate_reader_ocr else len([v for v in plate_cache.values() if v])
            light_vn = {"red":"ĐỎ","green":"XANH","yellow":"VÀNG"}.get(current_light,"--")
            stop_status = "✅ Cố định" if global_stop_polygon is not None else "⏳ Chờ..."
            stats_ph.markdown(f"""
### 📊 Live
| | |
|--|--|
| Frame | **{frame_count}/{total_frames}** |
| 🚦 Đèn | **{light_vn}** |
| 🛑 Vạch | **{stop_status}** |
| 🔴 VP Đèn | **{len(redlight_memory)}** |
| 🪖 VP Mũ | **{len(helmet_violated_ids)}** |
| 🔢 Biển số | **{plates_found}** |
            """)

        # =============================================
        # KẾT THÚC — TỔNG KẾT
        # =============================================
        cap.release()
        progress.progress(1.0, text="✅ Hoàn tất!")
        st.markdown("---")
        st.header("📊 Tổng kết")

        # Đếm từ violations_log (chính xác, không bị ảnh hưởng bởi cleanup redlight_memory)
        redlight_count = sum(1 for v in violations_log if v["type"] == "Vượt đèn đỏ")
        helmet_count   = len(helmet_violated_ids)
        total_vio      = redlight_count + helmet_count
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🔴 Vượt đèn đỏ", redlight_count)
        with c2: st.metric("🪖 Không mũ",    helmet_count)
        with c3: st.metric("⚠️ Tổng VP",     total_vio)

        if total_vio > 0: st.success(f"✅ Phát hiện **{total_vio}** vi phạm!")
        else: st.info("🎉 Không phát hiện vi phạm!")

        for v in violations_log:
            tid = v["track_id"]
            # Ưu tiên PaddleOCR cache, fallback về plate_cache
            if not v["plate"] and plate_reader_ocr:
                cached = plate_reader_ocr.get_cached_plate(tid)
                if cached:
                    v["plate"] = cached
            if not v["plate"] and tid in plate_cache and plate_cache[tid]:
                v["plate"] = plate_cache[tid]

        if violations_log:
            st.subheader("📋 Bảng Vi phạm")
            st.dataframe([{
                "Thời gian (s)": v["time"], "Loại VP": v["type"],
                "Phương tiện": v["vehicle"], "ID": v["track_id"],
                "Biển số": v["plate"] or "N/A", "Frame": v["frame"]
            } for v in violations_log], use_container_width=True)

            st.subheader("📸 Bằng chứng")
            ncols = min(3, len(violations_log))
            ev_cols = st.columns(ncols)
            for idx, v in enumerate(violations_log):
                if v.get("evidence") and os.path.exists(v["evidence"]):
                    with ev_cols[idx % ncols]:
                        img = cv2.imread(v["evidence"])
                        if img is not None:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                     caption=f"{v['type']} ID{v['track_id']}", use_container_width=True)

            if plate_cache:
                st.subheader("🔢 Biển số đã nhận dạng")
                st.dataframe([{"ID": tid, "Biển số": p or "Không đọc được"} 
                              for tid, p in plate_cache.items()], use_container_width=True)

            st.subheader("📥 Xuất Báo cáo")
            csv_path = os.path.join(evidence_dir, "violations.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=["Thời gian (s)", "Frame", "Loại vi phạm", "Phương tiện", "ID", "Biển số"])
                writer.writeheader()
                for v in violations_log:
                    writer.writerow({
                        "Thời gian (s)": v["time"], "Frame": v["frame"],
                        "Loại vi phạm": v["type"], "Phương tiện": v["vehicle"],
                        "ID": v["track_id"], "Biển số": v["plate"] or "N/A"
                    })

            col_csv, col_json = st.columns(2)
            with col_csv:
                with open(csv_path, 'r', encoding='utf-8-sig') as f:
                    st.download_button("📥 CSV", f.read(), "violations.csv", "text/csv", use_container_width=True)
            with col_json:
                report = {
                    "date": datetime.now().isoformat(), "video": uploaded_file.name,
                    "summary": {"redlight": len(redlight_memory), "helmet": len(helmet_violated_ids), "total": total_vio},
                    "violations": [{k:v for k,v in item.items() if k != "evidence"} for item in violations_log]
                }
                st.download_button("📥 JSON", json.dumps(report, indent=2, ensure_ascii=False),
                    "violations.json", "application/json", use_container_width=True)

else:
    st.markdown("""
    ### 📖 Hướng dẫn
    1. Upload video → 2. Nhấn **Bắt đầu Quét** → 3. Xem kết quả (Calibration tức thì!)

    > 🪄 **Kiến trúc "Ma Thuật" — One-time Initialization:**
    > - **Frame 1**: `models/vachkeduongbbox1.pt` (YOLO-BBox) tìm vạch dừng → `models/sam2_b.pt` (SAM) segment chính xác từng milimet vạch sơn. (Không dùng OpenCV)
    > - **Frame 2+**: SAM + YOLO-vạch được giải phóng khỏi RAM/GPU. Chỉ chạy 3 model nhẹ: đèn, xe, mũ → **Siêu nhanh!**
    > - **HSV Crop & Classify**: Kiểm tra mật độ pixel đỏ nửa trên hộp đèn → Chống nhiễu biển quảng cáo.
    """)