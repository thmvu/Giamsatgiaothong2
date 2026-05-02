"""
 HỆ THỐNG AI GIÁM SÁT GIAO THÔNG (One-time Initialization + YOLO-BBox + SAM)
=====================================
Kiến trúc "Ma Thuật" — Promptable Segmentation:
  - Giai đoạn 1 (Instant Calibration - Frame đầu tiên):
    1. Detect đèn giao thông (phathienden.pt) → lấy vị trí đèn.
    2. YOLO-BBox (vachkeduongbbox.pt) → tìm BBox vạch dừng gần đèn nhất.
    3. SAM (FastSAM-s.pt) → nhận BBox, segment chính xác từng milimet vạch sơn.
    4. Fallback: OpenCV (adaptiveThreshold + contour) nếu YOLO/SAM fail.
    → Lưu Polygon vĩnh viễn (STATIC_STOP_POLYGON).
    → Giải phóng YOLO-BBox + SAM khỏi bộ nhớ (chỉ chạy 1 lần duy nhất!).
  - Giai đoạn 2 (Real-time Detection — siêu nhanh):
      + Model 1: phathienden.pt  → Phát hiện đèn giao thông, xác thực HSV chống nhiễu.
      + Model 2: yolo11m.pt      → Phát hiện phương tiện (car/motorcycle/bus/truck)
      + Model 3: phathienmu.pt   → Kiểm tra mũ bảo hiểm (xe máy)
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
from utils.violation import has_crossed_line, is_below_line, is_overlapping_polygon

# === COCO CLASS IDs (yolo11m.pt) ===
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
MOTORBIKE_CLASS = 3

VN_NAMES = {2: "Ô tô", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải"}

# ===== GIAO DIỆN =====
st.set_page_config(page_title="AI Traffic Monitor", page_icon="🚦", layout="wide")
st.title("🚦 Hệ thống AI Giám sát Giao thông (ITS Pro)")
st.caption("One-time Calibration (YOLO-BBox + SAM) + Real-time Detection")


# ===== LOAD MODEL (cached) =====
@st.cache_resource
def load_model(path):
    return YOLO(path)

# Model đèn giao thông (chỉ detect đèn đỏ/xanh/vàng)
light_model = load_model("phathienden.pt")          # Model đèn (chỉ đèn)
vehicle_model = load_model("yolo11m.pt")            # Model xe: COCO
helmet_model = load_model("phathienmu.pt")          # Model mũ

# === CALIBRATION MODELS (sẽ giải phóng sau frame đầu tiên) ===
# Không dùng @st.cache_resource vì sẽ xóa khỏi RAM sau khi calibrate xong
@st.cache_resource
def load_calibration_models():
    """Load YOLO-BBox vạch + SAM2 cho calibration. Sẽ giải phóng sau."""
    yolo_line = YOLO("vachkeduongbbox.pt")   # YOLO detect BBox vạch kẻ đường
    sam2 = SAM("sam2_b.pt")                  # SAM2 segment chính xác theo góc camera
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
stop_line_extend_left = st.sidebar.slider("↔️ Mở rộng trái (px)", 0, 500, 150, 10,
    help="Kéo dài vạch dừng sang bên trái thêm bao nhiêu pixel")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Tính năng")
check_redlight = st.sidebar.checkbox("🔴 Phát hiện vượt đèn đỏ", value=True)
check_helmet_enabled = st.sidebar.checkbox("🪖 Kiểm tra mũ bảo hiểm", value=True)
check_plate_enabled = st.sidebar.checkbox("🔢 Nhận dạng biển số (OCR)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Hiệu năng")
process_every_n = st.sidebar.slider("Xử lý mỗi N frame", 1, 5, 1)
traffic_interval = st.sidebar.slider("Check đèn mỗi N frame", 1, 10, 3)
show_all = st.sidebar.checkbox("👁️ Hiện tất cả xe", value=True)

# ===== OCR (lazy load) =====
@st.cache_resource
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=True)
    except ImportError:
        return None

ocr_reader = load_ocr() if check_plate_enabled else None

# ===== HÀM HỖ TRỢ =====
# Hàm OpenCV Fallback đã được gỡ bỏ theo yêu cầu


def extract_stop_polygon_yolo(frame, yolo_line_model, sam2_model, light_bbox=None, conf=0.05, extend_left=100):
    """
    🎯 YOLO-BBox → SAM2 (với Y-clip) → Perspective-aware Polygon:
      1. Cắt ROI 60% dưới frame + CLAHE + YOLO predict.
      2. Group detection theo Y-level gần đèn → fit đường thẳng lấy Y-strip.
      3. SAM2 nhận BBox gộp → segment mask.
      4. CLIP mask vào đúng Y-strip của YOLO bbox → tránh flood-fill toàn mặt đường.
      5. Fit đường thẳng qua contour của mask → perspective polygon chính xác.
    """
    h_frame, w_frame = frame.shape[:2]

    # ── Bước 1: Tự động cắt ROI ──
    roi_top = int(h_frame * 0.4)
    roi_frame = frame[roi_top:, :]
    roi_h, roi_w = roi_frame.shape[:2]
    print(f"📌 ROI: cắt từ y={roi_top} xuống, kích thước={roi_w}x{roi_h}")

    # ── Bước 2: CLAHE ──
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    roi_enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    # ── Bước 3: YOLO-BBox predict ──
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
            if bw < roi_w * 0.20:
                print(f"  ⏩ Bỏ box {i}: width={bw:.0f}px < 20% frame")
                continue

            fy1, fy2 = by1 + roi_top, by2 + roi_top
            raw_bboxes.append(([bx1, fy1, bx2, fy2], conf_val, bw))
            print(f"  ✅ Box {i}: conf={conf_val:.3f}, aspect={aspect:.1f}, "
                  f"width={bw:.0f}px ({bw/roi_w*100:.0f}% frame), "
                  f"full=[{bx1:.0f},{fy1:.0f},{bx2:.0f},{fy2:.0f}]")

    if not raw_bboxes:
        print("⚠️ Không có vạch nào qua bộ lọc (aspect≥4 + width≥20%). Thử giảm Confidence.")
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
        sam_results = sam2_model.predict(frame, bboxes=[best_bb], verbose=False)

        if sam_results and sam_results[0].masks is not None \
                and len(sam_results[0].masks.data) > 0:
            mask = sam_results[0].masks.data[0].cpu().numpy()
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Resize về đúng kích thước frame
            if mask_uint8.shape[:2] != (h_frame, w_frame):
                mask_uint8 = cv2.resize(mask_uint8, (w_frame, h_frame),
                                        interpolation=cv2.INTER_NEAREST)

            # ★ Y-CLIP: chỉ giữ pixel trong dải Y của YOLO bbox ±padding
            padding = 15
            clip_mask = np.zeros_like(mask_uint8)
            yc_top = max(by1 - padding, 0)
            yc_bot = min(by2 + padding, h_frame)
            clip_mask[yc_top:yc_bot, :] = mask_uint8[yc_top:yc_bot, :]
            print(f"✂️ Y-clip mask: y=[{yc_top},{yc_bot}]")

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

        violated_ids = set()         # Sổ đen đèn đỏ
        helmet_violated_ids = set()  # Sổ đen mũ
        plate_cache = {}             
        prev_bbox_cache = {}         
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
                extend_left=stop_line_extend_left
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
                    if check_redlight and global_stop_polygon is not None and track_id is not None:
                        if track_id in violated_ids:
                            is_redlight_vio = True
                        elif current_light == "red":
                            # Vi phạm khi điểm đáy bbox (mũi xe / bánh xe) vượt qua stop_y
                            if y2 > stop_y:
                                is_redlight_vio = True
                                violated_ids.add(track_id)
                                ev = os.path.join(evidence_dir, f"redlight_ID{track_id}_f{frame_count}.jpg")
                                cv2.imwrite(ev, frame)
                                violations_log.append({
                                    "time": round(frame_count / max(fps,1), 2),
                                    "frame": frame_count, "type": "Vượt đèn đỏ",
                                    "vehicle": VN_NAMES.get(cls_id, class_name),
                                    "track_id": track_id, "plate": "", "evidence": ev
                                })

                    if track_id is not None:
                        prev_bbox_cache[track_id] = bbox

                    # --- KIỂM TRA MŨ BẢO HIỂM ---
                    if check_helmet_enabled and cls_id == MOTORBIKE_CLASS and track_id is not None:
                        if track_id in helmet_violated_ids:
                            is_helmet_vio = True
                        else:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0 and min(crop.shape[:2]) > 15:
                                h_res = helmet_model.predict(crop, conf=conf_helmet, verbose=False)
                                for hr in h_res:
                                    for b in hr.boxes:
                                        if hr.names[int(b.cls[0])] == 'Without Helmet':
                                            is_helmet_vio = True
                                            helmet_violated_ids.add(track_id)
                                            ev = os.path.join(evidence_dir, f"helmet_ID{track_id}_f{frame_count}.jpg")
                                            cv2.imwrite(ev, frame)
                                            violations_log.append({
                                                "time": round(frame_count / max(fps,1), 2),
                                                "frame": frame_count, "type": "Không đội mũ",
                                                "vehicle": VN_NAMES.get(cls_id, class_name),
                                                "track_id": track_id, "plate": "", "evidence": ev
                                            })
                                            break
                                    if is_helmet_vio: break

                    # --- BIỂN SỐ OCR ---
                    if check_plate_enabled and ocr_reader and track_id is not None:
                        if track_id in plate_cache:
                            plate_text = plate_cache[track_id]
                        else:
                            import re
                            plate_crop = frame[y1 + int((y2-y1)*0.6):y2, x1:x2]
                            if plate_crop.size > 0 and min(plate_crop.shape[:2]) > 15:
                                try:
                                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                    gray = cv2.resize(gray, None, fx=2, fy=2)
                                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    ocr_res = ocr_reader.readtext(thresh, detail=1)
                                    for (_, text, prob) in ocr_res:
                                        cleaned = re.sub(r'[^A-Za-z0-9\-.]', '', text).upper()
                                        if len(cleaned) >= 4 and prob > 0.3:
                                            plate_text = cleaned
                                            break
                                except Exception: pass
                            plate_cache[track_id] = plate_text
                            if plate_text:
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

            # --- VẼ HUD + GLOBAL STOP LINE ---
            # Stop line ẩn trên video (vẫn dùng để check vi phạm)
            # draw_polygon(frame, global_stop_polygon, (255, 0, 255), 2, alpha=0.3)
            # cv2.putText(frame, "STOP LINE", ...)
            draw_light_status(frame, current_light)

            # Hiển thị
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            plates_found = sum(1 for v in plate_cache.values() if v)
            light_vn = {"red":"ĐỎ","green":"XANH","yellow":"VÀNG"}.get(current_light,"--")
            stop_status = "✅ Cố định" if global_stop_polygon is not None else "⏳ Chờ..."
            stats_ph.markdown(f"""
### 📊 Live
| | |
|--|--|
| Frame | **{frame_count}/{total_frames}** |
| 🚦 Đèn | **{light_vn}** |
| 🛑 Vạch | **{stop_status}** |
| 🔴 VP Đèn | **{len(violated_ids)}** |
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

        total_vio = len(violated_ids) + len(helmet_violated_ids)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🔴 Vượt đèn đỏ", len(violated_ids))
        with c2: st.metric("🪖 Không mũ", len(helmet_violated_ids))
        with c3: st.metric("⚠️ Tổng VP", total_vio)

        if total_vio > 0: st.success(f"Phát hiện **{total_vio}** vi phạm!")
        else: st.info("🎉 Không phát hiện vi phạm!")

        for v in violations_log:
            tid = v["track_id"]
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
                    "summary": {"redlight": len(violated_ids), "helmet": len(helmet_violated_ids), "total": total_vio},
                    "violations": [{k:v for k,v in item.items() if k != "evidence"} for item in violations_log]
                }
                st.download_button("📥 JSON", json.dumps(report, indent=2, ensure_ascii=False),
                    "violations.json", "application/json", use_container_width=True)

else:
    st.markdown("""
    ### 📖 Hướng dẫn
    1. Upload video → 2. Nhấn **Bắt đầu Quét** → 3. Xem kết quả (Calibration tức thì!)

    > 🪄 **Kiến trúc "Ma Thuật" — One-time Initialization:**
    > - **Frame 1**: `vachkeduongbbox.pt` (YOLO-BBox) tìm vạch dừng → `FastSAM-s.pt` (SAM) segment chính xác từng milimet vạch sơn. (Không dùng OpenCV)
    > - **Frame 2+**: SAM + YOLO-vạch được giải phóng khỏi RAM/GPU. Chỉ chạy 3 model nhẹ: đèn, xe, mũ → **Siêu nhanh!**
    > - **HSV Crop & Classify**: Kiểm tra mật độ pixel đỏ nửa trên hộp đèn → Chống nhiễu biển quảng cáo.
    """)