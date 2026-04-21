"""
🚦 HỆ THỐNG AI GIÁM SÁT GIAO THÔNG
=====================================
Kiến trúc Dual-Model theo hướng dẫn giáo sư:
  - Model 1: phathienden.pt   → Phát hiện đèn (green/red/yellow/off)
  - Model 2: yolo11m.pt       → Phát hiện phương tiện (car/motorcycle/bus/truck)
  - Model 3: phathienmu.pt    → Kiểm tra mũ bảo hiểm (xe máy)

Logic vi phạm:
  - Đèn ĐỎ + cạnh dưới xe vượt stop_line → VI PHẠM vượt đèn đỏ
  - Xe máy + Without Helmet → VI PHẠM không đội mũ
"""

import streamlit as st
import cv2
import tempfile
import os
import csv
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# Import utils
from utils.drawing import draw_box, draw_stop_line, draw_light_status
from utils.violation import has_crossed_line

# === COCO CLASS IDs (yolo11m.pt) ===
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
MOTORBIKE_CLASS = 3

VN_NAMES = {2: "Ô tô", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải"}

# ===== GIAO DIỆN =====
st.set_page_config(page_title="AI Traffic Monitor", page_icon="🚦", layout="wide")
st.title("🚦 Hệ thống AI Giám sát Giao thông")
st.caption("phathienden.pt (đèn) + yolo11m.pt (xe) + phathienmu.pt (mũ)")


# ===== LOAD MODEL (cached) =====
@st.cache_resource
def load_model(path):
    return YOLO(path)

light_model = load_model("phathienden.pt")    # Model đèn: green(0), off(1), red(2), yellow(3)
vehicle_model = load_model("yolo11m.pt")       # Model xe: COCO
helmet_model = load_model("phathienmu.pt")     # Model mũ: With/Without Helmet


# ===== SIDEBAR =====
st.sidebar.header("⚙️ Cài đặt")
conf_light = st.sidebar.slider("Confidence: Đèn", 0.1, 0.9, 0.5, 0.05)
conf_vehicle = st.sidebar.slider("Confidence: Xe", 0.1, 0.9, 0.4, 0.05)
conf_helmet = st.sidebar.slider("Confidence: Mũ", 0.1, 0.9, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Tính năng")
check_redlight = st.sidebar.checkbox("🔴 Phát hiện vượt đèn đỏ", value=True)
check_helmet_enabled = st.sidebar.checkbox("🪖 Kiểm tra mũ bảo hiểm", value=True)
check_plate_enabled = st.sidebar.checkbox("🔢 Nhận dạng biển số (OCR)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Hiệu năng")
process_every_n = st.sidebar.slider("Xử lý mỗi N frame", 1, 5, 1)
traffic_interval = st.sidebar.slider("Check đèn mỗi N frame", 1, 10, 3,
    help="Đèn thay đổi chậm → không cần check mọi frame")
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

    # ==========================================================
    # BƯỚC 1: ĐẶT STOP LINE bằng click 2 điểm trên ảnh
    # ==========================================================

    # Session state lưu 2 điểm click
    if "stop_pts" not in st.session_state:
        st.session_state.stop_pts = []

    # Khởi tạo stop_line_pts trước (tránh lỗi khi check_redlight=False)
    stop_line_pts = None  # ((x1,y1), (x2,y2)) tọa độ video gốc

    if check_redlight:
        st.subheader("📍 Bước 1: Vẽ vạch dừng")
        st.markdown(
            "**Click 2 điểm** trên ảnh để xác định vạch dừng.  "
            "Điểm 1 → Điểm 2 sẽ được nối thành đường vạch.  "
            "Nhấn **Reset** nếu muốn chọn lại."
        )

        ret_first, first_frame = cap.read()
        if ret_first:
            # --- Scale ảnh hiển thị (giữ tỷ lệ, max 800px) ---
            DISPLAY_W = 800
            scale = DISPLAY_W / vid_w
            display_h = int(vid_h * scale)

            # Vẽ preview (với điểm đã click)
            preview = first_frame.copy()
            pts = st.session_state.stop_pts

            # Vẽ các điểm đã click lên ảnh gốc (trước khi scale)
            for idx, p in enumerate(pts):
                px, py = p["x"], p["y"]
                cv2.circle(preview, (px, py), 8, (0, 255, 255), -1)
                cv2.putText(preview, f"P{idx+1}", (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if len(pts) == 2:
                p1 = (pts[0]["x"], pts[0]["y"])
                p2 = (pts[1]["x"], pts[1]["y"])
                stop_line_pts = (p1, p2)

                # Vẽ đường stop line
                cv2.line(preview, p1, p2, (255, 0, 255), 3)
                cv2.putText(preview, "STOP LINE",
                            (min(p1[0], p2[0]), min(p1[1], p2[1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # Overlay vùng vi phạm phía dưới stop line
                avg_y = (p1[1] + p2[1]) // 2
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, avg_y), (vid_w, vid_h), (0, 0, 255), -1)
                preview = cv2.addWeighted(overlay, 0.12, preview, 0.88, 0)
                cv2.putText(preview, "VUNG VI PHAM",
                            (vid_w // 2 - 100, avg_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

            # Scale ảnh preview xuống để hiển thị
            preview_resized = cv2.resize(preview, (DISPLAY_W, display_h))
            preview_pil = Image.fromarray(cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB))

            # --- Hiển thị ảnh clickable ---
            n_pts = len(st.session_state.stop_pts)
            if n_pts < 2:
                st.info(f"🖱️ Click điểm **{n_pts + 1}/2** trên ảnh bên dưới")
            else:
                st.success("✅ Đã chọn đủ 2 điểm! Vạch dừng đã được xác định.")

            clicked = streamlit_image_coordinates(
                preview_pil,
                key="stop_line_click"
            )

            # Xử lý click mới
            if clicked is not None:
                # Scale ngược tọa độ về kích thước video gốc
                real_x = int(clicked["x"] / scale)
                real_y = int(clicked["y"] / scale)
                real_x = max(0, min(real_x, vid_w - 1))
                real_y = max(0, min(real_y, vid_h - 1))

                # Chỉ thêm nếu chưa đủ 2 điểm
                if len(st.session_state.stop_pts) < 2:
                    st.session_state.stop_pts.append({"x": real_x, "y": real_y})
                    st.rerun()

            # Nút reset
            if st.button("🔄 Chọn lại vạch dừng", key="reset_pts"):
                st.session_state.stop_pts = []
                st.rerun()

            # Thông tin điểm đã chọn
            if len(st.session_state.stop_pts) == 2:
                p1, p2 = st.session_state.stop_pts
                st.caption(
                    f"Điểm 1: ({p1['x']}, {p1['y']})  →  "
                    f"Điểm 2: ({p2['x']}, {p2['y']})"
                )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ==========================================================
    # BƯỚC 2: QUÉT VIDEO
    # ==========================================================
    can_start = not (check_redlight and stop_line_pts is None)

    if not check_redlight:
        stop_line_pts = None  # Không cần stop line

    if can_start and st.button("🚀 Bắt đầu Quét", type="primary", use_container_width=True):

        # --- Trạng thái ---
        current_light = "unknown"
        violated_ids = set()         # Xe đã vi phạm đèn đỏ (sổ đen)
        helmet_violated_ids = set()  # Xe đã vi phạm mũ (sổ đen)
        plate_cache = {}             # track_id → biển số
        prev_bbox_cache = {}         # track_id → bbox frame trước (dùng cross detection)
        violations_log = []

        # --- UI ---
        col_vid, col_stat = st.columns([3, 1])
        with col_vid:
            st_frame = st.empty()
        with col_stat:
            stats_ph = st.empty()
        progress = st.progress(0, text="Đang quét...")

        frame_count = 0
        processed_count = 0

        # =============================================
        # VÒNG LẶP CHÍNH
        # =============================================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % process_every_n != 0:
                continue
            processed_count += 1

            progress.progress(min(frame_count / total_frames, 1.0),
                              text=f"Frame {frame_count}/{total_frames}")

            # ==========================================
            # BƯỚC A: NHẬN DIỆN ĐÈN GIAO THÔNG
            # Dùng phathienden.pt (chạy mỗi N frame)
            # ==========================================
            if processed_count % traffic_interval == 1 or traffic_interval == 1:
                light_results = light_model(frame, conf=conf_light, verbose=False)

                for r in light_results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    # Lấy đèn có confidence cao nhất
                    best_idx = r.boxes.conf.argmax()
                    cls_id = int(r.boxes.cls[best_idx])
                    detected_light = light_model.names[cls_id]  # 'green'/'red'/'yellow'/'off'

                    if detected_light in ("red", "green", "yellow"):
                        current_light = detected_light

                    # Vẽ tất cả đèn phát hiện được
                    for i in range(len(r.boxes)):
                        cls = int(r.boxes.cls[i])
                        name = light_model.names[cls]
                        conf_val = float(r.boxes.conf[i])
                        color_map = {"red": (0,0,255), "green": (0,255,0),
                                     "yellow": (0,255,255), "off": (128,128,128)}
                        draw_box(frame, r.boxes.xyxy[i],
                                 f"{name.upper()} {conf_val:.2f}",
                                 color_map.get(name, (200,200,200)))

            # ==========================================
            # BƯỚC B: NHẬN DIỆN PHƯƠNG TIỆN + TRACKING
            # Dùng yolo11m.pt (COCO)
            # ==========================================
            vehicle_results = vehicle_model.track(
                frame, conf=conf_vehicle, classes=VEHICLE_CLASSES,
                persist=True, tracker="bytetrack.yaml", verbose=False
            )

            for r in vehicle_results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].tolist()
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cls_id = int(r.boxes.cls[i])
                    conf_val = float(r.boxes.conf[i])
                    class_name = vehicle_model.names[cls_id]
                    track_id = int(r.boxes.id[i]) if r.boxes.id is not None else None

                    is_redlight_vio = False
                    is_helmet_vio = False
                    plate_text = ""

                    # ==========================================
                    # BƯỚC C: KIỂM TRA VI PHẠM VƯỢT ĐÈN ĐỎ
                    # Dùng has_crossed_line(): kiểm tra tâm-dưới xe
                    # vượt qua ĐƯỜNG THẲNG 2 điểm (hình học chính xác)
                    # ==========================================
                    if check_redlight and stop_line_pts is not None and track_id is not None:
                        if track_id in violated_ids:
                            is_redlight_vio = True  # Đã trong sổ đen
                        elif current_light == "red":
                            prev_bbox = prev_bbox_cache.get(track_id)
                            crossed = has_crossed_line(prev_bbox, bbox, stop_line_pts)
                            if crossed:
                                is_redlight_vio = True
                                violated_ids.add(track_id)
                                # 📸 Chụp bằng chứng
                                ev = os.path.join(evidence_dir,
                                    f"redlight_ID{track_id}_f{frame_count}.jpg")
                                cv2.imwrite(ev, frame)
                                violations_log.append({
                                    "time": round(frame_count / max(fps,1), 2),
                                    "frame": frame_count,
                                    "type": "Vượt đèn đỏ",
                                    "vehicle": VN_NAMES.get(cls_id, class_name),
                                    "track_id": track_id,
                                    "plate": "",
                                    "evidence": ev
                                })

                    # Lưu bbox hiện tại cho frame sau
                    if track_id is not None:
                        prev_bbox_cache[track_id] = bbox

                    # ==========================================
                    # BƯỚC D: KIỂM TRA MŨ BẢO HIỂM (xe máy)
                    # Crop xe máy → chạy phathienmu.pt
                    # ==========================================
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
                                            ev = os.path.join(evidence_dir,
                                                f"helmet_ID{track_id}_f{frame_count}.jpg")
                                            cv2.imwrite(ev, frame)
                                            violations_log.append({
                                                "time": round(frame_count / max(fps,1), 2),
                                                "frame": frame_count,
                                                "type": "Không đội mũ bảo hiểm",
                                                "vehicle": VN_NAMES.get(cls_id, class_name),
                                                "track_id": track_id,
                                                "plate": "",
                                                "evidence": ev
                                            })
                                            break
                                    if is_helmet_vio:
                                        break

                    # ==========================================
                    # BƯỚC E: BIỂN SỐ (OCR, 1 lần mỗi xe)
                    # ==========================================
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
                                    _, thresh = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    ocr_res = ocr_reader.readtext(thresh, detail=1)
                                    for (_, text, prob) in ocr_res:
                                        cleaned = re.sub(r'[^A-Za-z0-9\-.]', '', text).upper()
                                        if len(cleaned) >= 4 and prob > 0.3:
                                            plate_text = cleaned
                                            break
                                except Exception:
                                    pass
                            plate_cache[track_id] = plate_text
                            # Cập nhật biển số vào violations
                            if plate_text:
                                for v in violations_log:
                                    if v["track_id"] == track_id and not v["plate"]:
                                        v["plate"] = plate_text

                    # ==========================================
                    # BƯỚC F: VẼ KẾT QUẢ
                    # ==========================================
                    has_violation = is_redlight_vio or is_helmet_vio

                    if has_violation:
                        # ĐỎ — VI PHẠM
                        vio_parts = []
                        if is_redlight_vio:
                            vio_parts.append("VUOT DEN DO")
                        if is_helmet_vio:
                            vio_parts.append("KHONG MU")
                        label = f"ID{track_id} {class_name}: {' | '.join(vio_parts)}"
                        if plate_text:
                            label += f" [{plate_text}]"
                        draw_box(frame, bbox, label, (0, 0, 255), 3)
                    elif show_all:
                        # XANH — BÌNH THƯỜNG
                        id_str = f"ID{track_id} " if track_id else ""
                        label = f"{id_str}{class_name} {conf_val:.2f}"
                        if plate_text:
                            label += f" [{plate_text}]"
                        draw_box(frame, bbox, label, (0, 255, 0))

            # ==========================================
            # VẼ HUD + STOP LINE
            # ==========================================
            if stop_line_pts is not None:
                p1, p2 = stop_line_pts
                cv2.line(frame, p1, p2, (255, 0, 255), 3)
                cv2.putText(frame, "STOP LINE",
                            (min(p1[0], p2[0]), min(p1[1], p2[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            draw_light_status(frame, current_light)

            # Hiển thị
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

            plates_found = sum(1 for v in plate_cache.values() if v)
            light_vn = {"red":"ĐỎ","green":"XANH","yellow":"VÀNG"}.get(current_light,"--")
            stats_ph.markdown(f"""
### 📊 Live
| | |
|--|--|
| Frame | **{frame_count}/{total_frames}** |
| 🚦 Đèn | **{light_vn}** |
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
        with c1:
            st.metric("🔴 Vượt đèn đỏ", len(violated_ids))
        with c2:
            st.metric("🪖 Không mũ", len(helmet_violated_ids))
        with c3:
            st.metric("⚠️ Tổng VP", total_vio)

        if total_vio > 0:
            st.success(f"Phát hiện **{total_vio}** vi phạm!")
        else:
            st.info("🎉 Không phát hiện vi phạm!")

        # Cập nhật biển số cuối cùng
        for v in violations_log:
            tid = v["track_id"]
            if not v["plate"] and tid in plate_cache and plate_cache[tid]:
                v["plate"] = plate_cache[tid]

        if violations_log:
            # Bảng vi phạm
            st.subheader("📋 Bảng Vi phạm")
            st.dataframe([{
                "Thời gian (s)": v["time"],
                "Loại VP": v["type"],
                "Phương tiện": v["vehicle"],
                "ID": v["track_id"],
                "Biển số": v["plate"] or "N/A",
                "Frame": v["frame"]
            } for v in violations_log], use_container_width=True)

            # Ảnh bằng chứng
            st.subheader("📸 Bằng chứng")
            ncols = min(3, len(violations_log))
            ev_cols = st.columns(ncols)
            for idx, v in enumerate(violations_log):
                if v.get("evidence") and os.path.exists(v["evidence"]):
                    with ev_cols[idx % ncols]:
                        img = cv2.imread(v["evidence"])
                        if img is not None:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                     caption=f"{v['type']} ID{v['track_id']}",
                                     use_container_width=True)

            # Biển số
            if plate_cache:
                st.subheader("🔢 Biển số đã nhận dạng")
                st.dataframe([{
                    "ID": tid, "Biển số": p or "Không đọc được"
                } for tid, p in plate_cache.items()], use_container_width=True)

            # Xuất CSV + JSON
            st.subheader("📥 Xuất Báo cáo")
            csv_path = os.path.join(evidence_dir, "violations.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "Thời gian (s)", "Frame", "Loại vi phạm",
                    "Phương tiện", "ID", "Biển số"
                ])
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
                    st.download_button("📥 CSV", f.read(), "violations.csv",
                                       "text/csv", use_container_width=True)
            with col_json:
                report = {
                    "date": datetime.now().isoformat(),
                    "video": uploaded_file.name,
                    "summary": {
                        "redlight": len(violated_ids),
                        "helmet": len(helmet_violated_ids),
                        "total": total_vio
                    },
                    "violations": [{k:v for k,v in item.items() if k != "evidence"}
                                   for item in violations_log]
                }
                st.download_button("📥 JSON",
                    json.dumps(report, indent=2, ensure_ascii=False),
                    "violations.json", "application/json",
                    use_container_width=True)

else:
    # Trang chủ
    st.markdown("""
    ### 📖 Hướng dẫn
    1. Upload video → 2. Đặt vạch dừng (slider) → 3. Nhấn Quét → 4. Xem kết quả

    ---
    ### 🏗️ Kiến trúc

    ```
    app.py
    ├── utils/
    │   ├── drawing.py      ← draw_box(), draw_stop_line(), draw_light_status()
    │   └── violation.py    ← check_violation(light, bbox, stop_line)
    │
    ├── phathienden.pt      ← Model đèn: green/red/yellow/off
    ├── yolo11m.pt          ← Model xe: car/motorcycle/bus/truck (COCO)
    └── phathienmu.pt       ← Model mũ: With/Without Helmet
    ```

    ### 🔄 Pipeline

    ```
    Frame
      │
      ├──→ [A] phathienden.pt → Đèn đỏ/xanh/vàng
      │
      ├──→ [B] yolo11m.pt + ByteTrack → Xe + ID tracking
      │         │
      │         ├──→ [C] check_violation() → Đèn đỏ + qua vạch = PHẠT
      │         ├──→ [D] phathienmu.pt → Xe máy không mũ = PHẠT
      │         └──→ [E] EasyOCR → Đọc biển số (1 lần/xe)
      │
      └──→ [F] draw_box() → Vẽ kết quả lên frame
    ```
    """)