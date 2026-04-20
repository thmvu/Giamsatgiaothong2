import streamlit as st
import cv2
import tempfile
import os
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ======================================================================
# 🚦 HỆ THỐNG AI PHÁT HIỆN VI PHẠM GIAO THÔNG
# Kiến trúc TỐI ƯU CHO CPU (không cần GPU mạnh):
#   - YOLO11m gốc (COCO): Bắt xe + đèn giao thông (class 9)
#   - OpenCV HSV: Phân loại màu đèn (gần như FREE, không tốn GPU)
#   - Custom model: Chạy 1 LẦN ở frame đầu để tìm stop_line
#   - Helmet model: Soi mũ trên ảnh crop nhỏ
# ======================================================================

# --- COCO CLASS IDs (yolo11m.pt) ---
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7
COCO_TRAFFIC_LIGHT = 9
COCO_VEHICLE_CLASSES = {COCO_BICYCLE, COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK}
COCO_MOTORBIKE_CLASSES = {COCO_MOTORCYCLE}
# Detect cả xe lẫn đèn trong 1 lần chạy
COCO_DETECT_CLASSES = list(COCO_VEHICLE_CLASSES) + [COCO_TRAFFIC_LIGHT]

# --- CUSTOM MODEL CLASS IDs (chỉ dùng để tìm stop_line) ---
CUSTOM_STOP_LINE = 6


# ===== HÀM PHÂN LOẠI MÀU ĐÈN BẰNG OPENCV HSV =====
# Cực nhẹ, chạy trên CPU gần như tức thì (< 1ms)
def classify_traffic_light_color(crop_img):
    """
    Nhận ảnh crop bounding box đèn giao thông.
    Dùng HSV color space để đếm pixel đỏ/xanh/vàng.
    Trả về: 'red' | 'green' | 'yellow' | 'unknown'
    """
    if crop_img is None or crop_img.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Dải màu ĐỎ trong HSV (đỏ nằm ở 2 đầu hue: 0-10 và 160-180)
    mask_red_1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_red_2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    # Dải màu XANH LÁ
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    # Dải màu VÀNG
    mask_yellow = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([35, 255, 255]))

    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)
    yellow_count = cv2.countNonZero(mask_yellow)

    # Cần ít nhất 20 pixel để tránh nhiễu
    min_pixels = 20
    max_count = max(red_count, green_count, yellow_count)

    if max_count < min_pixels:
        return "unknown"

    if red_count == max_count:
        return "red"
    elif green_count == max_count:
        return "green"
    elif yellow_count == max_count:
        return "yellow"

    return "unknown"


# ===== CẤU HÌNH GIAO DIỆN =====
st.set_page_config(
    page_title="AI Traffic Violation Detector",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Hệ thống AI Phát hiện Vi phạm Giao thông")
st.caption("Tối ưu cho CPU: YOLO gốc (bắt xe + đèn) → OpenCV HSV (phân loại màu) → Helmet Model (soi mũ)")


# ===== NẠP MODEL =====
@st.cache_resource
def load_model(path):
    return YOLO(path)

model_vehicle = load_model('yolo11m.pt')          # Bắt xe + đèn (COCO)
model_helmet = load_model('phathienmu.pt')         # Soi mũ bảo hiểm
model_traffic = load_model('yolo11m_traffic_best.pt')  # Chỉ dùng 1 lần tìm stop_line

with st.expander("📋 Thông tin Model"):
    st.markdown("""
    | Model | Nhiệm vụ | Tần suất chạy |
    |-------|----------|---------------|
    | **YOLO11m gốc** | Bắt xe + đèn giao thông | Mỗi frame |
    | **OpenCV HSV** | Phân loại màu đèn (đỏ/xanh/vàng) | Khi phát hiện đèn (FREE) |
    | **Custom model** | Tìm vạch dừng (stop_line) | **1 lần duy nhất** |
    | **Helmet model** | Soi mũ bảo hiểm | Khi thấy xe máy |
    """)


# ===== SIDEBAR =====
st.sidebar.header("⚙️ Cài đặt")
conf_vehicle = st.sidebar.slider("Confidence: Bắt xe + đèn", 0.05, 0.9, 0.3, 0.05)
conf_helmet = st.sidebar.slider("Confidence: Soi mũ", 0.1, 0.9, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Loại Vi phạm")
check_helmet = st.sidebar.checkbox("🪖 Không đội mũ bảo hiểm", value=True)
check_redlight = st.sidebar.checkbox("🔴 Vượt đèn đỏ", value=True)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Hiệu năng")
process_every_n = st.sidebar.slider(
    "Xử lý mỗi N frame",
    min_value=1, max_value=5, value=1,
    help="Tăng = nhanh hơn, giảm = chính xác hơn"
)
show_all_detections = st.sidebar.checkbox("👁️ Hiển thị mọi detection", value=True)


# ===== UPLOAD VIDEO =====
uploaded_file = st.file_uploader("📁 Tải video lên...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"📹 {uploaded_file.name} | {width}x{height} | {fps:.1f} FPS | {total_frames} frames ({total_frames/max(fps,1):.1f}s)")

    # Thư mục bằng chứng
    evidence_dir = os.path.join(os.path.dirname(__file__), "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    # ===== TÌM STOP LINE (Chạy custom model 1 lần trên frame đầu) =====
    stop_line_y = None
    stop_line_coords = None

    if check_redlight:
        st.write("🔍 Đang tìm vạch dừng (stop_line) trên frame đầu tiên...")

        # Đọc frame đầu tiên
        ret_first, first_frame = cap.read()
        if ret_first:
            # Chạy custom model CHỈ tìm stop_line (class 6)
            sl_results = model_traffic.predict(
                first_frame,
                classes=[CUSTOM_STOP_LINE],
                conf=0.3,
                verbose=False
            )

            for sl_res in sl_results:
                if sl_res.boxes is not None and len(sl_res.boxes) > 0:
                    # Lấy stop_line có confidence cao nhất
                    best_idx = sl_res.boxes.conf.argmax()
                    sx1, sy1, sx2, sy2 = sl_res.boxes.xyxy[best_idx].int().cpu().tolist()
                    stop_line_y = (sy1 + sy2) // 2
                    stop_line_coords = (sx1, sy1, sx2, sy2)
                    st.success(f"✅ Đã tìm thấy vạch dừng tại Y={stop_line_y}")

            if stop_line_y is None:
                st.warning("⚠️ Không tìm thấy vạch dừng tự động. Bạn có thể chỉnh tay bên dưới.")

        # Reset video về đầu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Cho phép user chỉnh stop_line bằng slider nếu cần
    if check_redlight:
        use_manual_stopline = st.checkbox("✏️ Chỉnh vạch dừng thủ công", value=(stop_line_y is None))
        if use_manual_stopline:
            manual_y = st.slider(
                "Vị trí Y của vạch dừng",
                min_value=0, max_value=height,
                value=stop_line_y if stop_line_y is not None else height // 2
            )
            stop_line_y = manual_y
            stop_line_coords = (0, manual_y - 5, width, manual_y + 5)

    # ===== NÚT BẮT ĐẦU =====
    if st.button("🚀 Bắt đầu Quét", type="primary", use_container_width=True):

        # Trạng thái
        helmet_violated_ids = set()
        redlight_violated_ids = set()
        vehicle_prev_positions = {}
        violations_log = []

        # UI
        col_video, col_stats = st.columns([3, 1])
        with col_video:
            st_frame = st.empty()
        with col_stats:
            stats_placeholder = st.empty()

        progress_bar = st.progress(0, text="Đang quét...")
        frame_count = 0

        # =============================================
        # VÒNG LẶP XỬ LÝ FRAME
        # =============================================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % process_every_n != 0:
                continue

            progress_bar.progress(
                min(frame_count / total_frames, 1.0),
                text=f"Frame {frame_count}/{total_frames}"
            )

            # ================================================
            # BƯỚC 1: YOLO GỐC — Bắt xe + đèn trong 1 lần
            # (Chỉ 1 inference duy nhất trên frame!)
            # ================================================
            results = model_vehicle.track(
                frame,
                classes=COCO_DETECT_CLASSES,
                conf=conf_vehicle,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # ================================================
            # BƯỚC 2: PHÂN LOẠI KẾT QUẢ
            # ================================================
            # Reset trạng thái đèn MỖI FRAME (fix bug hiện đèn đỏ sai)
            current_traffic_state = "unknown"

            for res in results:
                if res.boxes is None or len(res.boxes) == 0:
                    continue

                boxes_xyxy = res.boxes.xyxy.int().cpu().tolist()
                classes_list = res.boxes.cls.int().cpu().tolist()
                confs_list = res.boxes.conf.cpu().tolist()

                track_ids = None
                if res.boxes.id is not None:
                    track_ids = res.boxes.id.int().cpu().tolist()

                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    cls_id = classes_list[i]
                    conf_val = confs_list[i]
                    track_id = track_ids[i] if track_ids is not None else None
                    class_name = model_vehicle.names[cls_id]

                    # ------------------------------------------
                    # 🚦 ĐÈN GIAO THÔNG (COCO class 9)
                    # YOLO detect hình dạng → OpenCV HSV phân loại màu
                    # ------------------------------------------
                    if cls_id == COCO_TRAFFIC_LIGHT:
                        # Crop vùng đèn
                        crop_light = frame[y1:y2, x1:x2]
                        light_color = classify_traffic_light_color(crop_light)
                        current_traffic_state = light_color

                        # Vẽ bbox đèn với màu tương ứng
                        color_map = {
                            "red": (0, 0, 255),
                            "green": (0, 255, 0),
                            "yellow": (0, 255, 255),
                            "unknown": (128, 128, 128)
                        }
                        light_draw_color = color_map.get(light_color, (128, 128, 128))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), light_draw_color, 2)

                        label_map = {
                            "red": "RED LIGHT",
                            "green": "GREEN LIGHT",
                            "yellow": "YELLOW LIGHT",
                            "unknown": "LIGHT (?)"
                        }
                        light_label = f"{label_map.get(light_color, 'LIGHT')} {conf_val:.2f}"
                        cv2.putText(frame, light_label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_draw_color, 2)
                        continue

                    # ------------------------------------------
                    # 🚗🏍️ PHƯƠNG TIỆN
                    # ------------------------------------------
                    if cls_id in COCO_VEHICLE_CLASSES:
                        vehicle_bottom_y = y2
                        is_helmet_violation = False
                        is_redlight_violation = False

                        # === CHECK 1: MŨ BẢO HIỂM (xe máy) ===
                        if check_helmet and cls_id in COCO_MOTORBIKE_CLASSES and track_id is not None:
                            if track_id in helmet_violated_ids:
                                is_helmet_violation = True
                            else:
                                crop_img = frame[y1:y2, x1:x2]
                                if crop_img.size > 0 and crop_img.shape[0] > 10 and crop_img.shape[1] > 10:
                                    h_results = model_helmet.predict(crop_img, conf=conf_helmet, verbose=False)
                                    for h_res in h_results:
                                        for b in h_res.boxes:
                                            label_name = h_res.names[int(b.cls[0])]
                                            if label_name == 'Without Helmet':
                                                is_helmet_violation = True
                                                helmet_violated_ids.add(track_id)
                                                ev_path = os.path.join(evidence_dir,
                                                    f"helmet_ID{track_id}_f{frame_count}.jpg")
                                                cv2.imwrite(ev_path, frame)
                                                violations_log.append({
                                                    "type": "🪖 Không đội mũ",
                                                    "vehicle": class_name,
                                                    "track_id": track_id,
                                                    "frame": frame_count,
                                                    "time_sec": round(frame_count / max(fps, 1), 2),
                                                    "evidence": ev_path
                                                })
                                                break
                                        if is_helmet_violation:
                                            break

                        # === CHECK 2: VƯỢT ĐÈN ĐỎ ===
                        if check_redlight and track_id is not None and stop_line_y is not None:
                            if track_id in redlight_violated_ids:
                                is_redlight_violation = True
                            elif current_traffic_state == "red":
                                prev_y = vehicle_prev_positions.get(track_id)
                                if prev_y is not None:
                                    if prev_y <= stop_line_y < vehicle_bottom_y:
                                        is_redlight_violation = True
                                        redlight_violated_ids.add(track_id)
                                        ev_path = os.path.join(evidence_dir,
                                            f"redlight_ID{track_id}_f{frame_count}.jpg")
                                        cv2.imwrite(ev_path, frame)
                                        violations_log.append({
                                            "type": "🔴 Vượt đèn đỏ",
                                            "vehicle": class_name,
                                            "track_id": track_id,
                                            "frame": frame_count,
                                            "time_sec": round(frame_count / max(fps, 1), 2),
                                            "evidence": ev_path
                                        })

                            vehicle_prev_positions[track_id] = vehicle_bottom_y

                        # === VẼ KẾT QUẢ ===
                        has_violation = is_helmet_violation or is_redlight_violation

                        if has_violation:
                            color = (0, 0, 255)
                            vio_texts = []
                            if is_helmet_violation:
                                vio_texts.append("KHONG MU")
                            if is_redlight_violation:
                                vio_texts.append("VUOT DEN DO")
                            label_text = f"ID{track_id} {class_name}: {' | '.join(vio_texts)}"

                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
                            cv2.putText(frame, label_text, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                        elif show_all_detections:
                            color = (0, 255, 0)
                            id_text = f"ID{track_id} " if track_id else ""
                            label_text = f"{id_text}{class_name} {conf_val:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label_text, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ================================================
            # VẼ HUD (Heads Up Display)
            # ================================================
            # Vẽ stop_line
            if stop_line_y is not None and stop_line_coords is not None:
                sx1, _, sx2, _ = stop_line_coords
                cv2.line(frame, (sx1, stop_line_y), (sx2, stop_line_y), (255, 0, 255), 3)
                cv2.putText(frame, "STOP LINE", (sx1, stop_line_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Hộp trạng thái đèn
            state_info = {
                "red": ("DEN DO", (0, 0, 255)),
                "green": ("DEN XANH", (0, 255, 0)),
                "yellow": ("DEN VANG", (0, 255, 255)),
                "unknown": ("--", (128, 128, 128))
            }
            state_text, state_color = state_info.get(current_traffic_state, ("--", (128, 128, 128)))
            cv2.rectangle(frame, (5, 5), (250, 45), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (250, 45), state_color, 2)
            cv2.putText(frame, f"Den: {state_text}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

            # ================================================
            # HIỂN THỊ
            # ================================================
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

            stats_placeholder.markdown(f"""
### 📊 Thống kê

| | |
|--|--|
| 🎞️ Frame | **{frame_count}/{total_frames}** |
| 🚦 Đèn | **{state_text}** |
| 🪖 VP Mũ | **{len(helmet_violated_ids)}** |
| 🔴 VP Đèn | **{len(redlight_violated_ids)}** |
| 🚗 Xe | **{len(vehicle_prev_positions)}** |
            """)

        # ===== KẾT THÚC =====
        cap.release()
        progress_bar.progress(1.0, text="✅ Hoàn tất!")

        st.markdown("---")
        st.header("📊 Tổng kết")

        total_violations = len(helmet_violated_ids) + len(redlight_violated_ids)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🪖 Không mũ", f"{len(helmet_violated_ids)} xe")
        with col2:
            st.metric("🔴 Vượt đèn đỏ", f"{len(redlight_violated_ids)} xe")
        with col3:
            st.metric("⚠️ Tổng", f"{total_violations} xe")

        if total_violations > 0:
            st.success(f"✅ Phát hiện **{total_violations}** vi phạm!")
        else:
            st.info("🎉 Không phát hiện vi phạm nào!")

        if violations_log:
            st.subheader("📋 Chi tiết")
            display_data = [{
                "Loại": v["type"],
                "Xe": v["vehicle"],
                "ID": v["track_id"],
                "Frame": v["frame"],
                "Giây": v["time_sec"]
            } for v in violations_log]
            st.dataframe(display_data, use_container_width=True)

            st.subheader("📸 Bằng chứng")
            n_cols = min(3, len(violations_log))
            ev_cols = st.columns(n_cols)
            for idx, v in enumerate(violations_log):
                if os.path.exists(v.get("evidence", "")):
                    with ev_cols[idx % n_cols]:
                        img = cv2.imread(v["evidence"])
                        if img is not None:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                     caption=f"{v['type']} ID{v['track_id']} ({v['time_sec']}s)",
                                     use_container_width=True)

            report = {
                "date": datetime.now().isoformat(),
                "video": uploaded_file.name,
                "summary": {
                    "helmet": len(helmet_violated_ids),
                    "redlight": len(redlight_violated_ids),
                    "total": total_violations
                },
                "violations": [{k: v for k, v in item.items() if k != "evidence"} for item in violations_log]
            }
            st.download_button("📥 Tải JSON", json.dumps(report, indent=2, ensure_ascii=False),
                               "report.json", "application/json", use_container_width=True)

else:
    st.markdown("""
    ### 📖 Hướng dẫn

    1. Upload video → 2. Cấu hình sidebar → 3. Nhấn "Quét" → 4. Xem kết quả

    ---

    ### ⚡ Kiến trúc tối ưu CPU

    ```
    Video ──→ Tách frame
                 │
                 ├─→ YOLO gốc: Bắt xe + đèn giao thông (1 inference)
                 │       │
                 │       ├─ Xe máy → Crop → Helmet Model (soi mũ)
                 │       │
                 │       └─ Đèn → Crop → OpenCV HSV (phân loại màu - FREE!)
                 │
                 └─→ Custom model: Tìm stop_line (CHỈ 1 LẦN ở frame đầu)
    ```

    **So sánh tốc độ:**
    | Kiến trúc | Inference/frame | Tốc độ |
    |-----------|----------------|--------|
    | 3 model YOLO | 3 lần | 🐢 Chậm |
    | **1 YOLO + HSV** | **1-2 lần** | **🚀 Nhanh 2-3x** |
    """)