# 🚦 AI Traffic Monitor (ITS Pro) — Hệ Thống Giám Sát Giao Thông Thông Minh

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLO11m-green.svg)](https://github.com/ultralytics/ultralytics)
[![SAM](https://img.shields.io/badge/Segmentation-SAM2-orange.svg)](https://github.com/facebookresearch/segmenter-ant-mask-2)

> **Kiến trúc "Ma Thuật" One-time Calibration**: Sự kết hợp hủy diệt giữa YOLO11 và SAM2 để đạt độ chính xác cấp độ milimet nhưng vẫn giữ được tốc độ xử lý Real-time cực đỉnh! 🚀

---

## 🔥 Tính Năng "Cực Cháy"

*   **⚡ Instant Calibration (Frame 1)**: Chỉ dùng duy nhất Frame đầu tiên để tự động tìm vạch dừng bằng YOLO-BBox và segment chính xác bằng SAM2. Sau khi xong, SAM2 sẽ được "đá" khỏi RAM/GPU để nhường chỗ cho tốc độ xử lý xe!
*   **🔴 Phát Hiện Vượt Đèn Đỏ**:
    *   **HSV Verification**: Chống nhiễu cực mạnh từ biển quảng cáo, đèn đường bằng cách phân tích mật độ màu đỏ thực tế trong hộp đèn.
    *   **Bottom-Edge Logic**: Tính toán vi phạm dựa trên điểm tiếp xúc của bánh xe với vạch dừng, không phải tâm xe. Sai số gần như bằng 0!
*   **🪖 Kiểm Tra Mũ Bảo Hiểm**: Tự động soi từng xe máy để phát hiện các "thủ lĩnh" quên mũ bảo hiểm.
*   **🔢 Nhận Diện Biển Số (OCR)**: Tích hợp OCR để truy vết "danh tính" phương tiện vi phạm ngay lập tức.
*   **📸 Evidence Export**: Tự động chụp ảnh bằng chứng (có vẽ sẵn vạch vi phạm và BBox) và xuất báo cáo CSV/JSON cực chuyên nghiệp.

---

## 🏗️ Kiến Trúc Hệ Thống

Dự án được tối ưu hóa theo quy trình 2 giai đoạn:

### Giai đoạn 1: Khởi Tạo (Magic Frame)
1. **YOLO-Line**: Tìm kiếm BBox của vạch kẻ đường.
2. **SAM2**: "Vẽ" lại vạch kẻ đường với độ chính xác tuyệt đối theo góc camera.
3. **Release**: Giải phóng hoàn toàn YOLO-Line và SAM2 khỏi bộ nhớ.

### Giai đoạn 2: Giám Sát Real-time
*   **Model Đèn**: Theo dõi trạng thái xanh/vàng/đỏ.
*   **Model Xe (YOLO11m)**: Tracking phương tiện bằng ByteTrack.
*   **Model Mũ**: Soi lỗi không đội mũ bảo hiểm.
*   **OCR**: Đọc biển số khi có vi phạm.

---

## 📁 Cấu Trúc Thư Mục

```bash
.
├── app.py              # Main App (Giao diện Streamlit)
├── models/             # "Kho vũ khí" AI (.pt files)
│   ├── phathienden.pt  # Model nhận diện đèn
│   ├── yolo11m.pt      # Model track phương tiện
│   ├── phathienmu.pt   # Model soi mũ bảo hiểm
│   ├── sam2_b.pt       # SAM2 cho Calibration
│   └── vachkeduongbbox1.pt # YOLO detect vạch
├── utils/              # Logic xử lý hình học & vẽ
│   ├── drawing.py      # Render UI/Overlay
│   └── violation.py    # Logic tính toán vi phạm (Proximity + Side)
└── evidence/           # Nơi lưu bằng chứng vi phạm (Auto-generated)
```

---

## 🚀 Cài Đặt & Chạy

1. **Clone repo**:
   ```bash
   git clone https://github.com/your-username/Giamsatgiaothong2.git
   ```

2. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Yêu cầu: ultralytics, streamlit, opencv-python, easyocr, torch)*

3. **Khai hỏa**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Công Nghệ Sử Dụng

- **Engine**: YOLO11 (SOTA Object Detection)
- **Segmentation**: Segment Anything Model 2 (SAM2)
- **Tracking**: ByteTrack (Cực mượt, không mất dấu)
- **UI**: Streamlit (Hiện đại, dễ dùng)
- **Backend**: OpenCV & NumPy (Xử lý hình học tốc độ cao)

---

## 📸 Demo Hình Ảnh

*(Chèn ảnh chụp màn hình app hoặc GIF vi phạm tại đây để tăng độ cháy!)*

---
**Developed with ❤️ by Your Name**
*Hệ thống này được thiết kế để không một lỗi vi phạm nào có thể lọt lưới!*