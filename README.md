# 🚦 AI Traffic Monitor (ITS Pro) — Hệ Thống Giám Sát Giao Thông Thông Minh

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2019-61dafb.svg)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind%20CSS-38bdf8.svg)](https://tailwindcss.com/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLO11m-green.svg)](https://github.com/ultralytics/ultralytics)
[![SAM2](https://img.shields.io/badge/Segmentation-SAM2-orange.svg)](https://github.com/facebookresearch/segment-ant-mask-2)
[![RapidOCR](https://img.shields.io/badge/OCR-RapidOCR-brightgreen.svg)](https://github.com/RapidAI/RapidOCR)

Hệ thống giám sát và phát hiện vi phạm giao thông thời gian thực sử dụng các mô hình AI tiên tiến (YOLO11, SAM2, PaddleOCR/RapidOCR). Dự án hỗ trợ 2 chế độ chạy: **Web App hiện đại (React + FastAPI)** sử dụng kết nối WebSocket truyền luồng video tốc độ cao, và **Streamlit Desktop App** đơn giản, trực quan.

---

## 🚀 Hai Chế Độ Hoạt Động

### 1. Web App Mode (React + FastAPI + WebSocket) — *Khuyên dùng*
Kiến trúc Client-Server hiện đại:
* **Backend (FastAPI)**: Chịu trách nhiệm chạy pipeline AI, xử lý video qua luồng Thread nền, truyền frame nén và thông tin thống kê thời gian thực lên Client qua **WebSocket**.
* **Frontend (React + Vite + Tailwind CSS)**: Giao diện trực quan, hỗ trợ thiết lập tham số linh hoạt, xem log hệ thống dạng console và danh sách vi phạm kèm ảnh bằng chứng trực tiếp từ server.

### 2. Streamlit Mode (Desktop App)
* Phù hợp để chạy thử nghiệm nhanh cục bộ. Giao diện đơn giản được viết hoàn toàn bằng Python trên nền tảng Streamlit.

---

## 🔥 Tính Năng Nổi Bật

*   **⚡ Instant Calibration (Magic Frame)**: Chỉ dùng vài frame đầu tiên của video để tự động xác định vạch dừng bằng sự kết hợp giữa YOLO (phát hiện hộp cát vạch) và SAM2 (phân vùng chính xác vạch theo góc nghiêng của camera). Sau khi xác định xong, SAM2 được giải phóng hoàn toàn khỏi RAM/GPU để nhường hiệu năng tối đa cho luồng tracking!
*   **🔴 Phát Hiện Vượt Đèn Đỏ**:
    *   *HSV Traffic Light Verification*: Chống nhiễu từ biển quảng cáo bằng cách phân tích mật độ phân bổ màu đỏ thực tế trong khu vực đèn tín hiệu.
    *   *Bottom-Edge Logic*: Tính toán bánh xe tiếp xúc với vạch dừng (không dựa vào tâm xe), loại bỏ hoàn toàn các trường hợp báo còi sai khi đầu xe mới mấp mé vạch.
*   **🪖 Phát Hiện Không Đội Mũ Bảo Hiểm**: Nhận diện xe máy và quét người điều khiển để phát hiện các trường hợp không đội mũ bảo hiểm.
*   **🔢 Nhận Dạng Biển Số (OCR)**: Sử dụng YOLO để phát hiện khung biển số, tiền xử lý tăng độ nét siêu phân giải (Super-resolution) rồi chạy qua RapidOCR (ONNX PaddleOCR) để đọc ký tự chính xác.
*   **📸 Xuất Bằng Chứng Tự Động (Evidence)**: Tự động lưu ảnh chụp bằng chứng vi phạm vẽ sẵn bounding box, vạch giới hạn và xuất file CSV tổng hợp tại thư mục `evidence/`.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```bash
.
├── backend/                  # Mã nguồn Backend FastAPI
│   ├── main.py               # API endpoints, WebSocket server
│   └── processor.py          # AI Pipeline chính (Đèn, Xe, Mũ, Vạch dừng, Biển số)
├── frontend/                 # Mã nguồn Frontend React (Vite)
│   ├── src/
│   │   ├── components/       # Các component UI (Upload, Video, Bằng chứng...)
│   │   ├── App.jsx           # Component chính quản lý State & WebSocket
│   │   └── index.css         # CSS gốc chứa cấu hình Tailwind
│   └── package.json
├── core/                     # Các module AI cốt lõi
│   └── plate_reader.py       # Nhận dạng biển số (YOLO Detect + OCR)
├── utils/                    # Các hàm bổ trợ
│   ├── drawing.py            # Vẽ bounding box, vạch kẻ đường lên ảnh
│   └── violation.py          # Logic tính toán khoảng cách vi phạm hình học
├── models/                   # Thư mục chứa các trọng số mô hình (.pt / .onnx)
│   ├── yolo11m.pt            # YOLO11 phát hiện phương tiện gốc từ Ultralytics
│   ├── phathienden.pt        # YOLO phát hiện đèn tín hiệu giao thông
│   ├── phathienmu1.pt        # YOLO phát hiện không đội mũ bảo hiểm
│   ├── phathienvachmoi.onnx  # ONNX phát hiện vạch dừng
│   └── sam2_b.pt             # Trọng số Segment Anything 2
├── evidence/                 # Thư mục tự động sinh lưu ảnh bằng chứng & file CSV
├── app.py                    # Giao diện Streamlit Desktop cũ
├── requirements.txt          # Các thư viện Python cần thiết
├── run.bat                   # File khởi động nhanh chế độ Streamlit
└── run_web.bat               # File khởi động nhanh chế độ Web App (FastAPI + React)
```

---

## 🛠️ Cài Đặt & Khởi Chạy

### 1. Chuẩn Bị Môi Trường
Yêu cầu Python 3.10 trở lên và Node.js (dành cho frontend).

```bash
# Tạo môi trường ảo Python
python -m venv venv_paddle
source venv_paddle/bin/activate  # Trên Windows dùng: venv_paddle\Scripts\activate

# Cài đặt các thư viện Python
pip install -r requirements.txt
```

### 2. Khởi Chạy Nhanh Bằng File Script (Khuyên Dùng Trên Windows)

*   **Chạy chế độ Web App (React + FastAPI)**:
    Bấm đúp chuột vào file [run_web.bat](file:///c:/vscode/hoctap/Giamsatgiaothong2/run_web.bat). File script sẽ tự động chạy Backend uvicorn (Port 8000) và Frontend React (Port 3000), sau đó tự động mở trình duyệt tại: `http://localhost:3000`.
    
*   **Chạy chế độ Streamlit**:
    Bấm đúp chuột vào file [run.bat](file:///c:/vscode/hoctap/Giamsatgiaothong2/run.bat). Trình duyệt sẽ tự động mở tại: `http://localhost:8501`.

### 3. Khởi Chạy Thủ Công Bằng Dòng Lệnh

**Chạy Backend (FastAPI):**
```bash
# Đảm bảo đã active venv_paddle
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Chạy Frontend (React):**
```bash
cd frontend
npm install
npm run dev
```
Trình duyệt hiển thị tại: `http://localhost:3000`

---

## 🛡️ Danh Sách Mô Hình Sử Dụng
1.  **Vehicle Tracking**: YOLO11m (Ultralytics) + ByteTrack.
2.  **Traffic Light**: YOLO Custom (`phathienden.pt`).
3.  **Helmet Check**: YOLO Custom (`phathienmu1.pt`).
4.  **Stop Line**: YOLO Custom (`phathienvachmoi.onnx`) + SAM2 (`sam2_b.pt`).
5.  **License Plate Detector**: YOLO ONNX (`biensoxe.onnx`).
6.  **OCR Reader**: RapidOCR (ONNX PaddleOCR) siêu nhẹ chạy trên CPU.

---
*Phát triển và tối ưu hóa hệ thống ITS thông minh chất lượng cao!*