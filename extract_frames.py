import cv2
import os

# --- CẤU HÌNH ---
VIDEO_PATH = "videos_test/videoplayback5.mp4" # Đường dẫn video của em
OUTPUT_DIR = "dataset_extra"               # Thư mục lưu ảnh
FRAME_INTERVAL = 30                        # Cứ 30 frame cắt 1 ảnh (khoảng 1 giây/ảnh)

# Tạo thư mục nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract():
    cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    saved_count = 0
    
    print(f"🚀 Đang bắt đầu cắt ảnh từ: {VIDEO_PATH}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ lưu ảnh dựa trên interval để tránh rác dữ liệu
        if count % FRAME_INTERVAL == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, frame_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"📸 Đã lưu {saved_count} ảnh...")
        
        count += 1

    cap.release()
    print(f"✅ Hoàn thành! Đã lưu tổng cộng {saved_count} ảnh vào thư mục '{OUTPUT_DIR}'")

if __name__ == "__main__":
    extract()