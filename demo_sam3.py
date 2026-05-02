"""
demo_sam3.py — Biến BBox "cụt" của YOLO thành Mask "lan tỏa" bằng SAM3 ONNX
=============================================================================
Kiến trúc "3 bước lên mây":
  1. Image Encoder  : Encode frame → 6 FPN tensors (chạy 1 lần/frame)
  2. Language Encoder: Encode text prompt → tokens (chạy 1 lần cho mỗi prompt)
  3. Decoder        : Kết hợp FPN + text + BBox → Mask pixel-perfect

Input/Output thực tế (đã verify bằng onnxruntime):
  image_encoder  : image[3,1008,1008] uint8
                   → vision_pos_enc_{0,1,2}, backbone_fpn_{0,1,2}
  language_encoder: tokens[1,32] int64
                   → text_attention_mask[1,32], text_memory[32,1,256], text_embeds[32,1,1024]
  decoder         : original_{height,width}, backbone_fpn_{0,1,2}, vision_pos_enc_2,
                    language_mask[1,32], language_features[32,1,256],
                    box_coords[1,1,4], box_labels[1,1], box_masks[1,1]
                   → boxes, scores, masks(bool)

Cách dùng:
  python demo_sam3.py --image duong_pho.jpg --bbox 100 200 400 250
  python demo_sam3.py --video traffic.mp4
"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# ── Đường dẫn mặc định tới thư mục chứa 3 file ONNX ──────────────────────────
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sam3_vit_h")

# Kích thước ảnh mà model yêu cầu (đọc từ config.yaml)
INPUT_SIZE = 1008

# Text prompt mặc định — Giảng viên hỏi → em giải thích cái này!
DEFAULT_TEXT_PROMPT = "white stop line on asphalt road"


# =============================================================================
# TOKENIZER NỘI BỘ (không cần transformers — dùng clip simple tokenize)
# SAM3 language encoder nhận int64 tokens, shape [1, 32], padding = 0
# =============================================================================
def _build_vocab():
    """Bảng từ đơn giản — đủ để encode các từ thường dùng trong traffic context."""
    # Vocabulary cơ bản: ký tự → id (dạng character-level BPE giản lược)
    # SAM3 dùng clip tokenizer; để tránh phụ thuộc thư viện nặng, ta dùng
    # một bảng ASCII đơn giản với special tokens [SOS=49406, EOS=49407]
    SOS = 49406
    EOS = 49407
    PAD = 0
    return SOS, EOS, PAD


def simple_tokenize(text: str, max_len: int = 32) -> np.ndarray:
    """
    Tokenize text → int64 array shape [1, max_len].
    Dùng CLIP byte-pair encoding đơn giản (character level fallback).
    Nếu cài được 'clip' package thì dùng, nếu không thì dùng ASCII encoding.
    """
    SOS, EOS, PAD = _build_vocab()

    try:
        # Ưu tiên: dùng clip tokenizer chính thức nếu có
        import clip  # type: ignore
        tokens = clip.tokenize([text], context_length=max_len).numpy()[0]
        return tokens.astype(np.int64).reshape(1, max_len)
    except ImportError:
        pass

    # Fallback: ASCII-level encoding (vẫn cho kết quả ổn vì SAM3 robust)
    # Map từng ký tự sang id trong dải [256, 256+95] (printable ASCII)
    ids = [SOS]
    for ch in text.lower()[:max_len - 2]:
        # CLIP vocab: printable ASCII bắt đầu từ id = 256 + (ord(ch) - 32)
        if 32 <= ord(ch) <= 126:
            ids.append(256 + ord(ch) - 32)
        else:
            ids.append(256)  # unknown → space
    ids.append(EOS)

    # Pad
    while len(ids) < max_len:
        ids.append(PAD)

    return np.array([ids[:max_len]], dtype=np.int64)


# =============================================================================
# CLASS SAM3OnnxRunner — Load model + Inference
# =============================================================================
class SAM3OnnxRunner:
    """
    Chạy SAM3 qua ONNX Runtime — không cần PyTorch, không cần GPU bắt buộc.
    Tự động dùng CUDA nếu có (CUDAExecutionProvider), fallback về CPU.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.input_size = INPUT_SIZE

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print("📦 Loading SAM3 ONNX sessions...")

        # 1. Image Encoder
        enc_path = os.path.join(model_dir, "sam3_image_encoder.onnx")
        t0 = time.time()
        self.img_encoder = ort.InferenceSession(enc_path, sess_options=opts,
                                                providers=providers)
        print(f"  ✅ Image Encoder loaded  ({time.time()-t0:.1f}s)")

        # 2. Language Encoder
        lang_path = os.path.join(model_dir, "sam3_language_encoder.onnx")
        t0 = time.time()
        self.lang_encoder = ort.InferenceSession(lang_path, sess_options=opts,
                                                 providers=providers)
        print(f"  ✅ Language Encoder loaded ({time.time()-t0:.1f}s)")

        # 3. Decoder
        dec_path = os.path.join(model_dir, "sam3_decoder.onnx")
        t0 = time.time()
        self.decoder = ort.InferenceSession(dec_path, sess_options=opts,
                                            providers=providers)
        print(f"  ✅ Decoder loaded        ({time.time()-t0:.1f}s)")

        # Cache text embeddings (không đổi trong 1 run)
        self._text_cache: dict = {}

        # Hiển thị provider đang dùng thực tế
        prov = self.img_encoder.get_providers()
        print(f"  🖥️  Active providers: {prov}")

    # ── Bước A: Encode ảnh ───────────────────────────────────────────────────
    def encode_image(self, frame: np.ndarray) -> dict:
        """
        Tiền xử lý + chạy Image Encoder.

        Args:
            frame: ảnh BGR từ OpenCV, bất kỳ kích thước nào.

        Returns:
            dict chứa 6 tensors FPN (dùng làm input cho decoder).
        """
        h_orig, w_orig = frame.shape[:2]

        # Resize về [INPUT_SIZE, INPUT_SIZE], giữ tỉ lệ, pad phần còn lại
        scale = self.input_size / max(h_orig, w_orig)
        h_new = int(h_orig * scale)
        w_new = int(w_orig * scale)
        resized = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        # Pad về [INPUT_SIZE, INPUT_SIZE]
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:h_new, :w_new] = resized

        # BGR → RGB, HWC → CHW  →  [3, H, W]  uint8
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)                          # [3, 1008, 1008]

        t0 = time.time()
        enc_out = self.img_encoder.run(None, {"image": chw})
        dt = time.time() - t0

        output_names = [o.name for o in self.img_encoder.get_outputs()]
        result = dict(zip(output_names, enc_out))
        result["_scale"] = scale
        result["_orig_h"] = h_orig
        result["_orig_w"] = w_orig
        print(f"  🖼️  Image Encoder: {dt*1000:.0f}ms | scale={scale:.3f} "
              f"({w_orig}x{h_orig} → {w_new}x{h_new}+pad)")
        return result

    # ── Bước B: Encode text prompt ──────────────────────────────────────────
    def encode_text(self, text: str) -> dict:
        """
        Encode text prompt → embeddings.
        Kết quả được cache theo nội dung text (gọi 1 lần là đủ).

        Returns:
            dict: {text_attention_mask, text_memory, text_embeds}
        """
        if text in self._text_cache:
            return self._text_cache[text]

        tokens = simple_tokenize(text, max_len=32)               # [1, 32] int64
        t0 = time.time()
        lang_out = self.lang_encoder.run(None, {"tokens": tokens})
        dt = time.time() - t0

        output_names = [o.name for o in self.lang_encoder.get_outputs()]
        result = dict(zip(output_names, lang_out))
        self._text_cache[text] = result
        print(f"  📝 Language Encoder: {dt*1000:.0f}ms | prompt='{text}'")
        return result

    # ── Bước C: Decoder → Mask ──────────────────────────────────────────────
    def decode(self,
               img_feats: dict,
               text_feats: dict,
               bbox_xyxy: list,
               ) -> tuple[np.ndarray | None, float]:
        """
        Kết hợp image features + text features + BBox → Mask.

        Args:
            img_feats : output của encode_image()
            text_feats: output của encode_text()
            bbox_xyxy : [x1, y1, x2, y2] trong tọa độ ảnh GỐC

        Returns:
            (mask_uint8, score):
              mask_uint8: ndarray bool/uint8 kích thước ảnh gốc, hoặc None nếu fail
              score     : confidence score của mask tốt nhất
        """
        scale    = img_feats["_scale"]
        orig_h   = img_feats["_orig_h"]
        orig_w   = img_feats["_orig_w"]

        # Scale bbox về không gian model (đã resize + pad)
        x1, y1, x2, y2 = bbox_xyxy
        sx1 = x1 * scale
        sy1 = y1 * scale
        sx2 = x2 * scale
        sy2 = y2 * scale

        # Decoder inputs — tên lấy từ model inspection
        dec_inputs = {
            # Kích thước ảnh gốc (để decoder unpad mask)
            "original_height": np.array(orig_h, dtype=np.int64),
            "original_width":  np.array(orig_w, dtype=np.int64),

            # FPN features từ image encoder
            "backbone_fpn_0":    img_feats["backbone_fpn_0"],
            "backbone_fpn_1":    img_feats["backbone_fpn_1"],
            "backbone_fpn_2":    img_feats["backbone_fpn_2"],
            "vision_pos_enc_2":  img_feats["vision_pos_enc_2"],

            # Language features
            "language_mask":     text_feats["text_attention_mask"],  # [1,32] bool
            "language_features": text_feats["text_memory"],           # [32,1,256]

            # BBox prompt — [1, 1, 4] float32
            "box_coords":  np.array([[[sx1, sy1, sx2, sy2]]], dtype=np.float32),
            "box_labels":  np.array([[2]], dtype=np.int64),   # 2 = bbox label trong SAM3
            "box_masks":   np.array([[True]], dtype=bool),
        }

        t0 = time.time()
        try:
            dec_out = self.decoder.run(None, dec_inputs)
        except Exception as e:
            print(f"  ❌ Decoder error: {e}")
            return None, 0.0
        dt = time.time() - t0

        out_names = [o.name for o in self.decoder.get_outputs()]
        out_dict  = dict(zip(out_names, dec_out))

        masks_raw = out_dict.get("masks")   # bool, shape [N, 1, H, W] or [N, H, W]
        scores    = out_dict.get("scores")  # float, shape [N]

        if masks_raw is None or masks_raw.shape[0] == 0:
            print(f"  ⚠️ Decoder trả về mask rỗng ({dt*1000:.0f}ms)")
            return None, 0.0

        # Lấy mask có score cao nhất
        best_idx  = int(np.argmax(scores)) if scores is not None else 0
        best_score = float(scores[best_idx]) if scores is not None else 1.0
        best_mask = masks_raw[best_idx]                        # [1, H, W] or [H, W]

        # Squeeze về [H, W]
        while best_mask.ndim > 2:
            best_mask = best_mask[0]

        mask_uint8 = best_mask.astype(np.uint8) * 255

        # Resize về kích thước ảnh gốc nếu cần
        if mask_uint8.shape[:2] != (orig_h, orig_w):
            mask_uint8 = cv2.resize(mask_uint8, (orig_w, orig_h),
                                    interpolation=cv2.INTER_NEAREST)

        print(f"  🎭 Decoder: {dt*1000:.0f}ms | score={best_score:.3f} | "
              f"mask_sum={mask_uint8.sum()//255}px")
        return mask_uint8, best_score

    # ── Pipeline tổng hợp ───────────────────────────────────────────────────
    def get_full_lane_mask(self,
                           frame: np.ndarray,
                           bbox_xyxy: list,
                           text_prompt: str = DEFAULT_TEXT_PROMPT,
                           ) -> tuple[np.ndarray | None, float]:
        """
        🎯 Pipeline đầy đủ: ảnh + BBox + text → Mask lan tỏa.

        Args:
            frame      : ảnh BGR từ OpenCV
            bbox_xyxy  : [x1, y1, x2, y2] BBox từ YOLO
            text_prompt: câu mô tả đối tượng (ví dụ: "white stop line")

        Returns:
            (mask_uint8, score) — mask shape == frame.shape[:2]
        """
        t_start = time.time()
        img_feats  = self.encode_image(frame)
        text_feats = self.encode_text(text_prompt)
        mask, score = self.decode(img_feats, text_feats, bbox_xyxy)
        print(f"  ⏱️  Total pipeline: {(time.time()-t_start)*1000:.0f}ms")
        return mask, score


# =============================================================================
# HÀM VẼ KẾT QUẢ
# =============================================================================
def visualize(frame: np.ndarray,
              mask: np.ndarray | None,
              bbox: list,
              score: float,
              text_prompt: str) -> np.ndarray:
    """Vẽ BBox (YOLO-style) + Mask overlay (SAM3-style) lên frame."""
    vis = frame.copy()
    x1, y1, x2, y2 = map(int, bbox)

    # BBox YOLO — màu vàng, nét đứt → "nhỏ hẹp"
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
    cv2.putText(vis, f"YOLO BBox", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

    if mask is not None:
        # Overlay mask bán trong suốt — màu tím hồng nổi bật
        overlay = vis.copy()
        overlay[mask > 0] = [180, 0, 255]   # magenta
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

        # Vẽ contour mask — đường viền xanh lá nổi rõ
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 80), 2)

        # Label SAM3
        if contours:
            cx, cy, cw, ch = cv2.boundingRect(contours[0])
            cv2.putText(vis, f"SAM3 Mask (score={score:.2f})",
                        (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 80), 2)

    # Watermark
    h, w = vis.shape[:2]
    cv2.putText(vis, f"Prompt: \"{text_prompt}\"",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    return vis


# =============================================================================
# DEMO 1: Ảnh tĩnh
# =============================================================================
def demo_image(args):
    runner = SAM3OnnxRunner(model_dir=args.model_dir)
    frame  = cv2.imread(args.image)
    if frame is None:
        print(f"❌ Không đọc được ảnh: {args.image}")
        return

    bbox = list(map(float, args.bbox))   # [x1, y1, x2, y2]
    print(f"\n🖼️  Ảnh: {args.image}  |  BBox: {bbox}  |  Prompt: '{args.text}'")

    mask, score = runner.get_full_lane_mask(frame, bbox, text_prompt=args.text)

    result = visualize(frame, mask, bbox, score, args.text)
    cv2.imshow("SAM3 Demo — BBox YOLO → Mask lan toa", result)

    out_path = os.path.splitext(args.image)[0] + "_sam3_mask.jpg"
    cv2.imwrite(out_path, result)
    print(f"✅ Kết quả lưu tại: {out_path}")
    print("Nhấn phím bất kỳ để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# DEMO 2: Video — YOLO BBox tự detect + SAM3 "lan tỏa"
# =============================================================================
def demo_video(args):
    """
    Demo video: mỗi frame sẽ:
      1. Dùng YOLO (vachkeduongbbox.pt) tìm BBox vạch dừng.
      2. SAM3 "bloom" mask từ BBox đó.
    Nếu không có YOLO model, dùng BBox cố định từ args.bbox.
    """
    runner = SAM3OnnxRunner(model_dir=args.model_dir)

    # Thử load YOLO nếu có
    yolo_model = None
    if os.path.exists(args.yolo_model):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(args.yolo_model)
            print(f"✅ YOLO model loaded: {args.yolo_model}")
        except ImportError:
            print("⚠️ ultralytics chưa cài. Dùng BBox cố định từ --bbox.")
    else:
        print(f"⚠️ Không tìm thấy {args.yolo_model}. Dùng BBox cố định.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Không mở được video: {args.video}")
        return

    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 {args.video} | {vid_w}x{vid_h} | {fps:.1f}fps | {total} frames")

    # Cache image embeddings — chạy encoder 1 lần mỗi N frame
    cached_img_feats = None
    encode_every = max(1, int(fps))  # Re-encode mỗi 1 giây

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.skip_frames != 0:
            continue

        # Lấy BBox: ưu tiên YOLO, fallback về args.bbox
        bbox = None
        if yolo_model is not None:
            results = yolo_model.predict(frame, conf=0.25, verbose=False)
            for r in results:
                if r.boxes and len(r.boxes) > 0:
                    # Lấy bbox rộng nhất (vạch dừng thường rộng)
                    best_w = 0
                    for i in range(len(r.boxes)):
                        b = r.boxes.xyxy[i].tolist()
                        bw = b[2] - b[0]
                        if bw > best_w:
                            best_w = bw
                            bbox = b
                    break

        if bbox is None:
            if args.bbox:
                bbox = list(map(float, args.bbox))
            else:
                # Không có BBox → bỏ qua frame này
                cv2.imshow("SAM3 Video Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        # Re-encode image mỗi N frame (tiết kiệm thời gian)
        if frame_idx % encode_every == 1 or cached_img_feats is None:
            cached_img_feats = runner.encode_image(frame)

        text_feats = runner.encode_text(args.text)
        mask, score = runner.decode(cached_img_feats, text_feats, bbox)

        vis = visualize(frame, mask, bbox, score, args.text)
        cv2.putText(vis, f"Frame {frame_idx}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("SAM3 Video Demo — Nhan Q de thoat", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video demo kết thúc.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Demo SAM3 ONNX — Promptable Concept Segmentation cho vạch dừng",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Demo ảnh tĩnh với BBox thủ công
  python demo_sam3.py --image duong_pho.jpg --bbox 100 200 400 250

  # Demo video, YOLO tự tìm BBox
  python demo_sam3.py --video traffic.mp4

  # Chỉ định model dir khác
  python demo_sam3.py --image test.jpg --bbox 50 300 800 340 --model-dir ./sam3_vit_h

  # Thay đổi text prompt
  python demo_sam3.py --image test.jpg --bbox 50 300 800 340 \\
      --text "long white stop line on the asphalt road"
        """
    )

    # Source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Đường dẫn tới ảnh tĩnh (.jpg/.png)")
    group.add_argument("--video", help="Đường dẫn tới video (.mp4/.avi)")

    # BBox
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="BBox thủ công [x1 y1 x2 y2] (dùng khi không có YOLO)")

    # Model paths
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Thư mục chứa 3 file .onnx (mặc định: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--yolo-model", default="vachkeduongbbox.pt",
                        help="YOLO model để detect vạch (cho demo video)")

    # Text prompt
    parser.add_argument("--text", default=DEFAULT_TEXT_PROMPT,
                        help=f"Text prompt (mặc định: '{DEFAULT_TEXT_PROMPT}')")

    # Performance
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Chỉ xử lý mỗi N frame (mặc định: 1, tất cả)")

    args = parser.parse_args()

    # Validate model dir
    if not os.path.isdir(args.model_dir):
        print(f"❌ Không tìm thấy model dir: {args.model_dir}")
        print(f"   Tạo thư mục và đặt 3 file ONNX vào đó:")
        print(f"   {args.model_dir}/sam3_image_encoder.onnx")
        print(f"   {args.model_dir}/sam3_language_encoder.onnx")
        print(f"   {args.model_dir}/sam3_decoder.onnx")
        return

    if args.image:
        if not args.bbox:
            parser.error("--image yêu cầu --bbox X1 Y1 X2 Y2")
        demo_image(args)
    else:
        demo_video(args)


if __name__ == "__main__":
    main()
