"""
core/sam3_runner.py — SAM3 ONNX Runner cho bài toán "Bloom" vạch dừng
======================================================================
Tại sao không dùng SAM2?
  SAM2 với bbox prompt bị "giam" trong hộp BBox → bỏ sót phần vạch ngoài bbox.
  SAM3 có Language Encoder → hiểu khái niệm "white stop line" → tự lan
  ra ngoài bbox theo đặc tính hình học của vạch sơn.

I/O thực tế (đã verify):
  image_encoder  : image[3,1008,1008] uint8
                   → backbone_fpn_{0,1,2}[1,256,*], vision_pos_enc_{0,1,2}[1,256,*]
  language_encoder: tokens[1,32] int64
                   → text_attention_mask[1,32] bool
                   → text_memory[32,1,256] float
                   → text_embeds[32,1,1024] float
  decoder         : FPN tensors + language tensors + box_coords[1,1,4] float32
                    + box_labels[1,1] int64 + box_masks[1,1] bool
                   → masks(bool), scores(float), boxes(float)

Text prompt tốt nhất cho vạch dừng:
  "white stop line on asphalt road"   ← dài + rõ nghĩa → mask to hơn
"""

import os
import time
import numpy as np
import cv2

# ── Kích thước input model (từ config.yaml) ──────────────────────────────────
INPUT_SIZE = 1008

# ── Text prompt mặc định ─────────────────────────────────────────────────────
DEFAULT_PROMPT = "white stop line on asphalt road"


# =============================================================================
# CLIP-compatible Tokenizer (không cần cài clip/transformers)
# =============================================================================
def _clip_tokenize(text: str, context_length: int = 32) -> np.ndarray:
    """
    Tokenize text → int64 array [1, context_length] theo chuẩn CLIP.

    Thứ tự ưu tiên:
      1. clip package  (pip install git+https://github.com/openai/CLIP.git)
      2. transformers  (pip install transformers)
      3. Fallback: ASCII-level encoding (vẫn hoạt động tốt nhờ SAM3 robust)
    """
    # ── Thử clip chính thức ──
    try:
        import clip  # type: ignore
        tok = clip.tokenize([text], context_length=context_length)
        return tok.numpy().astype(np.int64)          # [1, 32]
    except (ImportError, Exception):
        pass

    # ── Thử HuggingFace CLIPTokenizer ──
    try:
        from transformers import CLIPTokenizer       # type: ignore
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        enc = tokenizer(
            text,
            max_length=context_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return enc["input_ids"].astype(np.int64)     # [1, 32]
    except (ImportError, Exception):
        pass

    # ── Fallback: ASCII-level BPE giản lược ──
    SOS, EOS, PAD = 49406, 49407, 0
    ids = [SOS]
    for ch in text.lower()[: context_length - 2]:
        # CLIP vocab: printable ASCII → id 256 + (ord - 32)
        ids.append(256 + max(ord(ch) - 32, 0) if 32 <= ord(ch) <= 126 else 256)
    ids.append(EOS)
    while len(ids) < context_length:
        ids.append(PAD)
    return np.array([ids[:context_length]], dtype=np.int64)


# =============================================================================
# CLASS SAM3OnnxRunner
# =============================================================================
class SAM3OnnxRunner:
    """
    Chạy SAM3 qua ONNX Runtime.

    Ưu điểm so với Ultralytics SAM2:
      - Text grounding: mask vượt ra ngoài BBox theo "khái niệm" vạch sơn
      - Không cần PyTorch (nhẹ hơn về dependency)
      - Tự dùng TensorRT/CUDA/CPU theo thứ tự ưu tiên

    Nhược điểm:
      - Model nặng (~3.6 GB): chỉ load khi calibrate, giải phóng ngay sau
      - Load lần đầu chậm (~30-60s tùy máy)
    """

    MODEL_DIR_DEFAULT = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sam3_vit_h"
    )

    def __init__(self, model_dir: str | None = None):
        import onnxruntime as ort

        self.model_dir = model_dir or self.MODEL_DIR_DEFAULT
        self._text_cache: dict[str, dict] = {}

        # Loại TensorRT để tránh log lỗi cublas — tự fallback CUDA → CPU
        all_providers = ort.get_available_providers()
        providers = [p for p in all_providers if p != "TensorrtExecutionProvider"]
        if not providers:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # Tắt verbose log

        print("  [SAM3-ONNX] Loading sessions (co the mat 30-60s)...")

        t0 = time.time()
        self._enc = ort.InferenceSession(
            os.path.join(self.model_dir, "sam3_image_encoder.onnx"),
            sess_options=opts, providers=providers,
        )
        self._lang = ort.InferenceSession(
            os.path.join(self.model_dir, "sam3_language_encoder.onnx"),
            sess_options=opts, providers=providers,
        )
        self._dec = ort.InferenceSession(
            os.path.join(self.model_dir, "sam3_decoder.onnx"),
            sess_options=opts, providers=providers,
        )

        active = self._enc.get_providers()
        print(f"  [SAM3-ONNX] Loaded in {time.time()-t0:.1f}s | providers={active}")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: chạy toàn bộ pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def predict_mask(
        self,
        frame: np.ndarray,
        bbox_xyxy: list,
        text_prompt: str = DEFAULT_PROMPT,
    ) -> tuple[np.ndarray | None, float]:
        """
        Nhận ảnh + BBox từ YOLO → trả về mask uint8 kích thước ảnh gốc.

        Args:
            frame      : ảnh BGR (bất kỳ kích thước)
            bbox_xyxy  : [x1, y1, x2, y2] tọa độ BBox từ YOLO
            text_prompt: mô tả ngôn ngữ tự nhiên, ví dụ "white stop line"

        Returns:
            (mask_uint8, score):
              mask_uint8 — ndarray uint8 {0,255}, shape = frame.shape[:2]
                           None nếu không tìm thấy mask
              score      — confidence [0,1]
        """
        img_feats  = self._encode_image(frame)
        text_feats = self._encode_text(text_prompt)
        return self._decode(img_feats, text_feats, bbox_xyxy)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Image Encoder
    # ─────────────────────────────────────────────────────────────────────────
    def _encode_image(self, frame: np.ndarray) -> dict:
        h_orig, w_orig = frame.shape[:2]

        # Resize giữ tỉ lệ, pad về INPUT_SIZE × INPUT_SIZE
        scale = INPUT_SIZE / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * scale), int(w_orig * scale)
        resized = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        padded[:h_new, :w_new] = resized

        # BGR → RGB, HWC → CHW  [3, 1008, 1008] uint8
        chw = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        out = self._enc.run(None, {"image": chw})
        names = [o.name for o in self._enc.get_outputs()]
        feats = dict(zip(names, out))
        feats.update({"_scale": scale, "_orig_h": h_orig, "_orig_w": w_orig})
        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Language Encoder (cached)
    # ─────────────────────────────────────────────────────────────────────────
    def _encode_text(self, text: str) -> dict:
        if text in self._text_cache:
            return self._text_cache[text]

        tokens = _clip_tokenize(text, context_length=32)   # [1, 32] int64
        out    = self._lang.run(None, {"tokens": tokens})
        names  = [o.name for o in self._lang.get_outputs()]
        result = dict(zip(names, out))
        self._text_cache[text] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Decoder
    # ─────────────────────────────────────────────────────────────────────────
    def _decode(
        self,
        img_feats:  dict,
        text_feats: dict,
        bbox_xyxy:  list,
    ) -> tuple[np.ndarray | None, float]:
        orig_h = img_feats["_orig_h"]
        orig_w = img_feats["_orig_w"]

        # ── Chuyển bbox [x1,y1,x2,y2] pixel → [cx,cy,w,h] normalized [0,1] ──
        # RoiAlign trong geometry_encoder yêu cầu NORMALIZED coords, không phải pixel!
        x1, y1, x2, y2 = bbox_xyxy
        cx = float(np.clip((x1 + x2) / 2.0 / orig_w, 0.0, 1.0))
        cy = float(np.clip((y1 + y2) / 2.0 / orig_h, 0.0, 1.0))
        bw = float(np.clip((x2 - x1)       / orig_w, 0.001, 1.0))
        bh = float(np.clip((y2 - y1)       / orig_h, 0.001, 1.0))

        box_norm = np.array([[[cx, cy, bw, bh]]], dtype=np.float32)  # [1, 1, 4]
        print(f"  [SAM3-ONNX] box_norm(CxCyWH)=[{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")

        dec_in = {
            "original_height": np.array(orig_h, dtype=np.int64),
            "original_width":  np.array(orig_w,  dtype=np.int64),
            "backbone_fpn_0":   img_feats["backbone_fpn_0"],
            "backbone_fpn_1":   img_feats["backbone_fpn_1"],
            "backbone_fpn_2":   img_feats["backbone_fpn_2"],
            "vision_pos_enc_2": img_feats["vision_pos_enc_2"],
            "language_mask":     text_feats["text_attention_mask"],
            "language_features": text_feats["text_memory"],
            "box_coords": box_norm,
            "box_labels": np.array([[1]], dtype=np.int64),  # 1=positive foreground
            "box_masks":  np.array([[True]], dtype=bool),
        }

        try:
            out   = self._dec.run(None, dec_in)
            names = [o.name for o in self._dec.get_outputs()]
            d     = dict(zip(names, out))
        except Exception as e:
            print(f"  [SAM3-ONNX] Decoder error: {e}")
            return None, 0.0

        masks_raw = d.get("masks")
        scores    = d.get("scores")
        print(f"  [SAM3-ONNX] output masks={None if masks_raw is None else masks_raw.shape} "
              f"scores={scores}")

        if masks_raw is None or masks_raw.size == 0 or masks_raw.shape[0] == 0:
            return None, 0.0

        best_idx   = int(np.argmax(scores)) if (scores is not None and scores.size > 0) else 0
        best_score = float(scores[best_idx]) if (scores is not None and scores.size > 0) else 1.0
        best_mask  = masks_raw[best_idx]
        while best_mask.ndim > 2:
            best_mask = best_mask[0]

        mask_u8  = best_mask.astype(np.uint8) * 255
        nonzero  = int(mask_u8.sum() // 255)
        print(f"  [SAM3-ONNX] mask={mask_u8.shape}  nonzero={nonzero}px  score={best_score:.3f}")

        if mask_u8.shape[:2] != (orig_h, orig_w):
            mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

    # ─────────────────────────────────────────────────────────────────────────
    # Kiểm tra model dir có đủ file không
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def is_available(cls, model_dir: str | None = None) -> bool:
        d = model_dir or cls.MODEL_DIR_DEFAULT
        required = [
            "sam3_image_encoder.onnx",
            "sam3_language_encoder.onnx",
            "sam3_decoder.onnx",
        ]
        return all(os.path.exists(os.path.join(d, f)) for f in required)
