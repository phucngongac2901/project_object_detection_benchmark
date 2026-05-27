"""
test_opencv.py — Pure OpenCV version (no GStreamer)
Single-thread: cv2.VideoCapture → preprocess → inference → postprocess → display

So sánh với:
  - test_gstreamer_single_thread.py  (GStreamer single-thread)
  - test_gstreamer_multi_thread.py   (GStreamer multi-thread)

Output: benchmark_logs/benchmark_log_opencv.csv
"""
import os
import cv2
import yaml
import numpy as np
import time
import csv

# Đăng ký thư mục CUDA + cuDNN DLLs trước khi import onnxruntime
_dll_dirs = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64",
]
for _d in _dll_dirs:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + ";" + os.environ.get("PATH", "")
import onnxruntime as ort

# ============================================================
# CONFIG
# ============================================================
base = os.getcwd()
MODEL_PATH = os.path.join("train_model/Train_3", "det_wor_hel_v3.onnx")
VIDEO_PATH = os.path.join(base, "test_video", "test.mp4")
CONFIG_PATH = os.path.join(base, "config.yaml")
LOG_DIR = os.path.join(base, "benchmark_logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "benchmark_log_opencv.csv")

WARMUP_FRAMES = 10  # Skip N frame đầu khi tính trung bình (GPU warm-up)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config(CONFIG_PATH)
CLASSES = config['head_model']['classes']
INPUT_WIDTH = config['head_model']['input_width']
INPUT_HEIGHT = config['head_model']['input_height']
CONFIDENCE_THRESHOLD = config['head_model']['confidence_threshold']
NMS_THRESHOLD = config['head_model']['nms_threshold']
COLORS = [(0, 0, 255), (0, 255, 0)]  # head: đỏ, helmet: xanh lá

# ============================================================
# OPENCV VIDEO CAPTURE
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Không thể mở video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Video: {video_width}x{video_height} @ {video_fps:.1f} FPS, {total_frames} frames")

# ============================================================
# ONNX MODEL
# ============================================================
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print(f"[INFO] ONNX providers: {session.get_providers()}")

# ============================================================
# CSV WRITER
# ============================================================
csv_file = open(CSV_PATH, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'frame_id', 'timestamp_s',
    'decode_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms',
    'ai_total_ms', 'pipeline_total_ms',
    'ai_fps', 'pipeline_fps', 'display_fps',
    'video_source_fps', 'method'
])

# ============================================================
# MAIN LOOP
# ============================================================
frame_id = 0
display_fps = 0.0
display_frame_count = 0
display_fps_timer = time.perf_counter()

# Rolling average cho display FPS (mượt hơn)
display_fps_history = []

print("[INFO] Bắt đầu xử lý...")

try:
    while True:
        t_pipeline_start = time.perf_counter()

        # ===== DECODE (cv2.read) =====
        t_decode_start = time.perf_counter()
        ret, frame = cap.read()
        t_decode_end = time.perf_counter()

        if not ret:
            print("[INFO] Hết video.")
            break

        frame_id += 1
        original_h, original_w = frame.shape[:2]

        # ===== PREPROCESS =====
        t_preprocess_start = time.perf_counter()
        blob = cv2.resize(frame, (640, 640))
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        t_preprocess_end = time.perf_counter()

        # ===== INFERENCE =====
        t_inference_start = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        t_inference_end = time.perf_counter()

        # ===== POSTPROCESS =====
        t_postprocess_start = time.perf_counter()
        output = outputs[0].squeeze(0).T

        boxes = []
        confidences = []
        class_ids = []

        for detection in output:
            cx, cy, w, h = detection[:4]
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence >= CONFIDENCE_THRESHOLD:
                x1 = int((cx - w / 2) * original_w / INPUT_WIDTH)
                y1 = int((cy - h / 2) * original_h / INPUT_HEIGHT)
                w_box = int(w * original_w / INPUT_WIDTH)
                h_box = int(h * original_h / INPUT_HEIGHT)
                boxes.append([x1, y1, w_box, h_box])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for i in indices:
            idx = int(i)
            x, y, w_box, h_box = boxes[idx]
            label = f"{CLASSES[class_ids[idx]]}: {confidences[idx]:.2f}"
            color = COLORS[class_ids[idx] % len(COLORS)]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        t_postprocess_end = time.perf_counter()

        # ===== TIMING =====
        decode_ms = (t_decode_end - t_decode_start) * 1000
        preprocess_ms = (t_preprocess_end - t_preprocess_start) * 1000
        inference_ms = (t_inference_end - t_inference_start) * 1000
        postprocess_ms = (t_postprocess_end - t_postprocess_start) * 1000
        ai_total_ms = preprocess_ms + inference_ms + postprocess_ms
        pipeline_total_ms = (t_postprocess_end - t_pipeline_start) * 1000
        ai_fps = 1000.0 / ai_total_ms if ai_total_ms > 0 else 0
        pipeline_fps = 1000.0 / pipeline_total_ms if pipeline_total_ms > 0 else 0

        # Display FPS — rolling window 30 frames
        now = time.perf_counter()
        display_fps_history.append(now)
        # Giữ lại chỉ 30 frame gần nhất
        if len(display_fps_history) > 30:
            display_fps_history.pop(0)
        if len(display_fps_history) >= 2:
            elapsed = display_fps_history[-1] - display_fps_history[0]
            display_fps = (len(display_fps_history) - 1) / elapsed if elapsed > 0 else 0
        else:
            display_fps = 0

        timestamp_s = (t_pipeline_start - display_fps_history[0]) if len(display_fps_history) > 0 else 0

        # ===== CSV LOG =====
        csv_writer.writerow([
            frame_id, f"{timestamp_s:.3f}",
            f"{decode_ms:.2f}", f"{preprocess_ms:.2f}", f"{inference_ms:.2f}", f"{postprocess_ms:.2f}",
            f"{ai_total_ms:.2f}", f"{pipeline_total_ms:.2f}",
            f"{ai_fps:.1f}", f"{pipeline_fps:.1f}", f"{display_fps:.1f}",
            f"{video_fps:.1f}", "opencv"
        ])

        # ===== TERMINAL LOG =====
        if frame_id % 10 == 0 or frame_id <= 5:
            warmup_tag = " [WARMUP]" if frame_id <= WARMUP_FRAMES else ""
            print(f"[Frame {frame_id:4d}] "
                  f"Decode: {decode_ms:.1f}ms | "
                  f"Pre: {preprocess_ms:.1f}ms | "
                  f"Inf: {inference_ms:.1f}ms | "
                  f"Post: {postprocess_ms:.1f}ms | "
                  f"AI: {ai_fps:.0f} FPS | "
                  f"Pipeline: {pipeline_fps:.0f} FPS | "
                  f"Display: {display_fps:.0f} FPS{warmup_tag}")

        # ===== FPS OVERLAY =====
        fps_text = f"AI: {ai_fps:.0f} | Pipeline: {pipeline_fps:.0f} | Display: {display_fps:.0f} FPS"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (8, 8), (18 + tw, 18 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (12, 12 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        detail_text = (f"Decode: {decode_ms:.1f}ms | Pre: {preprocess_ms:.1f}ms | "
                       f"Inf: {inference_ms:.1f}ms | Post: {postprocess_ms:.1f}ms")
        (tw2, th2), _ = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y2 = 18 + th + 10
        cv2.rectangle(frame, (8, y2), (18 + tw2, y2 + th2 + 10), (0, 0, 0), -1)
        cv2.putText(frame, detail_text, (12, y2 + th2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        method_text = "Method: OpenCV (Pure)"
        (tw3, th3), _ = cv2.getTextSize(method_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y3 = y2 + th2 + 10
        cv2.rectangle(frame, (8, y3), (18 + tw3, y3 + th3 + 10), (0, 0, 0), -1)
        cv2.putText(frame, method_text, (12, y3 + th3 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ===== DISPLAY =====
        cv2.imshow("Detections [OpenCV]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Benchmark log saved to: {CSV_PATH}")
    print(f"[INFO] Total frames processed: {frame_id}")
