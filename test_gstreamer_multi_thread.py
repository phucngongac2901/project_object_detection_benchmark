"""
test_gstreamer_multi_thread.py — Multi-threaded version with GStreamer
Thread 1 (Main): GStreamer decode + cv2.imshow display
Thread 2 (AI Worker): preprocess → inference → postprocess

So sánh với:
  - test_gstreamer_single_thread.py  (GStreamer single-thread)
  - test_opencv.py                   (Pure OpenCV)

Output: benchmark_logs/benchmark_log_gstreamer_multi.csv
"""
import gi
import os
import cv2
import yaml
import numpy as np
import queue
import time
import threading
import csv

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Đăng ký thư mục CUDA + cuDNN DLLs (system-level) trước khi import onnxruntime
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
MODEL_PATH = os.path.join(base, "train_model", "Train_3", "det_wor_hel_v3.onnx")
VIDEO_PATH = os.path.join(base, "test_video", "test.mp4")
new_video_path = VIDEO_PATH.replace("\\", "/")
CONFIG_PATH = os.path.join(base, "config.yaml")
LOG_DIR = os.path.join(base, "benchmark_logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "benchmark_log_gstreamer_multi.csv")

WARMUP_FRAMES = 10

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

# Lấy FPS gốc của video
_tmp_cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = _tmp_cap.get(cv2.CAP_PROP_FPS) if _tmp_cap.isOpened() else 25.0
_tmp_cap.release()
print(f"[INFO] Video source FPS: {video_fps:.1f}")

# ============================================================
# GSTREAMER PIPELINE
# ============================================================
Gst.init(None)

pipeline_str = (
    f'filesrc location="{new_video_path}" ! '
    "decodebin ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true sync=true max-buffers=5 drop=true"
)
pipeline = Gst.parse_launch(pipeline_str)
sink = pipeline.get_by_name("sink")
bus = pipeline.get_bus()

# ============================================================
# ONNX MODEL
# ============================================================
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print(f"[INFO] ONNX providers: {session.get_providers()}")

# ============================================================
# CSV WRITER (thread-safe via lock)
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
csv_lock = threading.Lock()

# ============================================================
# QUEUES & SHARED STATE
# ============================================================
# Queue 1: GStreamer callback → AI worker (raw frames)
frame_queue = queue.Queue(maxsize=5)

# Queue 2: AI worker → Display thread (annotated frames + timing)
result_queue = queue.Queue(maxsize=2)

# Flag để dừng tất cả threads
stop_event = threading.Event()

# ============================================================
# GSTREAMER CALLBACK — đẩy raw frame vào frame_queue
# ============================================================
def new_frame(sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    frame = np.frombuffer(map_info.data, np.uint8).reshape(height, width, 3)
    try:
        frame_queue.put_nowait(frame.copy())
    except queue.Full:
        pass  # bỏ frame nếu queue đầy
    buffer.unmap(map_info)

    return Gst.FlowReturn.OK

sink.connect("new-sample", new_frame)

# ============================================================
# AI WORKER THREAD — preprocess → inference → postprocess
# ============================================================
def ai_worker():
    """Thread riêng chạy AI detect, không block main thread."""
    ai_frame_id = 0
    first_frame_time = None

    while not stop_event.is_set():
        # Lấy frame từ queue (chờ tối đa 100ms)
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        ai_frame_id += 1
        t_total_start = time.perf_counter()
        if first_frame_time is None:
            first_frame_time = t_total_start

        original_h, original_w = frame.shape[:2]

        # ===== PREPROCESS =====
        t_pre = time.perf_counter()
        blob = cv2.resize(frame, (640, 640))
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        t_pre_end = time.perf_counter()

        # ===== INFERENCE =====
        t_inf = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        t_inf_end = time.perf_counter()

        # ===== POSTPROCESS =====
        t_post = time.perf_counter()
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

        # Vẽ bounding boxes lên frame
        for i in indices:
            idx = int(i)
            x, y, w_box, h_box = boxes[idx]
            label = f"{CLASSES[class_ids[idx]]}: {confidences[idx]:.2f}"
            color = COLORS[class_ids[idx] % len(COLORS)]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        t_post_end = time.perf_counter()

        # Tính timing
        preprocess_ms = (t_pre_end - t_pre) * 1000
        inference_ms = (t_inf_end - t_inf) * 1000
        postprocess_ms = (t_post_end - t_post) * 1000
        ai_total_ms = preprocess_ms + inference_ms + postprocess_ms
        pipeline_total_ms = (t_post_end - t_total_start) * 1000
        ai_fps = 1000.0 / ai_total_ms if ai_total_ms > 0 else 0
        pipeline_fps = 1000.0 / pipeline_total_ms if pipeline_total_ms > 0 else 0

        timestamp_s = time.perf_counter() - first_frame_time

        timing = {
            'frame_id': ai_frame_id,
            'timestamp_s': timestamp_s,
            'preprocess_ms': preprocess_ms,
            'inference_ms': inference_ms,
            'postprocess_ms': postprocess_ms,
            'ai_total_ms': ai_total_ms,
            'pipeline_total_ms': pipeline_total_ms,
            'ai_fps': ai_fps,
            'pipeline_fps': pipeline_fps,
        }

        warmup_tag = " [WARMUP]" if ai_frame_id <= WARMUP_FRAMES else ""
        if ai_frame_id % 10 == 0 or ai_frame_id <= 5:
            print(f"[AI Frame {ai_frame_id:4d}] "
                  f"Pre: {preprocess_ms:.1f}ms | "
                  f"Inf: {inference_ms:.1f}ms | "
                  f"Post: {postprocess_ms:.1f}ms | "
                  f"AI: {ai_fps:.0f} FPS | "
                  f"Pipeline: {pipeline_fps:.0f} FPS{warmup_tag}")

        # Đẩy kết quả vào result_queue (bỏ frame cũ nếu đầy)
        try:
            result_queue.put_nowait((frame, timing))
        except queue.Full:
            try:
                result_queue.get_nowait()  # bỏ frame cũ
            except queue.Empty:
                pass
            result_queue.put_nowait((frame, timing))

    print("[AI] Worker thread stopped.")

# ============================================================
# MAIN THREAD — GStreamer events + Display
# ============================================================
def main():
    # Khởi chạy AI worker thread
    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()
    print("[INFO] AI worker thread started.")

    pipeline.set_state(Gst.State.PLAYING)
    print("[INFO] Pipeline playing.")

    # Display FPS — rolling window
    display_fps = 0.0
    display_fps_history = []
    total_display_frames = 0

    try:
        while not stop_event.is_set():
            # Bước 1: Pump GStreamer events
            context = GLib.MainContext.default()
            context.iteration(False)

            # Kiểm tra Bus của GStreamer xem đã hết video (EOS) hoặc gặp lỗi (ERROR) chưa
            message = bus.pop_filtered(Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if message:
                if message.type == Gst.MessageType.EOS:
                    print("\n[INFO] Đã phát hết video (GStreamer EOS).")
                elif message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    print(f"\n[ERROR] GStreamer Error: {err} | {debug}")
                stop_event.set()
                break

            # Bước 2: Lấy frame đã detect từ AI worker
            annotated_frame = None
            timing = None
            try:
                annotated_frame, timing = result_queue.get_nowait()
            except queue.Empty:
                pass

            # Bước 3: Hiển thị
            if annotated_frame is not None:
                total_display_frames += 1

                # Display FPS — rolling window 30 frames
                now = time.perf_counter()
                display_fps_history.append(now)
                if len(display_fps_history) > 30:
                    display_fps_history.pop(0)
                if len(display_fps_history) >= 2:
                    elapsed = display_fps_history[-1] - display_fps_history[0]
                    display_fps = (len(display_fps_history) - 1) / elapsed if elapsed > 0 else 0
                else:
                    display_fps = 0

                # CSV LOG (thread-safe)
                with csv_lock:
                    csv_writer.writerow([
                        timing['frame_id'], f"{timing['timestamp_s']:.3f}",
                        "0.00", f"{timing['preprocess_ms']:.2f}",
                        f"{timing['inference_ms']:.2f}", f"{timing['postprocess_ms']:.2f}",
                        f"{timing['ai_total_ms']:.2f}", f"{timing['pipeline_total_ms']:.2f}",
                        f"{timing['ai_fps']:.1f}", f"{timing['pipeline_fps']:.1f}",
                        f"{display_fps:.1f}",
                        f"{video_fps:.1f}", "gstreamer_multi"
                    ])

                # Vẽ FPS overlay lên góc trái trên
                ai_fps = timing['ai_fps']
                pipeline_fps_val = timing['pipeline_fps']
                fps_text = f"AI: {ai_fps:.0f} | Pipeline: {pipeline_fps_val:.0f} | Display: {display_fps:.0f} FPS"
                (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (8, 8), (18 + tw, 18 + th + 10), (0, 0, 0), -1)
                cv2.putText(annotated_frame, fps_text, (12, 12 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Vẽ timing breakdown dòng 2
                detail_text = (f"Pre: {timing['preprocess_ms']:.1f}ms | "
                               f"Inf: {timing['inference_ms']:.1f}ms | "
                               f"Post: {timing['postprocess_ms']:.1f}ms")
                (tw2, th2), _ = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y2 = 18 + th + 10
                cv2.rectangle(annotated_frame, (8, y2), (18 + tw2, y2 + th2 + 10), (0, 0, 0), -1)
                cv2.putText(annotated_frame, detail_text, (12, y2 + th2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Method label dòng 3
                method_text = "Method: GStreamer (Multi-Thread)"
                (tw3, th3), _ = cv2.getTextSize(method_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y3 = y2 + th2 + 10
                cv2.rectangle(annotated_frame, (8, y3), (18 + tw3, y3 + th3 + 10), (0, 0, 0), -1)
                cv2.putText(annotated_frame, method_text, (12, y3 + th3 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                cv2.imshow("Detections [Multi-Thread]", annotated_frame)

            # Bước 4: Phím thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    finally:
        csv_file.close()

    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    ai_thread.join(timeout=2)
    cv2.destroyAllWindows()

    print(f"\n[INFO] Benchmark log saved to: {CSV_PATH}")
    print(f"[INFO] Total display frames: {total_display_frames}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
