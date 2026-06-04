"""
test_gstreamer_single_thread.py — GStreamer pipeline, Single-thread
GStreamer decode → queue → preprocess → inference → postprocess → display
Tất cả xử lý trên cùng 1 vòng lặp chính.

So sánh với:
  - test_gstreamer_multi_thread.py  (GStreamer multi-thread)
  - test_opencv.py                  (Pure OpenCV)

Output: benchmark_logs/benchmark_log_gstreamer_single.csv
"""
import gi
import os
import cv2
import yaml
from ultralytics import YOLO
import numpy as np
import queue
import time
import csv

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Đăng ký thư mục CUDA + cuDNN DLLs (system-level) trước khi import onnxruntime
# ONNX Runtime dùng LoadLibrary nên cần thêm vào PATH (không chỉ add_dll_directory)
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
# MODEL_PATH sẽ được tự động load từ config.yaml ở bên dưới
VIDEO_PATH = os.path.join(base, "test_video", "test.mp4")
new_video_path = VIDEO_PATH.replace("\\", "/")
CONFIG_PATH = os.path.join(base, "config.yaml")
Face_MODEL_PATH = os.path.join(base, "face_det_w_model", "human_detect.onnx")
LOG_DIR = os.path.join(base, "benchmark_logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "benchmark_log_gstreamer_single.csv")

WARMUP_FRAMES = 10  # Skip N frame đầu khi tính trung bình (GPU warm-up)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config(CONFIG_PATH)
CURRENT_MODEL_KEY = 'helmet_model'

MODEL_PATH = os.path.join(base, config[CURRENT_MODEL_KEY]['model_path'])
CLASSES = config[CURRENT_MODEL_KEY]['classes']
INPUT_WIDTH = config[CURRENT_MODEL_KEY]['input_width']
INPUT_HEIGHT = config[CURRENT_MODEL_KEY]['input_height']
CONFIDENCE_THRESHOLD = config[CURRENT_MODEL_KEY]['confidence_threshold']
NMS_THRESHOLD = config[CURRENT_MODEL_KEY]['nms_threshold']
COLORS = [(0, 255, 0), (0, 0, 255)]  # helmet: xanh lá, head: đỏ

# Lấy FPS gốc của video bằng OpenCV (để ghi vào log so sánh)
_tmp_cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = _tmp_cap.get(cv2.CAP_PROP_FPS) if _tmp_cap.isOpened() else 25.0
_tmp_cap.release()
print(f"[INFO] Video source FPS: {video_fps:.1f}")

# ============================================================
# GSTREAMER PIPELINE
# ============================================================
Gst.init(None)

pipeline_str=(
    f'filesrc location="{new_video_path}" ! '
    "decodebin ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true sync=true max-buffers=8 drop=true"
)
pipeline = Gst.parse_launch(pipeline_str)
sink = pipeline.get_by_name("sink")
bus = pipeline.get_bus()

# ============================================================
# ONNX MODEL
# ============================================================
#load model Face
# face_session = ort.InferenceSession(Face_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# #Load Model và inference
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
# GSTREAMER CALLBACK
# ============================================================
# Queue giới hạn 5 frame, thread-safe, tự động chặn nếu đầy
frame_queue = queue.Queue(maxsize=5)

def new_frame(sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()  # dữ liệu ảnh để xử lý
    caps = sample.get_caps()  # thông số định dạng khung hình (chuẩn màu, độ phân giải)
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    frame = np.frombuffer(map_info.data, np.uint8).reshape(height, width, 3)
    try:
        frame_queue.put_nowait(frame.copy())  # copy để tránh lỗi memory sau unmap
    except queue.Full:
        pass  # bỏ frame nếu queue đầy — appsink cũng đã drop=true
    buffer.unmap(map_info)

    return Gst.FlowReturn.OK
    
sink.connect("new-sample", new_frame)

# ============================================================
# MAIN LOOP
# ============================================================
pipeline.set_state(Gst.State.PLAYING)
print("[INFO] Pipeline playing.")

frame_id = 0
display_fps = 0.0
display_fps_history = []
first_frame_time = None
cv2.namedWindow("Detections [GStreamer Single-Thread]", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections [GStreamer Single-Thread]", 960, 540)
cv2.moveWindow("Detections [GStreamer Single-Thread]", 200, 50)

try:
    while True:
        t_pipeline_start = time.perf_counter()

        # Bước 1: Call GStreamer xử lý sự kiện
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
            break

        # Bước 2: Lấy frame tiếp theo từ queue (FIFO — giữ đúng thứ tự video)
        frame = None
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            frame_id += 1
            if first_frame_time is None:
                first_frame_time = t_pipeline_start

            original_h, original_w = frame.shape[:2]

            # ===== PREPROCESS =====
            t_preprocess_start = time.perf_counter()

            input_size = (640, 640)
            blob = cv2.resize(frame, input_size)
            blob = blob.astype(np.float32) / 255.0  # Normalize to [0, 1]
            blob = blob.transpose(2, 0, 1)          # HWC to CHW
            blob = np.expand_dims(blob, axis=0)     # Add batch dimension

            t_preprocess_end = time.perf_counter()

            # ===== INFERENCE =====
            t_inference_start = time.perf_counter()
            outputs = session.run(None, {input_name: blob})
            t_inference_end = time.perf_counter()

            # ===== POSTPROCESS =====
            t_postprocess_start = time.perf_counter()

            # Post-processing YOLOv8 ONNX output
            output = outputs[0].squeeze(0).T  # Shape: (8400, 4 + num_classes)

            class_scores = output[:, 4:]
            class_ids_arr = np.argmax(class_scores, axis=1)
            confidences_arr = class_scores[np.arange(len(class_scores)), class_ids_arr]

            mask = confidences_arr >= CONFIDENCE_THRESHOLD
            
            filtered_output = output[mask]
            confidences_arr = confidences_arr[mask]
            class_ids_arr = class_ids_arr[mask]

            boxes = []
            if len(filtered_output) > 0:
                cx = filtered_output[:, 0]
                cy = filtered_output[:, 1]
                w = filtered_output[:, 2]
                h = filtered_output[:, 3]

                x1 = ((cx - w / 2) * original_w / INPUT_WIDTH).astype(int)
                y1 = ((cy - h / 2) * original_h / INPUT_HEIGHT).astype(int)
                w_box = (w * original_w / INPUT_WIDTH).astype(int)
                h_box = (h * original_h / INPUT_HEIGHT).astype(int)

                boxes = np.column_stack((x1, y1, w_box, h_box)).tolist()
                confidences = confidences_arr.tolist()
                class_ids = class_ids_arr.tolist()
            else:
                confidences = []
                class_ids = []

            # NMS
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
            if len(display_fps_history) > 30:
                display_fps_history.pop(0)
            if len(display_fps_history) >= 2:
                elapsed = display_fps_history[-1] - display_fps_history[0]
                display_fps = (len(display_fps_history) - 1) / elapsed if elapsed > 0 else 0
            else:
                display_fps = 0

            timestamp_s = now - first_frame_time if first_frame_time else 0

            # ===== CSV LOG =====
            # GStreamer decode là async (callback), nên decode_ms = 0 (không đo trực tiếp được)
            csv_writer.writerow([
                frame_id, f"{timestamp_s:.3f}",
                "0.00", f"{preprocess_ms:.2f}", f"{inference_ms:.2f}", f"{postprocess_ms:.2f}",
                f"{ai_total_ms:.2f}", f"{pipeline_total_ms:.2f}",
                f"{ai_fps:.1f}", f"{pipeline_fps:.1f}", f"{display_fps:.1f}",
                f"{video_fps:.1f}", "gstreamer_single"
            ])

            # ===== TERMINAL LOG =====
            if frame_id % 10 == 0 or frame_id <= 5:
                warmup_tag = " [WARMUP]" if frame_id <= WARMUP_FRAMES else ""
                print(f"[Frame {frame_id:4d}] "
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            detail_text = (f"Pre: {preprocess_ms:.1f}ms | "
                           f"Inf: {inference_ms:.1f}ms | "
                           f"Post: {postprocess_ms:.1f}ms")
            (tw2, th2), _ = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y2 = 18 + th + 10
            cv2.rectangle(frame, (8, y2), (18 + tw2, y2 + th2 + 10), (0, 0, 0), -1)
            cv2.putText(frame, detail_text, (12, y2 + th2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            method_text = "Method: GStreamer (Single-Thread)"
            (tw3, th3), _ = cv2.getTextSize(method_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y3 = y2 + th2 + 10
            cv2.rectangle(frame, (8, y3), (18 + tw3, y3 + th3 + 10), (0, 0, 0), -1)
            cv2.putText(frame, method_text, (12, y3 + th3 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

            # Show result
            cv2.imshow("Detections [GStreamer Single-Thread]", frame)
        
        # Phím thoát phải nằm ngoài if nhưng nằm trong while True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
    print(f"\n[INFO] Benchmark log saved to: {CSV_PATH}")
    print(f"[INFO] Total frames processed: {frame_id}")
