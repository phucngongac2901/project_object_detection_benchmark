# Object Detection Pipeline Benchmark

## 📖 Giới thiệu (Introduction)
Dự án này là một hệ thống thử nghiệm và đánh giá hiệu năng (benchmark) cho pipeline nhận diện đối tượng (Object Detection) với mô hình YOLOv8. Mục tiêu chính là so sánh hiệu năng, độ trễ và tốc độ khung hình (FPS) giữa các phương pháp xử lý luồng video khác nhau:
1. **OpenCV (Pure)**: Sử dụng OpenCV thuần túy để giải mã và xử lý video tuần tự.
2. **GStreamer (Single-Thread)**: Sử dụng GStreamer để giải mã video kết hợp xử lý AI trên cùng một luồng.
3. **GStreamer (Multi-Thread)**: Tối ưu hóa bằng cách tách phần đọc video (GStreamer) và phần xử lý AI thành các luồng (threads) riêng biệt, giúp giảm thiểu độ trễ (latency) và tối đa hóa FPS.

Dự án sử dụng ONNX Runtime (hỗ trợ CUDA) để tăng tốc độ Inference trên GPU.

## 📂 Cấu trúc dự án (Project Structure)
- `test_opencv.py`: Script chạy pipeline với OpenCV.
- `test_gstreamer_single_thread.py`: Script chạy pipeline với GStreamer đơn luồng.
- `test_gstreamer_multi_thread.py`: Script chạy pipeline với GStreamer đa luồng.
- `benchmark_chart_all.py`: Tool tổng hợp file log `.csv` từ các bài test và xuất ra các biểu đồ phân tích chuyên sâu (FPS, KDE, CDF, Boxplot, Heatmap...).
- `config.yaml`: File cấu hình các thông số cho AI Model (thresholds, image size, classes...).
- `test_video/`: Nơi chứa video đầu vào để kiểm thử (`test.mp4`).
- `benchmark_logs/`: Thư mục (tự động tạo) lưu trữ file log dạng `.csv` và kết quả biểu đồ `charts/`.

## 🛠 Cài đặt (Installation)
Dự án được cấu hình bằng file `pyproject.toml`. Bạn có thể cài đặt các thư viện yêu cầu thông qua `pip` (hoặc `uv` nếu có cài đặt):

```bash
# Cài đặt bằng pip tiêu chuẩn của Python
pip install .

# Hoặc cài đặt bằng uv (nếu bạn sử dụng uv)
uv sync
```

## 🚀 Hướng dẫn sử dụng (Usage)

### 1. Chạy các bài test hiệu năng
Hãy chạy lần lượt từng script dưới đây để thu thập log hiệu năng. Hệ thống sẽ tự động tạo file log trong thư mục `benchmark_logs/`.

```bash
python test_opencv.py
python test_gstreamer_single_thread.py
python test_gstreamer_multi_thread.py
```
*(Ghi chú: Nếu bạn dùng `uv`, có thể gõ `uv run test_opencv.py` thay thế. Nhấn phím `q` trên cửa sổ video để dừng quá trình test)*

### 2. Trực quan hóa kết quả (Generate Charts)
Sau khi có các file logs `.csv`, bạn tiến hành chạy script vẽ biểu đồ:

```bash
python benchmark_chart_all.py
```
Biểu đồ tổng hợp, file excel tóm tắt và dashboard sẽ được sinh ra tại `benchmark_logs/charts/`.

### 3. Đọc báo cáo và Số liệu phân tích (Reports & Data)
Trong quá trình chạy và trực quan hóa, hệ thống đã tự động xuất ra hai loại tài liệu quan trọng để bạn đối chiếu:
- 📊 **File Excel (Dữ liệu gốc)**: Nằm tại `benchmark_logs/benchmark_results.xlsx`. File này chứa toàn bộ số liệu đo đạc thực tế (FPS, Latency, các chỉ số P50/P95/P99) của từng phương pháp xử lý.
- 📽️ **File PowerPoint (Slide thuyết trình)**: Nằm trong thư mục `slide/`. Đây là bản tóm tắt trực quan chứa các biểu đồ, so sánh cấu trúc pipeline, rất phù hợp để dùng cho các buổi báo cáo nhanh hoặc xem tổng quát kết quả.

## 🔧 Tùy chỉnh (Configuration)
Bạn có thể tinh chỉnh các thông số của model ở file `config.yaml`:
- **Đường dẫn model**: Cập nhật `model_path`.
- **Thông số input**: `input_width`, `input_height`.
- **Ngưỡng nhận diện**: `confidence_threshold`, `nms_threshold`.
