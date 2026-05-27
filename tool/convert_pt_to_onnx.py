from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO(r"face_det_w_model\human_detect.pt")

# Export the model to ONNX format (you can specify other formats as needed)
model.export(format="onnx")