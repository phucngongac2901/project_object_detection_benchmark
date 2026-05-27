import os
import json
import shutil
import yaml
from PIL import Image
base = os.getcwd()
folder_path = os.path.join(base, "dataset_3")
output_folder = os.path.join(base, "dataset_3", "anylabeling")


# Đọc class names từ data.yaml
with open(os.path.join(folder_path, 'data.yaml'), 'r', encoding='utf-8') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

#tạo folder output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

splits = ["train","valid","test"]
for split in splits:
    image_dir = os.path.join(folder_path,split,"images")
    label_dir = os.path.join(folder_path,split,"labels")

    if not os.path.exists(label_dir):
        continue
    
    #tạo folder output merged images
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            src = os.path.join(image_dir, filename)
            new = os.path.join(output_folder, filename)
            shutil.copy(src, new)
    
    #tạo folder output merged labels
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            src = os.path.join(label_dir, filename)
            new = os.path.join(output_folder, filename)
            shutil.copy(src, new)

for filename in os.listdir(output_folder):
    #chỉ xử lý file .jpg
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)

        #tim anh tuong ung
        txt_name = filename.replace(".jpg", ".txt")
        txt_path = os.path.join(output_folder, txt_name)

        #bo qua neu ko co anh di kem
        if not os.path.exists(txt_path):
            continue
        
        #Mở ảnh lấy kích thước (width, height) bằng PIL
        with Image.open(img_path) as img:
            img_resized = img.resize((640, 640))
            img_resized.save(img_path)  # Lưu đè ảnh gốc
            width, height = 640, 640
    
        #Khởi tạo dict theo cấu trúc AnyLabeling
        Anylabeling = {
            "version": "0.4.35",
            "imagePath": filename,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "shapes": []
        }
        with open(txt_path,'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            part = line.split()
            class_id = int(part[0])
            cx = float(part[1])
            cy = float(part[2])
            bw = float(part[3])
            bh = float(part[4])

            #convert YOLO normalized → pixel
            x1 = (cx - bw/2) * width
            y1 = (cy - bh/2) * height
            x2 = (cx + bw/2) * width
            y2 = (cy + bh/2) * height

            #tạo shape theo AnyLabeling
            # Map class index sang tên class
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            shape = {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle"
            }
            Anylabeling["shapes"].append(shape)

        #Ghi ra file.json
        json_name = filename.replace(".jpg", ".json")
        json_path = os.path.join(output_folder, json_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(Anylabeling, f, indent=2, ensure_ascii=False)
        
    