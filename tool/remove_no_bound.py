import json
import os

dataset_dir = r'C:\99.workspace\Project_2026\13.helmet_detection\dataset_1\anylabeling'

def remove_no_bound():
    total = len(os.listdir(dataset_dir))
    count_deleted_json = 0
    

    print(f"Total files: {total}")
    print("Bat dau xoa file khong co bounding box")
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dataset_dir, filename)
            txt_path = json_path.replace('.json', '.txt')
            jpg_path = json_path.replace('.json', '.jpg')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            shapes = data.get('shapes',[])
            if len(shapes) == 0:
                with open("nhat_ky_xoa.txt", "a") as log_file:
                    log_file.write(f"Đã xóa vĩnh viễn: {filename}\n")
                print(f"Xoa file: {filename}")
                os.remove(json_path)
                count_deleted_json += 1
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
    print(count_deleted_json)
if __name__ == '__main__':
    remove_no_bound()

                