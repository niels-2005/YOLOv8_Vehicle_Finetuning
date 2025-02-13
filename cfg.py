import os

class CFG:
    sample_img_path = "./src/sample_image.jpg"
    sample_video_path = "./src/sample_video.mp4"
    dataset_path = "./Vehicle_Detection_Image_Dataset/"
    img_train_path = os.path.join(dataset_path, "train", "images")
    img_valid_path = os.path.join(dataset_path, "valid", "images")
    yaml_file_path = os.path.join(dataset_path, "data.yaml")
    post_train_files_path = "./runs/detect/train"
    yolo_preset_name = "yolov8n.pt"
    file_format = ".jpg"
    imgsz = 640
    conf = 0.5
    device = 0
    epochs = 200
    patience = 50
    batch = 32
    optimizer = "auto"
    lr0 = 0.0001
    lrf = 0.1
    dropout = 0.1
    seed = 42