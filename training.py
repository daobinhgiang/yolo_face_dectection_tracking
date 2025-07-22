import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ultralytics import YOLO
import yaml
import cv2


data_dict = {
    # "train": "/kaggle/input/face-detection-dataset/images/train",
    # "val":   "/kaggle/input/face-detection-dataset/images/val",
    "train":"/home/daobinhgiang/projects/object_dect+tracking/images/train",
    "val":"/home/daobinhgiang/projects/object_dect+tracking/images/val",
    "nc":    1,
    "names": ["face"]
}

# model = YOLO("yolo11n.pt")

# Dump to a temp YAML
with open("temp_data.yaml", "w") as f:
    yaml.safe_dump(data_dict, f)


def train_and_save_yolov11(
        data_yaml: str,
        model_arch: str = "",
        epochs: int = 100,
        imgsz: int = 640,
        save_dir: str = "runs/train/yolov11_custom"
):
    """
    Train a YOLOv11 model and save the trained weights.

    Args:
        data_yaml (str): Path to the dataset YAML file.
        model_arch (str): Pretrained model architecture or .pt file.
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
        save_dir (str): Directory to save results and weights.
    """
    best_model_path = save_dir
    model = YOLO(model_arch)  # Load YOLOv11 architecture or checkpoint
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=save_dir,
        exist_ok=True
    )
    # Save the best model weights
    model.save(best_model_path)
    print(f"Training complete. Best model saved to {best_model_path}.")
    return best_model_path


best_model = train_and_save_yolov11(
    data_yaml="temp_data.yaml",
    # model_arch="yolo11n.pt",
    epochs=50,
    imgsz=640,
    save_dir=""
)