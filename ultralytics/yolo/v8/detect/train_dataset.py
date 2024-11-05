import os
import subprocess
import yaml
import torch
import torch.nn as nn

checkpoint_path = r"C:\Nam_Projects\AI_Learning\my-ml-project\apps\api-yolo8\YOLOv8-DeepSORT-Object-Tracking\ultralytics\yolo\v8\detect\deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7"
base_dir = os.path.dirname(__file__)
training_process_path = os.path.join(base_dir, "deep_sort_pytorch", "utils", "train_helper.py")
yaml_path = os.path.join(base_dir, "VehicalsYolo8.v5i.yolov8", "data.yaml")

try:
    with open(yaml_path, 'r') as file:
        data_config = yaml.safe_load(file)

    train_images_dir = data_config['train'] + '/images'
    train_labels_dir = data_config['train'] + '/labels'
    val_images_dir = data_config['val'] + '/images'
    val_labels_dir = data_config['val'] + '/labels'

    model = nn.Linear(10, 10)
    torch.save(model.state_dict(), checkpoint_path)

except Exception:
    pass

num_epochs = 40
subprocess.run(["python", training_process_path, str(num_epochs)], check=True)
