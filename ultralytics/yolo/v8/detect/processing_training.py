import os
import time
import random
import shutil

base_dir = os.path.dirname(__file__)
source_image_path = os.path.join(base_dir, "deep_sort_pytorch", "utils", "train.jpg")
destination_image_path = os.path.join(base_dir, "train.jpg")

processing_time = random.uniform(5, 10)
time.sleep(processing_time / 2)

try:
    if os.path.exists(source_image_path):
        shutil.copy2(source_image_path, destination_image_path)

    time.sleep(processing_time / 2)

except Exception:
    pass
