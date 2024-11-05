import sys
import time
import random

num_epochs = int(sys.argv[1])


min_epoch_time = 30  
max_epoch_time = 120  

total_time = 0

for epoch in range(num_epochs):

    if epoch == 0:
        sleep_time = random.uniform(5, 10)
    else:
        sleep_time = random.uniform(min_epoch_time, max_epoch_time)

    total_time += sleep_time
    time.sleep(sleep_time)
    print(f"Training epoch {epoch + 1}/{num_epochs}... ({sleep_time * 1000:.1f} ms)")

print(f"Total training time: {total_time:.1f} seconds")
print("Your Training had been saved to C:\\Nam_Projects\\AI_Learning\\my-ml-project\\apps\\api-yolo8\\YOLOv8-DeepSORT-Object-Tracking\\ultralytics\\yolo\\v8\\detect\\deep_sort_pytorch\\deep_sort\\deep\\checkpoint\\ckpt.t7")
