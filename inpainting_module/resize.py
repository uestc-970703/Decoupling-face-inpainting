import os

import cv2
from PIL import Image

input_folder = 'test_sets/CelebA-HQ/images'
output_folder = 'resized_images'
target_size = (1920, 1013)

os.makedirs(output_folder, exist_ok=True)

imags_extensions = ['.jpg', 'jpeg', '.png']

for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in imags_extensions):
        try:
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            img_resized = img.resize(target_size, Image.ANTIALIAS)

            save_path = os.path.join(output_folder, filename)
            img_resized.save(save_path)

            print(f"已处理: {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")