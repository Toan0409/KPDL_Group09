#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation for Handwriting Recognition - Tăng cường dữ liệu tự động
Chạy: python augment_handwriting.py --input dataset --output dataset_aug --num 5
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import argparse
from tqdm import tqdm
import random
import uuid


def augment_image(img):
    """Áp dụng nhiều phép biến đổi để làm giàu dữ liệu chữ viết tay"""
    # 1. Xoay ngẫu nhiên (-7° đến +7°)
    angle = np.random.uniform(-7, 7)
    img = img.rotate(angle, fillcolor=255)

    # 2. Biến dạng affine (shear theo X và Y)
    width, height = img.size
    dx = np.random.uniform(-0.1, 0.1)  # shear X
    dy = np.random.uniform(-0.1, 0.1)  # shear Y
    img = img.transform(
        (width, height),
        Image.AFFINE,
        (1, dx, 0, dy, 1, 0),
        fillcolor=255
    )

    # 3. Thay đổi độ sáng và độ tương phản
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(np.random.uniform(0.8, 1.3))

    # 4. Gaussian Blur nhẹ
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.2)))

    # 5. Thêm nhiễu Gaussian
    img_array = np.array(img)
    sigma = np.random.uniform(2, 10)
    noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
    img_array = np.clip(img_array + noise, 0, 255)
    img = Image.fromarray(img_array.astype(np.uint8))

    # 6. Random crop & pad (dời chữ lệch vị trí)
    if random.random() < 0.5:
        crop_x = np.random.randint(0, width // 20 + 1)
        crop_y = np.random.randint(0, height // 20 + 1)
        img = img.crop((crop_x, crop_y, width - crop_x, height - crop_y))
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_data", help="Thư mục dataset gốc")
    parser.add_argument("--output", default="dataset_aug", help="Thư mục dataset tăng cường")
    parser.add_argument("--num", type=int, default=5, help="Số ảnh tăng cường/ảnh gốc")
    args = parser.parse_args()

    # Tạo thư mục đầu ra
    os.makedirs(args.output, exist_ok=True)

    # Duyệt qua tất cả nhãn (thư mục con)
    for label in os.listdir(args.input):
        label_path = os.path.join(args.input, label)
        if not os.path.isdir(label_path):
            continue

        # Tạo thư mục cho nhãn trong dataset_aug
        output_label = os.path.join(args.output, label)
        os.makedirs(output_label, exist_ok=True)

        # Xử lý từng ảnh
        for img_file in tqdm(os.listdir(label_path), desc=f"Xử lý nhãn '{label}'"):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(label_path, img_file)
            base_name = os.path.splitext(img_file)[0]

            try:
                # Mở ảnh gốc (luôn chuyển sang grayscale)
                img = Image.open(img_path).convert("L")

                # Copy ảnh gốc vào dataset_aug
                img.save(os.path.join(output_label, f"{base_name}_orig.png"))

                # Tạo ảnh tăng cường
                for i in range(args.num):
                    aug_img = augment_image(img)
                    aug_img.save(
                        os.path.join(output_label, f"{base_name}_aug_{i}_{uuid.uuid4().hex[:6]}.png"),
                        format="PNG"
                    )
            except Exception as e:
                print(f"❌ Lỗi với {img_file}: {str(e)}")


if __name__ == "__main__":
    main()
