#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation for Handwriting Recognition - Tăng cường dữ liệu tự động
Chạy: python augment_handwriting.py
"""
import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def augment_image(img):
    """Áp dụng các biến đổi đơn giản phù hợp cho chữ viết tay"""
    # 1. Xoay ngẫu nhiên (-5° đến +5°)
    angle = np.random.uniform(-5, 5)
    img = img.rotate(angle, fillcolor=255)

    # 2. Thêm nhiễu Gaussian nhẹ
    img_array = np.array(img)
    noise = np.random.normal(0, 2, img_array.shape).astype(np.float32)
    img_array = np.clip(img_array + noise, 0, 255)
    img = Image.fromarray(img_array.astype(np.uint8))

    # 3. Biến dạng nhẹ (shear)
    width, height = img.size
    dx = np.random.randint(-3, 3)
    img = img.transform((width, height), Image.AFFINE, (1, dx * 0.01, 0, 0, 1, 0), fillcolor=255)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset", help="Thư mục dataset gốc")
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

                # Tạo ảnh tăng cường
                for i in range(args.num):
                    aug_img = augment_image(img)
                    aug_img.save(
                        os.path.join(output_label, f"{base_name}_aug{i}.png"),
                        format="PNG"
                    )
            except Exception as e:
                print(f"❌ Lỗi với {img_file}: {str(e)}")


if __name__ == "__main__":
    main()