import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import os
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ====== HÀM LOAD DỮ LIỆU ======
def load_dataset_from_folder(folder_path, target_size=(28, 28)):
    images = []
    labels = []
    for label in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    img = Image.open(file_path).convert("L")  # grayscale
                    img = img.resize(target_size)
                    img = np.array(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Lỗi đọc ảnh: {file_path} ({e})")
    return np.array(images), np.array(labels)

# ====== 1. LOAD DỮ LIỆU ======
train_folder = "data/training_data"
test_folder = "data/testing_data"

x_train, y_train = load_dataset_from_folder(train_folder, target_size=(28, 28))
x_test, y_test = load_dataset_from_folder(test_folder, target_size=(28, 28))

print("Kích thước tập train:", x_train.shape, y_train.shape)
print("Kích thước tập test:", x_test.shape, y_test.shape)

# Lấy danh sách lớp & số lớp
classes = np.unique(y_train)
num_classes = len(classes)
print(f"Số lớp: {num_classes}")
print("Danh sách lớp:", classes)

# ====== 2. THỐNG KÊ SỐ ẢNH MỖI LỚP ======
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].bar(unique_train, counts_train, color='skyblue')
axes[0].set_title('Số lượng ảnh mỗi nhãn (Train)')
axes[0].set_xlabel('Nhãn')
axes[0].set_ylabel('Số lượng ảnh')
axes[0].grid(axis='y', linestyle='--')

axes[1].bar(unique_test, counts_test, color='lightcoral')
axes[1].set_title('Số lượng ảnh mỗi nhãn (Test)')
axes[1].set_xlabel('Nhãn')
axes[1].set_ylabel('Số lượng ảnh')
axes[1].grid(axis='y', linestyle='--')

plt.tight_layout()
plt.show()

# ====== 3. HIỂN THỊ ẢNH NGẪU NHIÊN ======
def plot_random_validation_images(x_data, y_data, num_images=5):
    random_indices = np.random.choice(len(x_data), num_images, replace=False)
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Hiển thị ngẫu nhiên {num_images} ảnh', fontsize=16)
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(x_data[idx], cmap='gray')
        plt.title(f'Nhãn: {y_data[idx]}')
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_random_validation_images(x_test, y_test, num_images=5)

# ====== 4. XÂY DỰNG MÔ HÌNH CNN ======
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Tự động theo số lớp
])

model.summary()

# ====== 5. CHUẨN HÓA DỮ LIỆU ======
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Map nhãn thành số để train
label_to_index = {label: idx for idx, label in enumerate(classes)}
y_train_num = np.array([label_to_index[label] for label in y_train])
y_test_num = np.array([label_to_index[label] for label in y_test])

# ====== 6. COMPILE & TRAIN ======
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_num, epochs=10, batch_size=32,
                    validation_data=(x_test, y_test_num))

model.save("model.h5")

# Trong train.py sau khi train xong
import json
with open("labels.json", "w") as f:
    json.dump(list(label_to_index.keys()), f)

# ====== 7. VẼ LOSS & ACCURACY ======
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ====== 8. MA TRẬN NHẦM LẪN ======
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test_num, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Nhãn Dự đoán')
plt.ylabel('Nhãn Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

# ====== 9. BÁO CÁO CHI TIẾT ======
print("\n--- Bảng kết quả đánh giá hiệu suất của mô hình ---")
report = classification_report(y_test_num, y_pred, target_names=classes)
print(report)
