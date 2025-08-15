import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

# ---- CONFIG ----
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"
CANVAS_SIZE = 280   # pixels for drawing area (square)
DRAW_WIDTH = 18     # stroke width for drawing on canvas
BG_COLOR = "#f3f4f6"    # window bg
CANVAS_BG = "white"     # drawing bg

# ---- Load model and labels (with error handling) ----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH} - đặt file model.h5 cùng thư mục với script.")

model = load_model(MODEL_PATH)

labels = None
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    # warning but not fatal
    print("Cảnh báo: labels.json không tìm thấy. Ứng dụng sẽ hiển thị index lớp thay vì nhãn.")

# Get model input shape: expect (None, H, W, C)
inp = model.input_shape
# support tuple like (None, H, W, C)
if len(inp) == 4:
    _, MODEL_H, MODEL_W, MODEL_C = inp
else:
    # fallback
    MODEL_H = MODEL_W = 28
    MODEL_C = 1

MODEL_H = int(MODEL_H) if MODEL_H is not None else 28
MODEL_W = int(MODEL_W) if MODEL_W is not None else 28
MODEL_C = int(MODEL_C) if MODEL_C is not None else 1

# ---- GUI ----
root = tk.Tk()
root.title("Handwritten Character Recognizer")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# Main frame
main_frame = ttk.Frame(root, padding=12)
main_frame.grid(row=0, column=0)

style = ttk.Style(root)
style.theme_use('clam')

# Left: drawing area
left_frame = ttk.Frame(main_frame)
left_frame.grid(row=0, column=0, padx=(0,12))

canvas_frame = tk.Frame(left_frame, bg="#e5e7eb", bd=0)
canvas_frame.pack()

tk.Label(left_frame, text="Vẽ ký tự ở đây", font=("Helvetica", 12, "bold"), bg=BG_COLOR).pack(pady=(6,4))

canvas_widget = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=CANVAS_BG, bd=0, highlightthickness=0)
canvas_widget.pack(padx=6, pady=6)

# PIL image for drawing (keeps pixel-perfect)
pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)  # 'L' = grayscale
draw = ImageDraw.Draw(pil_image)

# Right: controls + result
right_frame = ttk.Frame(main_frame)
right_frame.grid(row=0, column=1, sticky="n")

result_frame = ttk.Frame(right_frame, padding=(6,6))
result_frame.pack()

lbl_title = tk.Label(result_frame, text="Kết quả dự đoán", font=("Helvetica", 14, "bold"), bg=BG_COLOR)
lbl_title.pack(pady=(2,8))

pred_label = tk.Label(result_frame, text="?", font=("Helvetica", 96, "bold"), fg="#0b6ef6", bg=BG_COLOR)
pred_label.pack()

conf_label = tk.Label(result_frame, text="Độ tin cậy: --%", font=("Helvetica", 12), bg=BG_COLOR)
conf_label.pack(pady=(6,14))

# Buttons
btn_frame = ttk.Frame(right_frame)
btn_frame.pack(pady=8, fill="x")

def clear_canvas():
    canvas_widget.delete("all")
    global pil_image, draw
    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
    draw = ImageDraw.Draw(pil_image)
    pred_label.config(text="?")
    conf_label.config(text="Độ tin cậy: --%")

def predict_from_canvas():
    # resize to model input, preprocess depending on channels
    img = pil_image.copy()
    img = img.resize((MODEL_W, MODEL_H), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0   # shape (H,W)
    if MODEL_C == 1:
        arr = arr.reshape(1, MODEL_H, MODEL_W, 1)
    else:
        # stack gray->3channels
        arr = np.stack([arr]*3, axis=-1)
        arr = arr.reshape(1, MODEL_H, MODEL_W, 3)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100
    label_text = labels[idx] if labels and idx < len(labels) else str(idx)
    pred_label.config(text=str(label_text))
    conf_label.config(text=f"Độ tin cậy: {conf:.1f}%")

btn_predict = ttk.Button(btn_frame, text="Dự đoán", command=predict_from_canvas)
btn_predict.pack(side="left", padx=6, ipadx=8)

btn_clear = ttk.Button(btn_frame, text="Xóa", command=clear_canvas)
btn_clear.pack(side="left", padx=6, ipadx=8)

def on_quit():
    if messagebox.askokcancel("Thoát", "Bạn có muốn thoát?"):
        root.destroy()

btn_quit = ttk.Button(btn_frame, text="Thoát", command=on_quit)
btn_quit.pack(side="left", padx=6, ipadx=8)

# Draw handlers
is_drawing = False
last_x = last_y = None

def start_draw(event):
    global is_drawing, last_x, last_y
    is_drawing = True
    last_x, last_y = event.x, event.y

def draw_motion(event):
    global is_drawing, last_x, last_y
    if not is_drawing:
        return
    x, y = event.x, event.y
    # draw on Tk canvas
    canvas_widget.create_line(last_x, last_y, x, y, width=DRAW_WIDTH, fill="black", capstyle=tk.ROUND, smooth=True)
    # draw on PIL image (use same coords)
    # convert canvas coords to PIL coords (1:1 since sizes match)
    draw.line([last_x, last_y, x, y], fill=0, width=DRAW_WIDTH)
    last_x, last_y = x, y

def end_draw(event):
    global is_drawing
    is_drawing = False

# Bind mouse events
canvas_widget.bind("<ButtonPress-1>", start_draw)
canvas_widget.bind("<B1-Motion>", draw_motion)
canvas_widget.bind("<ButtonRelease-1>", end_draw)

# Keyboard shortcuts
def on_key(event):
    k = event.keysym.lower()
    if k == "c":
        clear_canvas()
    elif k == "s":
        predict_from_canvas()
    elif k == "q" or k == "escape":
        on_quit()

root.bind_all("<Key>", on_key)

# Footer help text
help_lbl = tk.Label(root, text="Phím tắt: S = Dự đoán, C = Xóa, Q = Thoát", font=("Helvetica", 9), bg=BG_COLOR)
help_lbl.grid(row=1, column=0, pady=(6,10))

# Start
clear_canvas()
root.mainloop()
