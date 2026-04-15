import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import cv2
import torch
from collections import deque, Counter
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from models.cnn_lstm import CNN_LSTM

# ================= CONFIG =================
SEQ_LEN = 20
PRED_BUFFER_SIZE = 10
CONF_THRESHOLD = 0.4
TOP_K = 2  # jumlah kandidat tangan

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs/logs/cnn_lstm/run_20260409_111427/best_model.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LABEL =================
train_dir = os.path.join(BASE_DIR, "data/WLBisindo/split/train")
labels = sorted(os.listdir(train_dir))
label_map = {i: label for i, label in enumerate(labels)}
NUM_CLASSES = len(label_map)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= MODEL =================
model = CNN_LSTM(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# ================= BUFFER =================
frame_buffer = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=PRED_BUFFER_SIZE)

prev_gray = None
prev_boxes = []  # untuk tracking

# ================= HELPER =================
def smooth_box(prev_box, new_box, alpha=0.7):
    if prev_box is None:
        return new_box
    x = int(alpha * prev_box[0] + (1 - alpha) * new_box[0])
    y = int(alpha * prev_box[1] + (1 - alpha) * new_box[1])
    w = int(alpha * prev_box[2] + (1 - alpha) * new_box[2])
    h = int(alpha * prev_box[3] + (1 - alpha) * new_box[3])
    return (x, y, w, h)

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam error")
    exit()

print("🚀 Realtime Multi-Hand Tracking (ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes = []

    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ambil top-K kontur terbesar
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:TOP_K]

        for c in contours:
            area = cv2.contourArea(c)

            if area < 500 or area > 50000:
                continue

            x, y, w_box, h_box = cv2.boundingRect(c)

            # smooth tracking
            if len(prev_boxes) > 0:
                x, y, w_box, h_box = smooth_box(prev_boxes[0], (x, y, w_box, h_box))

            boxes.append((x, y, w_box, h_box))

            # draw bbox
            cv2.rectangle(frame, (x,y), (x+w_box,y+h_box), (0,255,0), 2)

    prev_gray = gray
    prev_boxes = boxes

    label_text = "Collecting..."

    # ================= PILIH 1 BOX TERBAIK =================
    if len(boxes) > 0:
        # ambil box terbesar
        x, y, w_box, h_box = max(boxes, key=lambda b: b[2]*b[3])

        crop = frame[y:y+h_box, x:x+w_box]

        if crop.size != 0:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img)

            frame_buffer.append(img)

    # ================= PREDICTION =================
    if len(frame_buffer) == SEQ_LEN:
        input_tensor = torch.stack(list(frame_buffer))
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        pred_buffer.append(pred)

        if len(pred_buffer) >= 5:
            final_pred = Counter(pred_buffer).most_common(1)[0][0]
        else:
            final_pred = pred

        if confidence < CONF_THRESHOLD:
            label_text = "..."
        else:
            label_text = f"{label_map.get(final_pred)} ({confidence:.2f})"

    # ================= DISPLAY =================
    cv2.putText(frame, f"Pred: {label_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Multi-Hand Tracking BISINDO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()