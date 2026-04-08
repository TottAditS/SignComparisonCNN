import sys
import os

# ================= FIX PATH =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# ================= IMPORT =================
import cv2
import torch
from collections import deque
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_lstm import CNN_LSTM

# ================= CONFIG =================
SEQ_LEN = 20

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs/logs/cnn_lstm/run_20260408_235004/best_model.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD LABEL MAP (DARI TRAIN FOLDER) =================
train_dir = os.path.join(BASE_DIR, "data/WLBisindo/split/train")

labels = sorted(os.listdir(train_dir))
label_map = {i: label for i, label in enumerate(labels)}

NUM_CLASSES = len(label_map)

print("NUM_CLASSES:", NUM_CLASSES)
print("Sample labels:", list(label_map.items())[:5])

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD MODEL =================
model = CNN_LSTM(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# ================= FRAME BUFFER =================
frame_buffer = deque(maxlen=SEQ_LEN)
PRED_BUFFER_SIZE = 10
pred_buffer = deque(maxlen=PRED_BUFFER_SIZE)

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam tidak bisa dibuka")
    exit()

print("🚀 Realtime started (ESC untuk keluar)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== PREPROCESS =====
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)

    frame_buffer.append(img)

    label_text = "Collecting..."

    # ===== PREDICTION =====
    if len(frame_buffer) == SEQ_LEN:
        input_tensor = torch.stack(list(frame_buffer))
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        pred_buffer.append(pred)

        # majority vote
        most_common = Counter(pred_buffer).most_common(1)[0][0]

        if confidence < 0.6:
            label_text = "..."
        else:
            label_text = f"{label_map.get(most_common)} ({confidence:.2f})"

    # ===== DISPLAY =====
    cv2.putText(frame, f"Pred: {label_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Realtime BISINDO", frame)

    # ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()