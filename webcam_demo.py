import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/color_classifier.h5")

class_names = ["red", "orange", "yellow", "green", "blue", "pink"]

display_colors = {
    "red":    (0,   0,   220),
    "orange": (0,   140, 255),
    "yellow": (0,   220, 220),
    "green":  (0,   200, 0),
    "blue":   (220, 80,  0),
    "pink":   (180, 105, 255),
}

CONFIDENCE_THRESHOLD = 75

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc webcam!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Nhan Q de thoat")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    roi_size = 200
    cx, cy = w // 2, h // 2
    x1, y1 = cx - roi_size // 2, cy - roi_size // 2
    x2, y2 = cx + roi_size // 2, cy + roi_size // 2

    roi = frame[y1:y2, x1:x2]

    img = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id] * 100

    if confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[class_id]
    else:
        label = "Unknown"

    color = display_colors.get(label, (128, 128, 128))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    corner_len = 20
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * corner_len, py), color, 5)
        cv2.line(frame, (px, py), (px, py + dy * corner_len), color, 5)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, label.upper(), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    bar_w = 400
    bar_fill = int((confidence / 100) * bar_w)
    cv2.rectangle(frame, (20, 125), (20 + bar_w, 148), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 125), (20 + bar_fill, 148), color, -1)

    tx = 20 + int((CONFIDENCE_THRESHOLD / 100) * bar_w)
    cv2.line(frame, (tx, 120), (tx, 153), (255, 255, 255), 2)
    cv2.putText(frame, f"{CONFIDENCE_THRESHOLD}%", (tx - 18, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (cls, score) in enumerate(zip(class_names, prediction)):
        c = display_colors[cls]
        text = f"{cls:<8} {score*100:5.1f}%"
        cv2.putText(frame, text, (w - 260, 40 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)

    cv2.putText(frame, "Dat vat vao vung giua  |  Q = thoat",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Color Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()