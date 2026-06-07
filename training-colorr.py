import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

IMG_SIZE = 64
NUM_CLASSES = 6
SAMPLES_PER_CLASS = 800
EPOCHS = 30
BATCH_SIZE = 32

class_names = ["red", "orange", "yellow", "green", "blue", "pink"]

COLOR_RANGES = {
    "red":    [(0,   160, 100), (10,  255, 255)],
    "orange": [(10,  150, 100), (25,  255, 255)],
    "yellow": [(25,  150, 100), (38,  255, 255)],
    "green":  [(38,  100, 80),  (85,  255, 255)],
    "blue":   [(100, 100, 80),  (130, 255, 255)],
    "pink":   [(0,   80,  150), (10,  180, 255)],
}

lighting_conditions = [
    {"name": "normal",  "v_scale": 1.0, "s_scale": 1.0, "tint": (0,  0,  0)},
    {"name": "dark",    "v_scale": 0.4, "s_scale": 0.9, "tint": (0,  0,  0)},
    {"name": "dark2",   "v_scale": 0.6, "s_scale": 0.9, "tint": (0,  0,  0)},
    {"name": "bright",  "v_scale": 1.3, "s_scale": 0.8, "tint": (0,  0,  0)},
    {"name": "bright2", "v_scale": 1.5, "s_scale": 0.7, "tint": (0,  0,  0)},
    {"name": "warm",    "v_scale": 0.9, "s_scale": 1.0, "tint": (0,  10, 20)},
    {"name": "warm2",   "v_scale": 1.0, "s_scale": 1.0, "tint": (0,  20, 40)},
    {"name": "cold",    "v_scale": 0.9, "s_scale": 1.0, "tint": (15, 5,  0)},
    {"name": "cold2",   "v_scale": 1.0, "s_scale": 1.0, "tint": (30, 10, 0)},
    {"name": "dim",     "v_scale": 0.7, "s_scale": 0.8, "tint": (0,  0,  0)},
]

SAMPLES_PER_LIGHT = SAMPLES_PER_CLASS // len(lighting_conditions)

print(f"Dang tao dataset: {SAMPLES_PER_CLASS} anh/mau x {NUM_CLASSES} mau = {SAMPLES_PER_CLASS * NUM_CLASSES} anh tong...")

X, y = [], []

for class_idx, cls in enumerate(class_names):
    h_min, s_min, v_min = COLOR_RANGES[cls][0]
    h_max, s_max, v_max = COLOR_RANGES[cls][1]

    for light in lighting_conditions:
        for _ in range(SAMPLES_PER_LIGHT):
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            h = np.random.randint(h_min, h_max + 1)
            s = int(np.clip(np.random.randint(s_min, s_max + 1) * light["s_scale"], 0, 255))
            v = int(np.clip(np.random.randint(v_min, v_max + 1) * light["v_scale"], 0, 255))

            img[:, :, 0] = h
            img[:, :, 1] = s
            img[:, :, 2] = v

            noise_h = np.random.randint(-5, 6,  (IMG_SIZE, IMG_SIZE), dtype=np.int16)
            noise_s = np.random.randint(-30, 31, (IMG_SIZE, IMG_SIZE), dtype=np.int16)
            noise_v = np.random.randint(-40, 41, (IMG_SIZE, IMG_SIZE), dtype=np.int16)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + noise_h, h_min, h_max).astype(np.uint8)
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + noise_s, 0, 255).astype(np.uint8)
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + noise_v, 0, 255).astype(np.uint8)

            bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            tint = np.array(light["tint"], dtype=np.int16)
            bgr = np.clip(bgr.astype(np.int16) + tint, 0, 255).astype(np.uint8)

            gradient = np.linspace(0.8, 1.0, IMG_SIZE, dtype=np.float32)
            bgr = (bgr.astype(np.float32) * gradient[np.newaxis, :, np.newaxis]).astype(np.uint8)
            bgr = cv2.GaussianBlur(bgr, (3, 3), 0)

            X.append(bgr)
            y.append(class_idx)

    print(f"  {cls}: {SAMPLES_PER_LIGHT * len(lighting_conditions)} anh done")

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)
y_cat = to_categorical(y, NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nBat dau train...\n")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("model/best_color_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model.save("model/color_classifier.h5")
print("\nModel saved!")

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n{'='*50}")
print(f"VALIDATION ACCURACY: {acc*100:.2f}%")
print(f"{'='*50}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
y_true = np.argmax(y_val, axis=1)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
plt.ylabel("Thuc te")
plt.xlabel("Du doan")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train", marker="o")
plt.plot(history.history["val_accuracy"], label="Validation", marker="o")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train", marker="o")
plt.plot(history.history["val_loss"], label="Validation", marker="o")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/training_history.png")
plt.close()

print("\nBieu do: results/training_history.png")
print("Confusion matrix: results/confusion_matrix.png")
print("\nTraining completed!")