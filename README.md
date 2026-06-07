#  Nhận Dạng Màu Sắc trong Điều Kiện Ánh Sáng Thay Đổi

> **Machine Learning & Robotics Project (A4)**  
> Phân loại 6 màu sắc cơ bản bằng CNN với độ chính xác **93.23%**

---

##  Mô Tả

Robot nhận dạng màu sắc của vật thể trong điều kiện ánh sáng thay đổi và phân loại thành 6 nhóm màu. Thay vì dùng ngưỡng HSV cố định (dễ sai khi ánh sáng thay đổi), dự án sử dụng **CNN** để học đặc trưng màu sắc bền vững hơn.

| Nhãn | Màu |
|------|-----|
| 1 |  Đỏ (Red) |
| 2 |  Cam (Orange) |
| 3 |  Vàng (Yellow) |
| 4 |  Xanh lá (Green) |
| 5 |  Xanh dương (Blue) |
| 6 |  Hồng (Pink) |

---

##  Cấu Trúc Dataset

Dataset được **sinh tự động bằng Python** (không cần chụp ảnh thực tế), mô phỏng 10 điều kiện ánh sáng:

```
Normal | Dark | Bright | Warm | Cold | Dim | Neon | Shadow | Sunset | Flash
```

```
dataset/
├── red/        # 1,000 ảnh
├── orange/     # 1,000 ảnh
├── yellow/     # 1,000 ảnh
├── green/      # 1,000 ảnh
├── blue/       # 1,000 ảnh
└── pink/       # 1,000 ảnh
                # Tổng: 6,000 ảnh
```

**Phân chia:** Train 80% (4,800 ảnh) | Validation 20% (1,200 ảnh)

---

##  Kiến Trúc Mô Hình (CNN)

```
Input (64x64x3)
    │
    ▼
Conv2D(32) + BatchNorm + MaxPool   → Đặc trưng màu cơ bản
    │
    ▼
Conv2D(64) + BatchNorm + MaxPool   → Đặc trưng màu sâu hơn
    │
    ▼
Conv2D(128) + BatchNorm + MaxPool  → Đặc trưng màu phức tạp
    │
    ▼
Flatten → Dense(128, ReLU) + Dropout(0.3)
    │
    ▼
Dense(6, Softmax) → Output (6 classes)
```

| Tham số | Giá trị |
|---------|---------|
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam (lr=0.001) |
| Batch Size | 32 |
| Epochs | 27 (Early Stopping) |

---

##  Kết Quả

### Độ Chính Xác Tổng Thể

| Tập dữ liệu | Accuracy | Loss |
|-------------|----------|------|
| Train | ~91% | ~0.20 |
| Validation | **93.23%** | ~0.19 |

>  Validation accuracy cao hơn train → mô hình **không bị overfitting**

### Kết Quả Theo Từng Màu

| Màu | Precision | Recall | F1-Score | Ghi chú |
|-----|-----------|--------|----------|---------|
| Red | 0.88 | 0.81 | 0.84 | Dễ nhầm với Pink |
| Orange | 0.98 | 0.94 | 0.96 | Rất tốt |
| Yellow | 0.94 | 0.98 | 0.96 | Rất tốt |
| Green | 0.99 | 0.94 | 0.97 | Xuất sắc |
| Blue | 1.00 | 1.00 | **1.00** | Hoàn hảo ✨ |
| Pink | 0.82 | 0.92 | 0.86 | Dễ nhầm với Red |
| **Avg** | **0.93** | **0.93** | **0.93** | Tổng thể tốt |

---

##  Demo Real-Time (Mô Phỏng Robot)

Pipeline xử lý trên robot (Python + OpenCV):

```
Camera → Resize (64x64) → Normalize [0,1] → CNN → Tên màu + Confidence%
```

Yêu cầu: Xử lý < **0.5 giây/ảnh**

---

## ⚙️ Cài Đặt & Chạy

```bash
# Clone repo
git clone https://github.com/yourusername/color-recognition
cd color-recognition

# Cài dependencies
pip install tensorflow opencv-python numpy matplotlib

# Sinh dataset
python generate_dataset.py

# Huấn luyện model
python train.py

# Chạy demo webcam
python demo.py
# Nhấn Q để thoát
```

---

##  Kết Luận

**Đã đạt được:**
- Pipeline hoàn chỉnh: sinh dataset → train → demo real-time
- Validation Accuracy **93.23%**, vượt mục tiêu đề ra (85%)
- Bền vững hơn ngưỡng HSV cố định trong điều kiện ánh sáng thay đổi

**Hạn chế:**
- Red & Pink dễ nhầm lẫn (~80–88%) do Hue gần nhau trong HSV
- Dataset synthetic chưa phản ánh hoàn toàn điều kiện thực tế
- Chỉ nhận diện 6 màu cơ bản

**Hướng phát triển:**
- Mở rộng lên 10 màu (thêm cyan, purple, white, black)
- Bổ sung ảnh thực tế vào dataset
- Tích hợp robot phân loại sản phẩm trên băng chuyền

