
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. XÂY MODEL ĐÚNG CẤU TRÚC ---
def build_model(num_classes=2):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

# --- 2. LOAD MODEL ---
model = build_model(num_classes=2)
state_dict = torch.load("models/resnet50_leaf_disease_final.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# --- 3. TRANSFORM ĐÚNG NHƯ LÚC TRAIN (128x128 + Normalize) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # ĐÚNG: 128, không phải 224
    transforms.ToTensor(),
    transforms.Normalize(                   # ĐÚNG: chuẩn hóa như lúc train
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 4. DỰ ĐOÁN ẢNH ---
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # [1, 3, 128, 128]
    
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_class].item()
    
    # Hiển thị ảnh + kết quả
    plt.figure(figsize=(6,5))
    plt.imshow(img)
    plt.title(f"DỰ ĐOÁN: {'KHỎE MẠNH' if pred_class == 0 else 'BỊ BỆNH'}\n"
              f"Độ tin cậy: {confidence:.1%}", fontsize=14, color='green' if pred_class == 0 else 'red')
    plt.axis('off')
    plt.show()
    
    return pred_class, confidence

# --- 5. CHẠY DỰ ĐOÁN ---
image_path = "test_img/diseases3.png"  # Thay bằng ảnh bạn muốn
pred, conf = predict_image(image_path)
print(f"\nKẾT QUẢ: LÁ {'KHỎE' if pred == 0 else 'BỊ BỆNH'} (Độ tin cậy: {conf:.1%})")