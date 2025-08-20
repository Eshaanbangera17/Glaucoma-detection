import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2
import os
import numpy as np
from collections import OrderedDict

# --------------------
# Config
# --------------------
IMG_PATH = r"C:\Users\Eshaan\OneDrive\Desktop\Glaucoma detection\data\raw\glaucoma\drishtiGS_086.png"
MODEL_PATH = r"C:\Users\Eshaan\OneDrive\Desktop\Glaucoma detection\models\mobilenet_from_folders.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Preprocessing
# --------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# --------------------
# Load Model
# --------------------
print("Loading model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use width_mult=0.5 to match checkpoint (16 channels instead of 32)
model = models.mobilenet_v2(weights=None, width_mult=0.5)
model.classifier[1] = nn.Linear(model.last_channel, 2)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# If checkpoint has "state_dict" inside
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Clean keys (remove "module." if saved with DataParallel)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()
print("Model loaded successfully!")

# ------------------------
# Preprocessing pipeline
# ------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------
# Inference
# ------------------------
print(f"Loading image: {IMG_PATH}")
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inp = transform(img_rgb).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(inp)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

# ------------------------
# Results
# ------------------------
classes = ["Normal", "Glaucoma"]
pred_idx = np.argmax(probs)
pred_class = classes[pred_idx]
confidence = probs[pred_idx] * 100

print(f"Prediction: {pred_class} ({confidence:.2f}%)")

# ------------------------
# Display with OpenCV
# ------------------------
label = f"{pred_class} ({confidence:.1f}%)"
cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255) if pred_class == "Glaucoma" else (0, 255, 0), 2)
cv2.imshow("Fundus Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()