import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys, os
import cv2
from collections import defaultdict
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from primary_model.Isa_try2 import EmotionAlexNet

model = EmotionAlexNet(num_classes=7, use_residual=True)
ckpt = torch.load('./models/BEST_EmotionAlexNet_RAFDB_epoch18_20250810_012019.pt', map_location='cpu')
sd = ckpt.get('state_dict', ckpt)

first_conv_key = None
for k in sd.keys():
    if k.endswith('weight') and ('features.0' in k or 'conv1' in k or 'features.1' in k or 'features.2' in k or 'features' in k):
        first_conv_key = k
        break
    
    
if first_conv_key is None:
    W = sd[first_conv_key]
    first_conv_module = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    Cin_model = first_conv_module.in_channels
    Cin_ckpt = W.shape[1]
    if Cin_ckpt != Cin_model:
        if Cin_ckpt == 3 and Cin_model == 1:
            sd[first_conv_key] = W.mean(dim=1, keepdim=True)
        elif Cin_ckpt == 1 and Cin_model == 3:
            sd[first_conv_key] = W.repeat(1, 3, 1, 1)
        else:
            sd[first_conv_key] = W.repeat(1, Cin_model, 1, 1) / Cin_model

            
model.load_state_dict(sd, strict=False)
    
model.eval()

emotion_classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']   
EMOTION_MAPPING = {
    'anger': 'ANGRY',
    'disgust': 'DISGUSTED',
    'fear': 'FEARFUL',
    'happiness': 'HAPPY',
    'neutral': 'NEUTRAL',
    'sadness': 'SAD',
    'surprise': 'SURPRISED'
}

# Preprocessing (match training)
transform = transforms.Compose([
    #transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(pil_img):
    
    gray = np.array(pil_img.convert('L'))
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return {"error": "No face detected"}
    
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    
    face_pil = Image.fromarray(gray[y:y+h, x:x+w])
    input_tensor = transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.exp(output).squeeze().numpy()
        
        
    pred_idx = int(np.argmax(probs))
    emotion = emotion_classes[pred_idx]
    
    prob_dict = {label: float(f"{probs[i]:.3f}") for i, label in enumerate(emotion_classes)}
    return {
        "emotion": emotion.upper(),
        "probabilities": prob_dict,
        "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    }