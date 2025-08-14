import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys, os
import cv2
from collections import defaultdict
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from primary_model.Isa_amend1 import EmotionAlexNet

model = EmotionAlexNet(num_classes=7, use_residual=True)
ckpt = torch.load('./models/BEST_EmotionAlexNet_RAFDB_epoch18_20250810_012019.pt', map_location='cpu')
sd = ckpt.get('state_dict', ckpt)
            
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
    pil_rgb = pil_img.convert('RGB')
    rgb = np.array(pil_rgb) 
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return {"error": "No face detected"}
    
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    
    face_rgb = rgb[y:y+h, x:x+w] 
    face_pil = Image.fromarray(face_rgb)

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