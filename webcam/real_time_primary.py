import cv2
import torch
import numpy as np
import sys
import os
from collections import defaultdict
from torchvision import transforms
from PIL import Image

# Append parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from primary_model.facial_expression_detection import EmotionAlexNet

# Load trained model


checkpoint = torch.load('primary_model/best_model_1.pth', map_location=torch.device('cpu'))
model = EmotionAlexNet(num_classes=7, use_residual=True)

# Filter out classifier parameters from checkpoint
filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('classifier.')}

# Load weights except the classifier
model.load_state_dict(filtered_state_dict, strict=False)

model.eval()


# Class names (must match training order)
emotion_classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# Mapping to uppercase display
EMOTION_MAPPING = {
    'anger': 'ANGRY',
    'disgust': 'DISGUSTED',
    'fear': 'FEARFUL',
    'happiness': 'HAPPY',
    'neutral': 'NEUTRAL',
    'sadness': 'SAD',
    'surprise': 'SURPRISED'
}

# Display settings
BOX_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2

# Preprocessing (match training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Initialize emotion count
emotion_counter = defaultdict(int)

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        pil_img = Image.fromarray(roi_gray)
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            raw_label = emotion_classes[pred]
            readable_emotion = EMOTION_MAPPING.get(raw_label, raw_label)
            emotion_counter[readable_emotion] += 1

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, 1)
        (text_w, text_h), _ = cv2.getTextSize(readable_emotion, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), BOX_COLOR, -1)
        cv2.putText(frame, readable_emotion, (x, y - 5), TEXT_FONT, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS)

    # Draw emotion distribution on right side
    total_preds = sum(emotion_counter.values())
    if total_preds > 0:
        h, w, _ = frame.shape
        chart_width = 180
        bar_height = 25
        gap = 8
        start_y = 30
        base_x = w - chart_width

        for idx, emotion in enumerate(EMOTION_MAPPING.values()):
            count = emotion_counter[emotion]
            percent = (count / total_preds) * 100
            bar_len = int((percent / 100) * (chart_width - 60))

            y = start_y + idx * (bar_height + gap)
            cv2.putText(frame, f"{emotion[:8]} {percent:4.1f}%", (base_x, y + 18),
                        TEXT_FONT, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (base_x + 100, y), (base_x + 100 + bar_len, y + bar_height),
                          (0, 255, 0), -1)

    cv2.imshow("Webcam Emotion Detection (CNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
