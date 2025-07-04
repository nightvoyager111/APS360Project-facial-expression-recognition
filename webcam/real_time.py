import cv2
import numpy as np
from joblib import load
from skimage.transform import resize
from skimage.feature import hog
from collections import defaultdict

# Load trained model
model_path = 'combined_unbalanced_hog_svm.joblib'  # Or 'FERPlus_hog_svm.joblib'
clf, scaler, emotion_labels = load(model_path)

# HOG parameters
HOG_PARAMS = {
    'orientations': 8,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Display settings
BOX_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2

# Emotion mapping
EMOTION_MAPPING = {
    'surprise': 'SURPRISED',
    'fear': 'FEARFUL',
    'disgust': 'DISGUSTED',
    'happiness': 'HAPPY',
    'sadness': 'SAD',
    'anger': 'ANGRY',
    'neutral': 'NEUTRAL'
}

# Initialize counters
emotion_counter = defaultdict(int)

# Video capture
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = resize(face_roi, (48, 48))
        hog_features = hog(face_resized, **HOG_PARAMS).reshape(1, -1)
        hog_features_scaled = scaler.transform(hog_features)

        pred = clf.predict(hog_features_scaled)
        emotion = emotion_labels[pred[0]]
        readable_emotion = EMOTION_MAPPING.get(emotion, emotion)
        emotion_counter[readable_emotion] += 1

        # Draw bounding box and label
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
            percent = (count / total_preds) * 100 if total_preds > 0 else 0
            bar_len = int((percent / 100) * (chart_width - 60))

            y = start_y + idx * (bar_height + gap)
            cv2.putText(frame, f"{emotion[:8]} {percent:4.1f}%", (base_x, y + 18),
                        TEXT_FONT, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (base_x + 100, y), (base_x + 100 + bar_len, y + bar_height),
                          (0, 255, 0), -1)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
