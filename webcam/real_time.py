import cv2
import numpy as np
from joblib import load
from skimage.transform import resize
from skimage.feature import hog

# Load trained model
model_path = 'RAF-DB_hog_svm.joblib'  # Can also use 'FERPlus_hog_svm.joblib'
clf, scaler, emotion_labels = load(model_path)

# HOG parameters (must match training settings)
HOG_PARAMS = {
    'orientations': 8,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Display settings
BOX_COLOR = (0, 255, 255)  
BOX_THICKNESS = 1         
TEXT_COLOR = (0, 255, 255)   
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2
EMOTION_MAPPING = {         # Custom emotion word mapping
    'surprise': 'SURPRISED',
    'fear': 'FEARFUL',
    'disgust': 'DISGUSTED',
    'happiness': 'HAPPY',
    'sadness': 'SAD',
    'anger': 'ANGRY',
    'neutral': 'NEUTRAL'
}

# Video capture
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        # Extract and preprocess face region
        face_roi = gray[y:y+h, x:x+w]
        face_resized = resize(face_roi, (48, 48))  # Match training size
        
        # Extract HOG features
        hog_features = hog(face_resized, **HOG_PARAMS)
        hog_features = hog_features.reshape(1, -1)
        
        # Standardize features
        hog_features_scaled = scaler.transform(hog_features)
        
        # Predict emotion
        pred = clf.predict(hog_features_scaled)
        emotion = EMOTION_MAPPING.get(emotion_labels[pred[0]], emotion_labels[pred[0]])
        
        # Display results with customized appearance
        cv2.rectangle(frame, (x,y), (x+w,y+h), BOX_COLOR, BOX_THICKNESS)
        
        # Calculate text size for better placement
        (text_width, text_height), _ = cv2.getTextSize(emotion, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
        cv2.rectangle(frame, 
                     (x, y - text_height - 10), 
                     (x + text_width, y), 
                     BOX_COLOR, -1)  # Filled rectangle background
        cv2.putText(frame, emotion, 
                   (x, y - 5),  # Adjusted vertical position
                   TEXT_FONT, TEXT_SCALE, 
                   (0, 0, 0),  # Black text for better visibility
                   TEXT_THICKNESS)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()