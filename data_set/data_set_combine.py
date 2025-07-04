import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import warnings


# --- CONFIGURATION ---
FERPLUS_TRAIN_DIR = 'fer2013plus/fer2013/train'
FERPLUS_TEST_DIR = 'fer2013plus/fer2013/test'
RAFDB_TRAIN_DIR = 'RAF-DB/train'
RAFDB_TEST_DIR = 'RAF-DB/test'
IMAGE_SIZE = (48, 48)
HOG_PARAMS = {
    'orientations': 8,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}


# Standardized emotion classes
STANDARD_EMOTIONS = {'happy': 'happiness', 'happiness': 'happiness', '4': 'happiness', 'sad': 'sadness', 'sadness': 'sadness', '5': 'sadness', 'fear': 'fear', '2': 'fear', 'disgust': 'disgust',
                    '3': 'disgust', 'angry': 'anger', 'anger': 'anger', '6': 'anger', 'neutral': 'neutral', '7': 'neutral', 'surprise': 'surprise', '1': 'surprise'}


def get_standard_classes():
    """Return only 7 standardized emotion classes"""
    return sorted(list(set(STANDARD_EMOTIONS.values())))


def standardize_emotion(label):
    """Convert any emotion label to standard 7 classes"""
    return STANDARD_EMOTIONS.get(label, None)  # Returns None for unknown emotions


def load_image_folder(data_dir, class_to_idx):
    """Load images while standardizing emotion labels"""
    X, y = [], []
    for orig_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, orig_label)
        if not os.path.isdir(class_dir):
            continue
           
        # Convert to standard emotion
        std_emotion = standardize_emotion(orig_label)
        if std_emotion is None:
            continue
           
        # Only process if this is one of 7 classes
        if std_emotion in class_to_idx:
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, fname)
                    img = imread(img_path, as_gray=True)
                    img = resize(img, IMAGE_SIZE)
                    X.append(img)
                    y.append(class_to_idx[std_emotion])
   
    return np.array(X), np.array(y)

def load_and_combine_datasets():
    combined_classes = get_standard_classes()
    class_to_idx = {cls: idx for idx, cls in enumerate(combined_classes)}
   
    # Load and combine training data
    X_fer_train, y_fer_train = load_image_folder(FERPLUS_TRAIN_DIR, class_to_idx)
    X_raf_train, y_raf_train = load_image_folder(RAFDB_TRAIN_DIR, class_to_idx)
    X_train = np.concatenate((X_fer_train, X_raf_train))
    y_train = np.concatenate((y_fer_train, y_raf_train))
   
    # Load and combine test data
    X_fer_test, y_fer_test = load_image_folder(FERPLUS_TEST_DIR, class_to_idx)
    X_raf_test, y_raf_test = load_image_folder(RAFDB_TEST_DIR, class_to_idx)
    X_test = np.concatenate((X_fer_test, X_raf_test))
    y_test = np.concatenate((y_fer_test, y_raf_test))


    return X_train, y_train, X_test, y_test, combined_classes

