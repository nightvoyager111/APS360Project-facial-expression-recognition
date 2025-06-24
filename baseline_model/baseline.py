import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
FERPLUS_TRAIN_DIR = 'fer2013plus/fer2013/train'
FERPLUS_TEST_DIR = 'fer2013plus/fer2013/test'
RAFDB_TRAIN_DIR = 'RAF-DB/train'
RAFDB_TEST_DIR = 'RAF-DB/test'
IMAGE_SIZE = (48, 48)  # Common size for facial expression datasets
HOG_PARAMS = {
    'orientations': 8,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

def get_classes(data_dir):
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def load_image_folder(data_dir, classes, class_to_idx):
    X, y = [], []
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                img = imread(img_path, as_gray=True)
                img = resize(img, IMAGE_SIZE)
                X.append(img)
                y.append(class_to_idx[class_name])
    return np.array(X), np.array(y)

def extract_hog_features(images):
    features = []
    for img in images:
        hog_feat = hog(img, **HOG_PARAMS)
        features.append(hog_feat)
    return np.array(features)

def evaluate_dataset(train_dir, test_dir, dataset_name):
    print(f"\n===== {dataset_name} =====")
    classes = get_classes(train_dir)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    print("Loading training data...")
    X_train_img, y_train = load_image_folder(train_dir, classes, class_to_idx)
    print(f"Loaded {len(X_train_img)} training images.")

    print("Loading test data...")
    X_test_img, y_test = load_image_folder(test_dir, classes, class_to_idx)
    print(f"Loaded {len(X_test_img)} test images.")

    print("Extracting HOG features for training...")
    X_train = extract_hog_features(X_train_img)
    print("Extracting HOG features for test...")
    X_test = extract_hog_features(X_test_img)

    print("Training SVM...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=classes))

def main():
    evaluate_dataset(FERPLUS_TRAIN_DIR, FERPLUS_TEST_DIR, "FERPlus")
    evaluate_dataset(RAFDB_TRAIN_DIR, RAFDB_TEST_DIR, "RAF-DB")

if __name__ == "__main__":
    main()
