import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from joblib import dump, load
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_set')))
from data_set_combine import load_and_combine_datasets
from data_set_combine import show_sample_images
from data_set_combine import augment_dataset

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore")  # Commented out to only suppress ConvergenceWarning

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
RAFDB_CLASS_ORDER = ['1', '2', '3', '4', '5', '6', '7']
RAFDB_CLASS_NAMES = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']


def get_classes(train_dir, test_dir):
    train_classes = set([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    test_classes = set([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    # Only use classes present in both train and test
    return sorted(list(train_classes & test_classes))

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

def plot_learning_curve(X, y, dataset_name):
    print(f"Plotting learning curve for {dataset_name}...")
    estimator = LinearSVC(dual=False, max_iter=2000)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.5, 1.0, 4), shuffle=True, random_state=42)
    train_error = 1 - np.mean(train_scores, axis=1)
    val_error = 1 - np.mean(val_scores, axis=1)
    fig = plt.figure()
    plt.plot(train_sizes, train_error, 'o-', label='Training error')
    plt.plot(train_sizes, val_error, 'o-', label='Validation error')
    plt.title(f'Learning Curve')
    plt.xlabel('Training set size')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig

"""def evaluate_dataset(train_dir, test_dir, dataset_name):
    print(f"\n===== {dataset_name} =====")
    if dataset_name == "RAF-DB":
        # Use fixed class order and names for RAF-DB
        classes = RAFDB_CLASS_ORDER
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        target_names = RAFDB_CLASS_NAMES
    else:
        classes = get_classes(train_dir, test_dir)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        target_names = classes
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Plot learning curve using only the training set
    fig = plot_learning_curve(X_train, y_train, dataset_name)

    print("Training SVM...")
    clf = LinearSVC(dual=False, max_iter=2000)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
    dump((clf, scaler, target_names), f'{dataset_name}_hog_svm.joblib')

    return fig
"""
def main():
    X_train_img, y_train, X_test_img, y_test, class_names = load_and_combine_datasets()
    #Display sample images (visual sanity check)
    show_sample_images(X_train_img, y_train, class_names, num_per_class=2)
    print(f"Loaded {len(X_train_img)} training and {len(X_test_img)} test images.")

    # Augment the training dataset
    X_train_img, y_train = augment_dataset(X_train_img, y_train, times=2)
    print(f"After augmentation: {len(X_train_img)} training images.")
    
    print("Extracting HOG features for training...")
    X_train = extract_hog_features(X_train_img)

    print("Extracting HOG features for test...")
    X_test = extract_hog_features(X_test_img)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    fig = plot_learning_curve(X_train, y_train, "Combined")

    print("Training SVM...")
    clf = LinearSVC(dual=False, max_iter=2000)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=class_names))
    dump((clf, scaler, class_names), 'combined_unbalanced_hog_svm.joblib')

    plt.show()


if __name__ == "__main__":
    main()
