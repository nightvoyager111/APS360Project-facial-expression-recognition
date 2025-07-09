import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import warnings
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate
from skimage.transform import resize

import collections
from matplotlib import cm



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

def show_sample_images(X, y, class_names, num_per_class=2):
    fig, axes = plt.subplots(len(class_names), num_per_class, figsize=(num_per_class * 2, len(class_names) * 2))
    for class_idx, class_name in enumerate(class_names):
        i = 0
        for idx in range(len(X)):
            if y[idx] == class_idx:
                ax = axes[class_idx, i] if num_per_class > 1 else axes[class_idx]
                ax.imshow(X[idx], cmap='gray')
                ax.axis('off')
                ax.set_title(class_name if i == 0 else "")
                i += 1
                if i == num_per_class:
                    break
    plt.tight_layout()
    plt.show(block=True)
    plt.show()
    
def augment_img(img):
    """Apply random augmentations to a single image"""
    # 1. Random horizontal flip
    if random.random() < 0.5:
        img = np.fliplr(img)
    # 2. Random small rotation
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        img = rotate(img, angle, reshape=False, mode='nearest')
        
    # 3. Random brightness shift
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 1)
        
    # 4.Add Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.03, img.shape)
        img = np.clip(img + noise, 0, 1)
        
    # 5. Random crop and resize
    if random.random() < 0.5:
        zoom_factor = random.uniform(0.85, 1.0)
        h, w = img.shape
        zh, zw = int(h * zoom_factor), int(w * zoom_factor)
        top = random.randint(0, h - zh)
        left = random.randint(0, w - zw)
        img = img[top:top + zh, left:left + zw]
        img = resize(img, IMAGE_SIZE, anti_aliasing=True)
        
    return img

def augment_dataset(X, y, times=1):
    """Apply augmentations by applying random transformations to each image"""
    X_augmented, y_augmented = [], []
    
    for i in range(len(X)):
        for _ in range(times):  
            img_aug = augment_img(X[i])
            X_augmented.append(img_aug)
            y_augmented.append(y[i])
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    return np.concatenate((X, X_augmented)), np.concatenate((y, y_augmented))
       
       
def show_augmented_versions(X, y, class_names, target_class_idx=0):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for idx in range(len(X)):
        if y[idx] == target_class_idx:
            base_img = X[idx]
            break
        
    for i in range(5):
        aug_img = augment_img(base_img)
        ax = axes[i]
        ax.imshow(aug_img, cmap='gray')
        ax.axis('off')
        
    plt.suptitle(f"Augmented Samples for Class: {class_names[target_class_idx]}")
    plt.tight_layout()
    plt.show()
        
    
   
def show_class_distribution(y, class_names):
    counts = collections.Counter(y)
    ordered_counts = [counts[i] for i in range(len(class_names))]
    
    # Add a color fade effect to the bars
    colors = cm.Blues(np.linspace(0.4, 0.9, len(class_names)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, ordered_counts, color=colors, edgecolor='black', linewidth=0.6, width=0.6)

    for bar, count in zip(bars, ordered_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 3, str(count), 
                 ha='center', va='bottom', fontsize=9, color='black')
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.title("Number of Samples per Class", fontsize=14, fontweight='bold')
    plt.ylabel("Number", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("visual_class_distribution.png")
    plt.show()

 
def show_dataset_distribution_comparison():
    class_names = get_standard_classes()
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Load separately
    _, y_fer = load_image_folder(FERPLUS_TRAIN_DIR, class_to_idx)
    _, y_raf = load_image_folder(RAFDB_TRAIN_DIR, class_to_idx)
    
    # Count
    fer_counts = collections.Counter(y_fer)
    raf_counts = collections.Counter(y_raf)
    fer_vals = [fer_counts[i] for i in range(len(class_names))]
    raf_vals = [raf_counts[i] for i in range(len(class_names))]
    total_vals = [f + r for f, r in zip(fer_vals, raf_vals)]

    x = np.arange(len(class_names))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x, fer_vals, label='FER2013+', color=cm.Oranges(0.6))
    bars2 = ax.bar(x, raf_vals, bottom=fer_vals, label='RAF-DB', color=cm.Blues(0.6))
    
    # Label each part of the stack
    for idx in range(len(class_names)):
        # Label FER
        ax.text(x[idx], fer_vals[idx] / 2, str(fer_vals[idx]),
                ha='center', va='center', fontsize=9, color='black')
        # Label RAF
        ax.text(x[idx], fer_vals[idx] + raf_vals[idx] / 2, str(raf_vals[idx]),
                ha='center', va='center', fontsize=9, color='black')
            
    ax.set_title("Sample Count per Class in FER2013+ vs RAF-DB", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("visual_dataset_comparison.png")
    plt.show()
    
def main():
    X_train_img, y_train, X_test_img, y_test, class_names = load_and_combine_datasets()
    #show_augmented_versions(X_train_img, y_train, class_names, target_class_idx=0)  
    #show_class_distribution(y_train, class_names)
    show_dataset_distribution_comparison()
if __name__ == "__main__":
    main()
   