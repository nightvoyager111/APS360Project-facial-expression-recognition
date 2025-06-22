import pandas as pd
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


df = pd.read_csv('./data/fer2013.csv')  




def process_fer_image(pixels, size=(48, 48)):
    image = np.array(pixels.split(), dtype='float32').reshape(48, 48)
    image /= 255.0  # Normalize to [0,1]
    return resize(image, size)


X_images = np.array([process_fer_image(p) for p in df['pixels']])
y_labels = np.array(df['emotion'])




def extract_hog_features(images):
    return np.array([
        hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            orientations=9, block_norm='L2-Hys') for img in images
    ])


X_features = extract_hog_features(X_images)


X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, stratify=y_labels, random_state=42)




clf = SVC(kernel='rbf', decision_function_shape='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("Classification Report:\n")
print(classification_report(y_test, y_pred))



