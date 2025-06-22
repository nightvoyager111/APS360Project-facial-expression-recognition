import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Let's assume you already have X (features) and y (labels)
# For demonstration, we'll simulate this with random data
# In practice, extract features using deep learning or classical methods
num_samples = 1000
num_features = 100  # e.g., from CNN or HOG
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 7, size=num_samples)  # 7 classes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[
    'Happiness', 'Sadness', 'Fear', 'Disgust', 'Anger', 'Neutral', 'Surprise'
]))
