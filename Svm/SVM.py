import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import seaborn as sns
# Paths to data and annotations
train_data_dir = r'./NDVI_Images/Train_images'
test_data_dir = r'./NDVI_Images/Test_Images'
train_annotations_file = r'./NDVI_Images/Train_Labels_CSV.csv'
test_annotations_file = r'./NDVI_Images/Test_Labels_CSV.csv'

# Function to extract feature vectors from images
def extract_features_from_images(image_paths):
    features = []
    for i, path in enumerate(image_paths):
        try:
            print(f"Processing training image {i + 1}/{len(image_paths)}: {path}")

            # Read the image using OpenCV
            image = cv2.imread(path)

            # Check if the image is empty
            if image is None or image.size == 0:
                print(f"Warning: Skipping empty image at path {path}")
                continue

            # Resize the image to a fixed size
            resized_image = cv2.resize(image, (224, 224))

            # Flatten the image into a 1D array
            flattened_image = resized_image.flatten()

            # Append the flattened image as a feature vector
            features.append(flattened_image)
        except Exception as e:
            print(f"Error processing image at path {path}: {str(e)}")

    return np.array(features)

# Function to load image paths and corresponding labels from annotations file
def load_data(data_dir, annotations_file):
    annotations_df = pd.read_csv(annotations_file)

    image_paths = [os.path.join(data_dir, img_name) for img_name in annotations_df['filename']]
    labels = annotations_df['class'].values

    return image_paths, labels

# Load training data
print("Loading training data...")
X_train_paths, y_train = load_data(train_data_dir, train_annotations_file)

# Load testing data
print("Loading testing data...")
X_test_paths, y_test = load_data(test_data_dir, test_annotations_file)

# Extract features from training images
print("Extracting features from training images...")
X_train = extract_features_from_images(X_train_paths)

# Extract features from testing images
print("Extracting features from testing images...")
X_test = extract_features_from_images(X_test_paths)

# Initialize SVM classifier
print("Initializing SVM classifier...")
clf = svm.SVC(kernel='linear')

# Train the SVM classifier
print("Training SVM classifier...")
clf.fit(X_train, y_train)

# Save the trained model
print("Saving the trained model...")
joblib.dump(clf, 'svm_model4.pkl')

# Make predictions on the test set
print("Making predictions on the test set...")
test_predictions = clf.predict(X_test)

# Evaluate accuracy on the test set
print("Evaluating accuracy on the test set...")
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy}')

# Create and plot the confusion matrix with class labels
class_names = ['Healthy', 'Stressed']
conf_matrix = confusion_matrix(y_test, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()