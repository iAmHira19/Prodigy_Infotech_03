import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


# Path to the dataset
dataset_path = 'Ml prodigy info tech/train/train'  # Adjust this to your folder containing the extracted images

# Image size (resize)
image_size = (64, 64)

# Lists to hold the images and labels
images = []
labels = []

# Load images and labels
print("Loading images...")
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    if img_path.endswith('.jpg'):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
        if img is None:
            print(f"Warning: Failed to read image: {img_name}")
            continue
        img = cv2.resize(img, image_size)  # Resize the image to 64x64
        img = img / 255.0  # Normalize the image
        images.append(img.flatten())  # Flatten and append the image
        label = 0 if 'cat' in img_name else 1  # Assign labels based on filename
        labels.append(label)

print(f"Total images loaded: {len(images)}")


# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM model
print("Training SVM model...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions...")
y_pred = svm_model.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['cat', 'dog'])
disp.plot(cmap='Blues')

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to predict new images
def predict_image(image_path, model, image_size):
    print(f"Predicting image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
    if img is None:
        raise FileNotFoundError(f"The file {image_path} does not exist or is not readable.")

    img = cv2.resize(img, image_size)  # Resize the image to 64x64
    img = img / 255.0  # Normalize the image
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for SVM input
    prediction = model.predict(img)
    return 'cat' if prediction == 0 else 'dog'


# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['cat', 'dog'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# Example of predicting a new image
test_image_path = 'Ml prodigy info tech/test1/test1/1.jpg'  # Change this to the path of a test image
if not os.path.isfile(test_image_path):
    raise FileNotFoundError(f"The file {test_image_path} does not exist. Please check the path and try again.")

prediction = predict_image(test_image_path, svm_model, image_size)
print(f'The predicted class for the image is: {prediction}')
