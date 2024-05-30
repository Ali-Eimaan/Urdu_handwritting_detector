import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer

# Load the CSV file for test data
data = pd.read_csv('Data/data.csv')

# Define image size
IMG_HEIGHT = 128  # Same as used in training
IMG_WIDTH = 128

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Load test images and labels
test_images = []
test_labels = []

for index, row in data.iterrows():
    img_path = row['image']
    labels_json = eval(row['label'])  # Assuming 'label' column contains JSON-like data
    
    # Extract labels from the JSON structure
    labels = [label_entry['rectanglelabels'][0] for label_entry in labels_json]
    
    try:
        img = load_and_preprocess_image(img_path)
        test_images.append(img)
        test_labels.append(labels)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

test_images = np.array(test_images)

# Encode labels as multi-hot vectors
mlb = MultiLabelBinarizer()
test_labels = mlb.fit_transform(test_labels)

# Load the trained model
model = tf.keras.models.load_model('urdu_handwriting_recognition_model.h5')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Get predictions on test data
predictions = model.predict(test_images)
predicted_labels = mlb.inverse_transform((predictions > 0.5).astype(np.int64))

# Compare predictions with true labels
for i in range(len(test_labels)):
    true_labels = mlb.inverse_transform(np.array([test_labels[i]]))[0]
    pred_labels = predicted_labels[i]
    print(f"True Labels = {true_labels}, Predicted Labels = {pred_labels}")
