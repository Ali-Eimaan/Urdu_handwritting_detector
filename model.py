import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Load the CSV file
csv_file_path = 'Data/data.csv'
data = pd.read_csv(csv_file_path)

# Define image size and other parameters
IMG_HEIGHT = 128  # You can adjust this based on your needs
IMG_WIDTH = 128
BATCH_SIZE = 32

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Load images and labels
images = []
labels = []

for index, row in data.iterrows():
    img_path = row['image']
    label_info = eval(row['label'])
    
    for label_entry in label_info:
        label = label_entry['rectanglelabels']
        try:
            img = load_and_preprocess_image(img_path)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

images = np.array(images)

# Encode labels as multi-hot vectors
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(mlb.classes_), activation='sigmoid')  # Use sigmoid for multi-label classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary cross-entropy for multi-label classification
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val))

# Save the model
model.save('urdu_handwriting_recognition_model.h5')
