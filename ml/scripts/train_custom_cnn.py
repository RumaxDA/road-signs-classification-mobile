import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

DATA_DIR = "D:/GitHub/Road-signs-classification-mobile/ml/data/Train"
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CATEGORIES = 43

def load_gtsrb_data(data_dir):
    #Load images and labels from the dataset dir
    images = []
    labels = []
    print(f"Loading data from: {data_dir}")
    for category in tqdm(range(NUM_CATEGORIES), desc="Loading classes"): 
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                image = cv2.imread(img_path)
                image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image_resized)
                labels.append(category)
            except Exception as e:
                print(f"Error loading file {img_path}: {e}")
    return np.array(images), np.array(labels)


def build_custom_cnn():
    #Builds and returns a custom CNN model
    model = Sequential([
        tf.keras.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(filters = 32, kernel_size=(3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(rate = 0.25),
        Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2)),
        Dropout(rate = 0.25),
        Flatten(),
        Dense(units=512, activation = 'relu'),
        Dropout(rate = 0.5),
        Dense(units=NUM_CATEGORIES, activation = 'softmax')
    ])
    return model

if __name__ == "__main__":
    # Loading raw data
    images, labels = load_gtsrb_data(DATA_DIR)
    print(f"\nLoaded {len(images)} images")

    # Preprocess the data
    print("Preparing data for training...")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size = 0.2, random_state = 42, stratify=labels
    )

    # Normalize images
    X_train = X_train.astype('float32') / 255.0
    x_test = X_test.astype('float32') / 255.0

    # One hot encoding
    y_train = to_categorical(y_train, NUM_CATEGORIES)
    y_test = to_categorical(y_test, NUM_CATEGORIES)

    print("Data preparation complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Build the model
    print("\nBuilding and compiling the model...")
    model = build_custom_cnn()
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    model.summary()

    # Train the model
    epochs = 20
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(X_train, y_train,
                        batch_size = 32,
                        epochs = epochs,
                        validation_data = (X_test, y_test))
    
    # Save the trained model 
    print("\nTraining complete. Saving the model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, '..', 'trained_models')
    save_dir = os.path.normpath(save_dir)
    
    os.makedirs(save_dir, exist_ok = True)
    model_path = os.path.join(save_dir, 'custom_cnn_model.h5')
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")

