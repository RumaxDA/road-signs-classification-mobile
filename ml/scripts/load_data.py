import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = "D:\GitHub\Road-signs-classification-mobile\data\Train"
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CATEGORIES = 43

def load_gtsrb_data(data_dir):
    images = []
    labels = []

    print(f"Loading data from: {data_dir}")

    for category in tqdm(range(NUM_CATEGORIES), desc = "Loading classes"):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):
            print(f"Warning: Directory for class {category} does not exist. Skipping.")
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

    return (np.array(images), np.array(labels))
    
if __name__ == "__main__":
    images, labels = load_gtsrb_data(DATA_DIR)

    print("\n--- Data Loading Summary ---")
    print(f"Total images loaded: {len(images)}")
    print(f"Total labels loaded: {len(labels)}")
    print(f"Images array shape: {images.shape}")
    print(f"Labels array shape: {labels.shape}")
    print(f"----------------------------------")
