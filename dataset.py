import os
import cv2
import numpy as np

BASE_PATH = r"C:\Users\venka\Downloads\RIM-ONE_DL_images\RIM-ONE_DL_images\partitioned_randomly"
IMG_SIZE = (300, 300)

def load_split(split_name):
    images = []
    labels = []

    for class_name, label in [("normal", 0), ("glaucoma", 1)]:
        folder = os.path.join(BASE_PATH, split_name, class_name)

        for file_name in os.listdir(folder):
            img_path = os.path.join(folder, file_name)

            img = cv2.imread(img_path)
            if img is None:
                print("Could not read:", img_path)
                continue

            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)