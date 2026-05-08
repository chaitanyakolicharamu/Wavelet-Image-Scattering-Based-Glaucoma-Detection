import cv2
import numpy as np

IMG_SIZE = (300, 300)

def get_2d_representation(img, mode="gray"):
    img = cv2.resize(img, IMG_SIZE)

    if mode == "blue":
        return img[:, :, 0]

    if mode == "green":
        return img[:, :, 1]

    if mode == "red":
        return img[:, :, 2]

    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raise ValueError("mode must be one of: blue, green, red, gray")


def enhance_image(img_2d):
    blurred = cv2.GaussianBlur(img_2d, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

    return morph


def preprocess_image(img, mode="gray", use_enhancement=False):
    img_2d = get_2d_representation(img, mode)

    if use_enhancement:
        img_2d = enhance_image(img_2d)

    return img_2d