import numpy as np
import torch
from kymatio.torch import Scattering2D

IMG_SIZE = 300

def extract_wavelet_features(images, J=2):
    scattering = Scattering2D(
        J=J,
        shape=(IMG_SIZE, IMG_SIZE)
    )

    features = []

    print(f"Extracting wavelet scattering features with J={J}...")

    for i, img in enumerate(images):
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        with torch.no_grad():
            scattering_coeffs = scattering(img_tensor)

        coeffs = scattering_coeffs.squeeze(0).numpy()

        # Paper-style feature reduction
        feature_vector = coeffs.mean(axis=(1, 2))

        features.append(feature_vector)

        if (i + 1) % 25 == 0 or (i + 1) == len(images):
            print(f"Wavelet features: {i + 1}/{len(images)} completed")

    return np.array(features)