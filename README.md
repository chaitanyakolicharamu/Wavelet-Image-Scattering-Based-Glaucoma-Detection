# Wavelet Image Scattering Based Glaucoma Detection

This project recreates and analyzes the paper:

**“Wavelet Image Scattering Based Glaucoma Detection”**

The implementation focuses on glaucoma detection using image processing, wavelet scattering feature extraction, and machine learning classification techniques on retinal fundus images.

---

# Project Overview

Glaucoma is a major eye disease that can cause irreversible blindness if not detected early. This project uses fundus retinal images and applies image processing + wavelet scattering techniques to classify images as:

* Normal
* Glaucoma

The goal of this implementation is to understand how Wavelet Image Scattering Networks (WISN) can extract stable and meaningful retinal features for glaucoma detection.

---

# Dataset

Dataset Used:

* RIM-ONE DL Dataset

Classes:

* Normal
* Glaucoma

Dataset Structure:

* Training Images: 339
* Testing Images: 146

---

# Pipeline

The implemented pipeline includes:

1. Fundus Image Input
2. RGB & Gray Channel Extraction
3. Image Preprocessing

   * Gaussian Filtering
   * Morphological Operations
   * Image Resizing (300×300)
4. Wavelet Scattering Transform (Scattering2D)
5. Mean Pooling / Feature Reduction
6. Machine Learning Classification

   * Support Vector Machine (SVM)
   * Logistic Regression (LR)
7. Prediction

   * Normal / Glaucoma

---

# Image Processing Techniques Used

* Channel Extraction
* Gaussian Filtering
* Morphological Operations
* Image Resizing
* Multi-scale Wavelet Feature Extraction

---

# Wavelet Scattering Network (WISN)

The project uses Wavelet Scattering Transform for feature extraction.

Key advantages:

* Stable feature extraction
* Translation invariance
* Multi-scale texture analysis
* Effective for small medical datasets
* Reduces dependency on large deep learning models

---

# Models Used

## Support Vector Machine (SVM)

* Main classifier used
* Best performing model

## Logistic Regression (LR)

* Used for comparison

---

# Results

Best Configuration:

* Red Channel
* J = 6
* SVM Classifier

Performance:

* Accuracy: 86.30%
* F1-Score: 80.77%

Observations:

* Channel selection affects performance
* Wavelet scale significantly impacts feature quality
* Preprocessing improves feature consistency

---

# Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Kymatio
* Matplotlib

---

# Project Structure

```bash id="rkp1ux"
Wavelet-Image-Scattering-Based-Glaucoma-Detection/
│
├── dataset/
├── preprocessing/
├── feature_extraction/
├── models/
├── results/
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```

# Installation

```bash id="dmlp6n"
pip install -r requirements.txt
```

# Run Project

```bash id="g7qqlr"
python main.py
```

# Research & Educational Purpose

This project was recreated for:

* IMPRO & Computer Vision
* Research understanding
* Medical image processing analysis
* Wavelet scattering experimentation

---

# Author

Venkata Chaitanya Kolicharamu

GitHub:
[GitHub Profile](https://github.com/chaitanyakolicharamu?utm_source=chatgpt.com)
