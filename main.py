import os, time, joblib
import numpy as np

from dataset import load_split
from preprocess import preprocess_image
from features import extract_wavelet_features

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CACHE_DIR = "cache"
USE_ENHANCEMENT = False

def preprocess_dataset(images, mode, split_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{split_name}_{mode}_enhance_{USE_ENHANCEMENT}.pkl")

    if os.path.exists(path):
        print(f"Loading cached preprocessed {split_name} {mode}")
        return joblib.load(path)

    processed = []
    print(f"Preprocessing {split_name} {mode} images...")

    for i, img in enumerate(images):
        processed.append(preprocess_image(img, mode=mode, use_enhancement=USE_ENHANCEMENT))
        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"{split_name} {mode}: {i + 1}/{len(images)}")

    processed = np.array(processed)
    joblib.dump(processed, path)
    return processed

def get_features(images, mode, split_name, J):
    path = os.path.join(CACHE_DIR, f"{split_name}_{mode}_J{J}_enhance_{USE_ENHANCEMENT}_features.pkl")

    if os.path.exists(path):
        print(f"Loading cached features {split_name} {mode} J={J}")
        return joblib.load(path)

    features = extract_wavelet_features(images, J=J)
    joblib.dump(features, path)
    return features

def main():
    start = time.time()
    print("FINAL TRY: RED + GRAY combined wavelet features, J=6")

    X_train_raw, y_train = load_split("training_set")
    X_test_raw, y_test = load_split("test_set")

    J = 6
    modes = ["red", "gray"]

    train_features = []
    test_features = []

    for mode in modes:
        X_train_processed = preprocess_dataset(X_train_raw, mode, "train")
        X_test_processed = preprocess_dataset(X_test_raw, mode, "test")

        X_train_features = get_features(X_train_processed, mode, "train", J)
        X_test_features = get_features(X_test_processed, mode, "test", J)

        train_features.append(X_train_features)
        test_features.append(X_test_features)

    X_train_combined = np.concatenate(train_features, axis=1)
    X_test_combined = np.concatenate(test_features, axis=1)

    print("Combined train features:", X_train_combined.shape)
    print("Combined test features:", X_test_combined.shape)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("svm", SVC(kernel="linear", probability=True))
    ])

    param_grid = {
        "selector__k": [50, 75, 100, 125, 150, "all"],
        "svm__C": [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_combined, y_train)

    print("Best params:", grid.best_params_)

    y_pred = grid.predict(X_test_combined)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\n===== FINAL RESULT =====")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Glaucoma F1-score: {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    
    joblib.dump(grid.best_estimator_, "final_best_red_gray_svm.pkl")

    print(f"\nFinished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()