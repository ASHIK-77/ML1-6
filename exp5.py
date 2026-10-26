# EXPERIMENT 5
# IMAGE CLASSIFICATION USING K-NEAREST NEIGHBORS (KNN)
# ON THE FASHION-MNIST DATASET

import time
import warnings
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

# Download Fashion-MNIST
print("Downloading Fashion-MNIST dataset (may take some time)...")
fashion_mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X = fashion_mnist.data
y = fashion_mnist.target.astype(int)
print(f"Dataset size: {X.shape[0]} images, {X.shape[1]} pixels per image.")

# Select a stratified subset of 12,000 images
X_subset, _, y_subset, _ = train_test_split(
    X, y, train_size=12000, stratify=y, random_state=42
)

# Split subset into 10,000 training and 2,000 testing images
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=2000, stratify=y_subset, random_state=42
)
print("Training images:", X_train.shape[0])
print("Testing images:", X_test.shape[0])

# Normalize pixel values from 0-255 to 0-1
print(f"Before scaling: min={X_train.min()}, max={X_train.max()}")
X_train = X_train / 255.0
X_test = X_test / 255.0
print(f"After scaling: min={X_train.min()}, max={X_train.max()}")

k_values = [1, 3, 5, 7, 9, 15]
results = {}
print(f"{'K value':<10} | {'Accuracy':<12} | {'Prediction time (s)':<20}")
print("-" * 48)

# Train and evaluate KNN for different K values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=1)
    knn.fit(X_train, y_train)
    start_time = time.time()
    y_pred = knn.predict(X_test)
    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    results[k] = {
        "accuracy": accuracy,
        "time": elapsed_time,
        "prediction": y_pred
    }
    print(f"{k:<10} | {accuracy * 100:<11.2f}% | {elapsed_time:<20.2f}")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

best_k = max(results, key=lambda k: results[k]["accuracy"])
print(f"\nBest K: {best_k} with {results[best_k]['accuracy'] * 100:.2f}% accuracy")

print("\nClassification Report:")
print(classification_report(
    y_test,
    results[best_k]["prediction"],
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, results[best_k]["prediction"]))
