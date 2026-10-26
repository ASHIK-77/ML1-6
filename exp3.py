# EXPERIMENT 3
# LOGISTIC REGRESSION USING BREAST CANCER DATASET
# MLE vs MAP (L1 AND L2)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Load dataset: 0 = malignant, 1 = benign
data = load_breast_cancer()
X = data.data
y = data.target
print("Data Shape:", X.shape)
print("Classes:", np.unique(y))

# Split while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLE model: no regularization
mle_model = LogisticRegression(
    penalty=None, max_iter=5000, random_state=42
)
mle_model.fit(X_train, y_train)
y_pred_mle = mle_model.predict(X_test)

# MAP with L2 regularization
map_l2 = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs",
    max_iter=5000, random_state=42
)
map_l2.fit(X_train, y_train)
y_pred_l2 = map_l2.predict(X_test)

# MAP with L1 regularization
map_l1 = LogisticRegression(
    penalty="l1", C=1.0, solver="liblinear",
    max_iter=5000, random_state=42
)
map_l1.fit(X_train, y_train)
y_pred_l1 = map_l1.predict(X_test)

def evaluate(model_name, y_true, y_pred):
    print("\n" + "=" * 40)
    print(model_name)
    print("=" * 40)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

# Evaluate all models
evaluate("MLE", y_test, y_pred_mle)
evaluate("MAP (L2)", y_test, y_pred_l2)
evaluate("MAP (L1)", y_test, y_pred_l1)

print("\nNumber of Zero Coefficients")
print("MLE:", np.sum(mle_model.coef_[0] == 0))
print("MAP L2:", np.sum(map_l2.coef_[0] == 0))
print("MAP L1:", np.sum(map_l1.coef_[0] == 0))

models = [
    ("MLE", mle_model, y_pred_mle),
    ("MAP L2", map_l2, y_pred_l2),
    ("MAP L1", map_l1, y_pred_l1)
]
results = []
for name, model, pred in models:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 score": f1_score(y_test, pred, zero_division=0)
    })
results_df = pd.DataFrame(results)
print("\nPerformance Comparison")
print(results_df.to_string(index=False))

ax = results_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Performance: MLE vs MAP-L2 vs MAP-L1")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.02)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.grid(axis="y", alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=2)
plt.tight_layout()
plt.show()

# ROC curves and AUC
plt.figure(figsize=(9, 7))
for name, model, pred in models:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve: MLE vs MAP L2 vs MAP L1")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
