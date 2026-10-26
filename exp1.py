# EXPERIMENT 1
# LINEAR REGRESSION USING GRADIENT DESCENT AND NORMAL EQUATION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load California Housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["Price"] = housing.target

# Input: average rooms; output: house price
X = data[["AveRooms"]].values
y = data["Price"].values

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale input features for gradient descent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).flatten()
X_test_scaled = scaler.transform(X_test).flatten()

# Gradient Descent
w, b = 0.0, 0.0
learning_rate = 0.01
epochs = 1000
cost_history = []
n = len(X_train_scaled)

for i in range(epochs):
    y_pred = w * X_train_scaled + b
    dw = (1 / n) * np.sum((y_pred - y_train) * X_train_scaled)
    db = (1 / n) * np.sum(y_pred - y_train)
    w -= learning_rate * dw
    b -= learning_rate * db
    cost = (1 / (2 * n)) * np.sum((y_pred - y_train) ** 2)
    cost_history.append(cost)
    if i % 100 == 0:
        print(f"Epoch {i}, Cost = {cost:.4f}")

y_pred_gd = w * X_test_scaled + b
print("\nGradient Descent")
print("----------------")
print("Weight:", w)
print("Bias:", b)
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2 Score:", r2_score(y_test, y_pred_gd))

# Normal Equation (closed-form simple linear regression)
x_train_flat = X_train.flatten()
x_test_flat = X_test.flatten()
x_mean = np.mean(x_train_flat)
y_mean = np.mean(y_train)
w_ne = np.sum((x_train_flat - x_mean) * (y_train - y_mean)) / np.sum((x_train_flat - x_mean) ** 2)
b_ne = y_mean - w_ne * x_mean
y_pred_ne = w_ne * x_test_flat + b_ne

print("\nNormal Equation")
print("----------------")
print("Weight:", w_ne)
print("Bias:", b_ne)
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R2 Score:", r2_score(y_test, y_pred_ne))

# Sort values for a clear regression line
idx = np.argsort(x_test_flat)
x_sorted = x_test_flat[idx]
y_pred_ne_sorted = y_pred_ne[idx]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cost_history, linewidth=2)
plt.title("Gradient Descent Cost Convergence")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.scatter(x_test_flat, y_test, alpha=0.5, label="Actual Data")
plt.plot(x_sorted, y_pred_ne_sorted, linewidth=2, label="Normal Equation")
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("Normal Equation Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
