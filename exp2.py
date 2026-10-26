# EXPERIMENT 2
# LINEAR AND POLYNOMIAL REGRESSION USING AUTO MPG DATASET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the Auto MPG dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

# Select displacement and MPG, then remove missing rows
data = df[["displacement", "mpg"]].dropna()
X = data[["displacement"]]
y = data["mpg"]

print("Missing values:")
print(data.isnull().sum())
print("\nFeature:")
print(X.head())
print("\nTarget:")
print(y.head())

plt.figure(figsize=(8, 5))
plt.scatter(X["displacement"], y)
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Displacement vs MPG")
plt.grid(True)
plt.show()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nLinear Regression Results")
print("-------------------------")
print("MSE:", mse_linear)
print("R-squared:", r2_linear)

results = [{
    "Model": "Linear Regression",
    "Degree": 1,
    "MSE": mse_linear,
    "R-squared": r2_linear
}]
polynomial_models = {}

# Polynomial Regression for degrees 2, 3, 4 and 5
for degree in range(2, 6):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred_poly)
    r2 = r2_score(y_test, y_pred_poly)
    results.append({
        "Model": "Polynomial Regression",
        "Degree": degree,
        "MSE": mse,
        "R-squared": r2
    })
    polynomial_models[degree] = {"poly": poly, "model": poly_model}

results_df = pd.DataFrame(results)
print("\nModel Comparison")
print(results_df.to_string(index=False))

# Generate values for smooth regression curves
X_plot = pd.DataFrame({
    "displacement": np.linspace(X["displacement"].min(), X["displacement"].max(), 300)
})

plt.figure(figsize=(10, 6))
plt.scatter(X["displacement"], y, label="Actual Data", alpha=0.6)
plt.plot(X_plot["displacement"], linear_model.predict(X_plot), label="Linear Regression")
for degree in range(2, 6):
    poly = polynomial_models[degree]["poly"]
    model = polynomial_models[degree]["model"]
    plt.plot(
        X_plot["displacement"],
        model.predict(poly.transform(X_plot)),
        label=f"Polynomial Degree {degree}"
    )
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear and Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

# MSE comparison graph
plt.figure(figsize=(8, 5))
bars = plt.bar(results_df["Degree"].astype(str), results_df["MSE"])
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.xlabel("Polynomial Degree (1 = Linear)")
plt.ylabel("Mean Squared Error")
plt.title("Comparison of MSE")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# R-squared comparison graph
plt.figure(figsize=(8, 5))
plt.plot(results_df["Degree"], results_df["R-squared"], marker="o")
plt.xlabel("Polynomial Degree")
plt.ylabel("R-squared")
plt.title("Comparison of R-squared")
plt.grid(True)
plt.show()
