# EXPERIMENT 6
# CUSTOMER SEGMENTATION USING AN ID3-STYLE DECISION TREE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Update this path if Online Retail.xlsx is stored elsewhere
file_path = r"C:\Users\CSEC2\Downloads\Online Retail.xlsx"
df = pd.read_excel(file_path)

print(df.head())
df.info()

# Data cleaning
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("Cleaned data shape:", df.shape)
print(df.head())

# Aggregate transactions to customer level
customer_data = df.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfTransactions=("InvoiceNo", "nunique"),
    NumberOfProducts=("StockCode", "nunique"),
    AverageOrderValue=("TotalAmount", "mean"),
    Country=("Country", "first")
).reset_index()

print("\nCustomer-level data:")
print(customer_data.head())

# Create Low, Medium and High Value segments from total spending
customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpend"],
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)
print("\nSegment counts:")
print(customer_data["Segment"].value_counts())

# Select input features and target label; TotalSpend is excluded
features = [
    "TotalQuantity",
    "NumberOfTransactions",
    "NumberOfProducts",
    "AverageOrderValue"
]
X = customer_data[features]
y = customer_data["Segment"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ID3-style tree: entropy criterion approximates information-gain splitting
id3_model = DecisionTreeClassifier(
    criterion="entropy", random_state=42, max_depth=4
)
id3_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = id3_model.predict(X_test)
print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": id3_model.feature_importances_
}).sort_values("Importance", ascending=True)

# Visualize the decision tree
plt.figure(figsize=(22, 12))
plot_tree(
    id3_model,
    feature_names=features,
    class_names=[str(c) for c in id3_model.classes_],
    filled=True,
    rounded=True
)
plt.title("Customer Segmentation using ID3-Style Decision Tree")
plt.show()

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Customer Behavior Feature")
plt.title("Feature Importance in Customer Segmentation")
plt.tight_layout()
plt.show()
