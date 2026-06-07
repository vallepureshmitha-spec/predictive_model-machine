import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. LOAD THE CLEANED DATASET
# =====================================================================
input_file = "cleaned_project_data.csv"

if not os.path.exists(input_file):
    print(f"Error: '{input_file}' not found. Please run your cleaning script first!")
    exit()

df = pd.read_csv(input_file)
print("=== 1. DATASET LOADED FOR MODELING ===")
print(df.head())

# =====================================================================
# 2. FEATURE ENGINEERING & ENCODING
# =====================================================================
print("\n=== 2. PROCESSING CATEGORICAL FEATURES ===")

# Machine Learning models require numbers, so we encode text columns
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Let's say we want to predict 'Status' (e.g., FT vs PT)
# Make sure 'Status' is your target variable column name
target_variable = "Status"

if target_variable not in df.columns:
    # Fallback to the last categorical column if 'Status' isn't found
    target_variable = categorical_cols[-1]

print(f"Target variable chosen for prediction: '{target_variable}'")

# Encode all categorical features to numerical values
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate Features (X) and Target Label (y)
X = df.drop(columns=[target_variable])
y = df[target_variable]

# =====================================================================
# 3. TRAIN-TEST SPLIT
# =====================================================================
# Splitting data: 80% for training the model, 20% for testing performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# =====================================================================
# 4. MODEL TRAINING (RANDOM FOREST)
# =====================================================================
print("\n=== 4. TRAINING RANDOM FOREST MODEL ===")

# Initialize the algorithm
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the hidden test set
y_pred = model.predict(X_test)
# Get probability scores for the ROC curve evaluation
y_probs = model.predict_proba(X_test)[:, 1]

# =====================================================================
# 5. MODEL EVALUATION METRICS
# =====================================================================
print("\n=== 5. PERFORMANCE METRICS ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate precision metrics
cm = confusion_matrix(y_test, y_pred)
auc_score = (
    roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 1.0
)
print(f"ROC-AUC Score: {auc_score:.4f}")

# =====================================================================
# 6. PERFORMANCE VISUALIZATION
# =====================================================================
print("\n=== 6. GENERATING PERFORMANCE VISUALIZATIONS ===")
plt.figure(figsize=(14, 6))

# Plot 1: Confusion Matrix Grid
plt.subplot(1, 2, 1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=plt.gca(), values_format="d")
plt.title("Confusion Matrix Dashboard", fontsize=14, fontweight="bold")
plt.grid(False)  # Turn off background grid lines for readability

# Plot 2: ROC Curve
plt.subplot(1, 2, 2)
if len(np.unique(y_test)) > 1:
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC Curve (AUC = {auc_score:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
else:
    # Fallback if testing subset only contains 1 distinct class label
    plt.text(
        0.5,
        0.5,
        "ROC Curve requires at least\n2 distinct classes in the test data.",
        ha="center",
        va="center",
        fontsize=12,
    )
    plt.title("ROC Curve (Insufficient Data)", fontsize=14, fontweight="bold")

plt.tight_layout()
print("Showing performance visuals...")
plt.show()
