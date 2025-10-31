"""
Permutation importance on the held-out test split used on Day 1.
Outputs: figures/perm_importance.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load model
model = joblib.load("models/model.pkl")

# Load dataset and split like Day 1
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Baseline MAE
base_mae = mean_absolute_error(y_test, model.predict(X_test))

# Permutation importance (on test set)
r = permutation_importance(
    model, X_test, y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Sort by mean importance
idx = np.argsort(r.importances_mean)[::-1]
features = X_test.columns[idx]
means = r.importances_mean[idx]
stds = r.importances_std[idx]

# Plot
plt.figure(figsize=(7, 4.5))
plt.bar(range(len(features)), means, yerr=stds)
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.ylabel("Permutation importance (↑ = larger MAE increase)")
plt.title(f"Permutation Importance (baseline MAE = {base_mae:.3f})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/perm_importance.png", dpi=180)
plt.close()

print("Saved:", f"{FIG_DIR}/perm_importance.png")
