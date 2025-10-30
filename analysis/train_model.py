"""
Baseline model training for 'human-ai-decision-study'.

What this script does:
1) Loads the California Housing dataset from scikit-learn.
2) Splits into train/test.
3) Trains a RandomForestRegressor (tabular-ML baseline).
4) Evaluates with Mean Absolute Error (MAE).
5) Saves the trained model and a small test sample for later steps.

Run from project root with your venv active:
    python analysis/train_model.py
"""

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1) Load the California Housing dataset
# as_frame=True returns a pandas DataFrame
data = fetch_california_housing(as_frame=True)
df = data.frame.copy() #making a copy of the dataframe to avoid modifying the original data

#print(data.target_names) -> ['MedHouseVal']
#Sanity: target column is MedHouseVal

# 2) Separate features(X) and target(Y)
X = df.drop(columns=['MedHouseVal']) #all input features
y = df['MedHouseVal'] #target column, the value we want to predict

# 3) Split into train/test
# # Keep 20% aside as unseen test data to estimate generalization performance.
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------- 4) Define the model ----------
# Random Forest is a strong, low-maintenance baseline for tabular data.
# n_estimators: number of trees. More trees -> more stable up to a point.
# n_jobs=-1 uses all CPU cores to speed up training.
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# ---------- 5) Train (fit) the model ----------
model.fit(X_train, y_train)

# ---------- 6) Evaluate on test data ----------
# MAE is average |prediction - true|. Lower is better.
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Test MAE: {mae:.3f}")  # e.g., 0.45 ≈ $45,000 average error

# ---------- 7) Save artifacts ----------
# Save trained model to reuse later (so you don't retrain every time).
joblib.dump(model, "models/model.pkl")
print("Saved model → models/model.pkl")

# Save a small slice of the test set for tomorrow's demos (app + SHAP).
sample = X_test.copy()
sample["y_true"] = y_test.values
sample.head(100).to_csv("data/test_sample.csv", index=False)
print("Saved test sample → data/test_sample.csv")