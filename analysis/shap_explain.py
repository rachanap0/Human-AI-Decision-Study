"""
SHAP explainability for 'human-ai-decision-study'.

Outputs:
- figures/shap_summary.png   (beeswarm: global view of feature effects)
- figures/shap_bar.png       (bar chart: mean |SHAP| by feature)
- figures/shap_force_one.png (optional: single prediction force plot image)

"""

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 1) Load trained model
model = joblib.load("models/model.pkl")

# 2) Load the full dataset (as DataFrame)
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# Split X/y consistently with training
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 3) Use a manageable background/sample for SHAP speed
#    (SHAP on full dataset is slow; sampling ~1000 rows is enough)
SAMPLE_N = 1000 if len(X) > 1000 else len(X)
X_sample = X.sample(n=SAMPLE_N, random_state=0)

# 4) Choose the right explainer
# RandomForestRegressor -> TreeExplainer is efficient and accurate
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 5) SHAP summary beeswarm (global)
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=180)
plt.close()
print("Saved:", os.path.join(FIG_DIR, "shap_summary.png"))

# 6) SHAP mean absolute values (bar chart)
# shap.summary_plot(..., plot_type="bar") draws a bar ranking
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=180)
plt.close()
print("Saved:", os.path.join(FIG_DIR, "shap_bar.png"))

# 7) Optional: single-row force plot saved as PNG
# Note: shap.force_plot is JS/HTML by default; we can rasterize it via save_matplotlib
try:
    one_row = X_sample.iloc[[0]]
    one_sv = explainer.shap_values(one_row)
    fp = shap.force_plot(explainer.expected_value, one_sv, one_row, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_force_one.png"), dpi=180)
    plt.close()
    print("Saved:", os.path.join(FIG_DIR, "shap_force_one.png"))
except Exception as e:
    # If force plot fails (some environments), skip gracefully
    print("Skipped force plot:", e)
