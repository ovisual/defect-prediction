import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

REDUCED = ["lloc", "cc_max", "comments", "fan_out", "lines_added", "n_authors", "prior_defects"]
df = pd.read_csv("data/processed/all_repos.csv")


def fit(cols):
    m = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                      LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return m.fit(df[cols], df["defect_broad"])


full, size = fit(REDUCED), fit(["loc"])
joblib.dump(full, "models/defect_lr_reduced.joblib")
joblib.dump(size, "models/defect_lr_size_only.joblib")
json.dump({"features": REDUCED, "size_features": ["loc"], "trained_on": sorted(df.repo.unique()),
           "n_files": len(df), "cutoff": "2025-10-01"}, open("models/model_card.json", "w"), indent=2)

coef = pd.Series(full[-1].coef_[0], index=REDUCED).sort_values(ascending=False).round(3)
print("Standardised coefficients (log1p scale):\n", coef)
print("\nSaved models/defect_lr_reduced.joblib, defect_lr_size_only.joblib, model_card.json")