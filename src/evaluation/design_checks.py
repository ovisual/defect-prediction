import sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, x): [s.write(x) for s in self.streams]
    def flush(self): [s.flush() for s in self.streams]


sys.stdout = Tee(sys.__stdout__, open("results/eda_report.txt", "w", encoding="utf-8"))


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


FEATURES = ["loc", "sloc", "lloc", "comments", "cc_sum", "cc_max", "cc_mean",
            "n_functions", "fan_out", "n_commits", "lines_added", "lines_deleted",
            "n_authors", "prior_defects"]
df = pd.read_csv("data/processed/all_repos.csv")
X, y, groups = df[FEATURES], df["defect_broad"], df["repo"]

# 1. zeros and skew ---------------------------------------------------------
section("1. ZEROS AND SKEW (decides log transform)")
t = pd.DataFrame({"zeros_%": (X == 0).mean() * 100,
                  "skew_raw": X.apply(skew),
                  "skew_log1p": np.log1p(X).apply(skew)}).round(2)
print(t)
print("\nFiles with no history before cutoff (n_commits == 0):",
      int((df.n_commits == 0).sum()))
print("Identical feature rows:", int(X.duplicated().sum()))

# 2. correlated pairs -------------------------------------------------------
section("2. SPEARMAN CORRELATION PAIRS (|r| >= 0.80)")
corr = X.corr(method="spearman")
corr.round(3).to_csv("results/spearman_corr.csv")
pairs = sorted(((a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES)
                for b in FEATURES[i + 1:] if abs(corr.loc[a, b]) >= 0.8),
               key=lambda p: -abs(p[2]))
for a, b, r in pairs:
    print(f"{a:15}{b:15}{r:7.3f}")

# 3. univariate power -------------------------------------------------------
section("3. UNIVARIATE POWER (AUC of each feature alone; 0.5 = useless)")
auc = {f: roc_auc_score(y, X[f]) for f in FEATURES}
spear = {f: pd.Series(X[f]).corr(y, method="spearman") for f in FEATURES}
uni = pd.DataFrame({"auc": auc, "spearman_vs_label": spear}).round(3)
print(uni.sort_values("auc", ascending=False))

# 4. clusters and suggested representatives ---------------------------------
section("4. FEATURE CLUSTERS (|r| > ~0.85) AND SUGGESTED REPRESENTATIVE")
dist = 1 - corr.abs().values
np.fill_diagonal(dist, 0)
cl = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.15, criterion="distance")
for c in sorted(set(cl)):
    members = [f for f, k in zip(FEATURES, cl) if k == c]
    best = max(members, key=lambda f: abs(auc[f] - 0.5))
    print(f"cluster {c}: {members}  -> keep: {best}")

# 5. VIF --------------------------------------------------------------------
section("5. VARIANCE INFLATION FACTORS (log1p features; >10 = redundant)")
C = np.corrcoef(np.log1p(X).values, rowvar=False)
print(pd.Series(np.diag(np.linalg.pinv(C)), index=FEATURES).round(1)
      .sort_values(ascending=False))

# 6. per-repo consistency ---------------------------------------------------
section("6. PER-REPO UNIVARIATE AUC (do the same features work everywhere?)")
rows = {}
for repo, g in df.groupby("repo"):
    rows[repo.split("/")[1]] = {f: roc_auc_score(g.defect_broad, g[f]) for f in FEATURES}
print(pd.DataFrame(rows).round(2))

# 7. labels -----------------------------------------------------------------
section("7. LABEL CHECKS")
print("strict vs broad (rows=strict, cols=broad):")
print(pd.crosstab(df.defect_strict, df.defect_broad))
print("\nPositives (broad) and files per repo:")
print(df.groupby("repo").defect_broad.agg(positives="sum", files="count"))

# 8. baselines --------------------------------------------------------------
section("8. BASELINES: pooled stratified 10-fold vs leave-one-repo-out (AUC)")
log = FunctionTransformer(np.log1p)


def lr():
    return make_pipeline(log, StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  n_jobs=-1, random_state=42)


def evaluate(model, cols):
    Xc = X[cols]
    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    p = cross_val_predict(model, Xc, y, cv=skf, method="predict_proba")[:, 1]
    per_repo = {}
    for tr, te in LeaveOneGroupOut().split(Xc, y, groups):
        m = clone(model).fit(Xc.iloc[tr], y.iloc[tr])
        per_repo[groups.iloc[te[0]].split("/")[1]] = roc_auc_score(
            y.iloc[te], m.predict_proba(Xc.iloc[te])[:, 1])
    return roc_auc_score(y, p), per_repo


setups = {
    "LR  loc only":           (lr(), ["loc"]),
    "LR  n_commits only":     (lr(), ["n_commits"]),
    "LR  prior_defects only": (lr(), ["prior_defects"]),
    "LR  all 14":             (lr(), FEATURES),
    "RF  all 14":             (rf(), FEATURES),
}
summary, detail = [], {}
for name, (model, cols) in setups.items():
    a, per = evaluate(model, cols)
    summary.append({"model": name, "pooled_10fold": round(a, 3),
                    "LORO_mean": round(np.mean(list(per.values())), 3),
                    "LORO_min": round(min(per.values()), 3)})
    detail[name] = per
print(pd.DataFrame(summary).to_string(index=False))
print("\nLeave-one-repo-out AUC per held-out repo:")
print(pd.DataFrame(detail).round(2))

print("\nReport saved to results/eda_report.txt")