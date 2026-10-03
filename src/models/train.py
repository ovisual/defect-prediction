import os, sys, time, warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv          # 1 repeat, smaller forests (for testing)
SEED = 42
REPEATS = 1 if QUICK else 3
TREES = 100 if QUICK else 200
OUT = "results/model_results.csv"

FEATURES = ["loc", "sloc", "lloc", "comments", "cc_sum", "cc_max", "cc_mean",
            "n_functions", "fan_out", "n_commits", "lines_added", "lines_deleted",
            "n_authors", "prior_defects"]
STATIC = ["lloc", "cc_max", "comments", "fan_out"]
SETS = {
    "SIZE": ["loc"],
    "STATIC": STATIC,
    "PROCESS": ["lines_added", "n_authors", "prior_defects"],
    "REDUCED": STATIC + ["lines_added", "n_authors", "prior_defects"],
    "REDUCED_NO_PRIOR": STATIC + ["lines_added", "n_authors"],
    "ALL14": FEATURES,
}
IMPORTANCE_GROUPS = {
    "size (loc,sloc,lloc,cc_sum,n_functions)": ["loc", "sloc", "lloc", "cc_sum", "n_functions"],
    "complexity shape (cc_max,cc_mean)": ["cc_max", "cc_mean"],
    "comments": ["comments"],
    "fan_out": ["fan_out"],
    "churn (commits,added,deleted,authors)": ["n_commits", "lines_added", "lines_deleted", "n_authors"],
    "prior_defects": ["prior_defects"],
}
MODELS = ["RandomForest", "NaiveBayes", "LogisticRegression", "DecisionTree", "SVM"]


def build(name, imb):
    clfs = {
        "RandomForest": (RandomForestClassifier(n_estimators=TREES, random_state=SEED),
                         {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 5]}),
        "NaiveBayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]}),
        "LogisticRegression": (LogisticRegression(max_iter=2000), {"clf__C": [0.1, 1, 10]}),
        "DecisionTree": (DecisionTreeClassifier(random_state=SEED),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 20]}),
        "SVM": (SVC(kernel="rbf"), {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}),
    }
    clf, grid = clfs[name]
    if imb == "cw" and name != "NaiveBayes":
        clf.set_params(class_weight="balanced")
    steps = [("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]
    if imb == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return GridSearchCV(Pipeline(steps), grid, scoring="roc_auc", n_jobs=-1,
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED))


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def get_splits(df, X, y, scheme):
    if scheme == "pooled":
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=REPEATS, random_state=SEED)
        return [(f"f{i:02d}", tr, te) for i, (tr, te) in enumerate(cv.split(X, y))]
    return [(df.repo.iloc[te[0]].split("/")[1], tr, te)
            for tr, te in LeaveOneGroupOut().split(X, y, df.repo)]


def run_config(df, fset, model, imb, scheme):
    X, y = df[SETS[fset]], df["defect_broad"]
    rows = []
    for fold, tr, te in get_splits(df, X, y, scheme):
        gs = build(model, imb).fit(X.iloc[tr], y.iloc[tr])
        yt, s, p = y.iloc[te], score(gs, X.iloc[te]), gs.predict(X.iloc[te])
        rows.append(dict(scheme=scheme, features=fset, model=model, imbalance=imb, fold=fold,
                         precision=precision_score(yt, p, zero_division=0),
                         recall=recall_score(yt, p, zero_division=0),
                         f1=f1_score(yt, p, zero_division=0),
                         auc=roc_auc_score(yt, s)))
    return rows


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj


def wide_auc(res, scheme, fset, imb):
    sub = res[(res.scheme == scheme) & (res.features == fset) & (res.imbalance == imb)]
    return sub.pivot(index="fold", columns="model", values="auc")


def stats_report(res):
    print("\n" + "=" * 70 + "\nSTATISTICAL TESTS (AUC per fold, REDUCED features)\n" + "=" * 70)
    print("Note: repeated-CV folds overlap, so p-values are optimistic. Treat as supporting evidence.")
    for scheme in ("pooled", "loro"):
        for imb in ("cw", "smote"):
            w = wide_auc(res, scheme, "REDUCED", imb)
            if w.shape[1] < 3:
                continue
            chi, p = friedmanchisquare(*[w[c] for c in w])
            best = w.mean().idxmax()
            print(f"\n[{scheme} / {imb}] Friedman chi2={chi:.2f}, p={p:.4f}; best={best}")
            others = [c for c in w if c != best]
            ps = []
            for c in others:
                try:
                    ps.append(wilcoxon(w[best], w[c]).pvalue)
                except ValueError:
                    ps.append(1.0)
            for c, raw, adj in zip(others, ps, holm(ps)):
                print(f"   {best} vs {c:20} p={raw:.4f}  Holm-adjusted={adj:.4f}")

    print("\nDoes anything beat file size alone? (Wilcoxon on AUC, imbalance=cw)")
    for model in ("LogisticRegression", "RandomForest"):
        for scheme in ("pooled", "loro"):
            sub = res[(res.scheme == scheme) & (res.model == model) & (res.imbalance == "cw")]
            piv = sub.pivot(index="fold", columns="features", values="auc")
            if "SIZE" not in piv:
                continue
            for other in ("REDUCED", "ALL14", "STATIC", "PROCESS"):
                if other in piv:
                    try:
                        p = wilcoxon(piv[other], piv["SIZE"]).pvalue
                    except ValueError:
                        p = 1.0
                    d = (piv[other] - piv["SIZE"]).mean()
                    print(f"   {model:19}{scheme:7} {other:8} - SIZE: mean AUC diff {d:+.3f}, p={p:.4f}")


def grouped_importance(df):
    print("\n" + "=" * 70 + "\nRQ2: GROUPED PERMUTATION IMPORTANCE (AUC drop on held-out repos)\n" + "=" * 70)
    X, y, rng, out = df[FEATURES], df["defect_broad"], np.random.default_rng(SEED), []
    for tr, te in LeaveOneGroupOut().split(X, y, df.repo):
        repo = df.repo.iloc[te[0]].split("/")[1]
        if y.iloc[te].sum() < 10:
            continue                         # too few positives for a reliable AUC
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        rf.fit(X.iloc[tr], y.iloc[tr])
        Xte, yte = X.iloc[te], y.iloc[te]
        base = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
        for g, cols in IMPORTANCE_GROUPS.items():
            drops = []
            for _ in range(10):
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].values[rng.permutation(len(Xte))]
                drops.append(base - roc_auc_score(yte, rf.predict_proba(Xp)[:, 1]))
            out.append(dict(repo=repo, group=g, auc_drop=np.mean(drops)))
    imp = pd.DataFrame(out).pivot(index="group", columns="repo", values="auc_drop")
    imp["MEAN"] = imp.mean(axis=1)
    imp = imp.sort_values("MEAN", ascending=False).round(3)
    imp.to_csv("results/feature_importance.csv")
    print(imp)


def vif(df, cols):
    C = np.corrcoef(np.log1p(df[cols]).values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(C)), index=cols).round(1)


def main():
    df = pd.read_csv("data/processed/all_repos.csv")
    print("VIF of REDUCED set (log1p):\n", vif(df, SETS["REDUCED"]).to_string())

    res = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    done = set(zip(res.scheme, res.features, res.model, res.imbalance)) if len(res) else set()

    configs = [("REDUCED", m, imb) for m in MODELS for imb in ("cw", "smote")]
    configs += [(f, m, "cw") for m in ("LogisticRegression", "RandomForest")
                for f in ("SIZE", "STATIC", "PROCESS", "REDUCED_NO_PRIOR", "ALL14")]

    for fset, model, imb in configs:
        for scheme in ("pooled", "loro"):
            if (scheme, fset, model, imb) in done:
                continue
            t0 = time.time()
            res = pd.concat([res, pd.DataFrame(run_config(df, fset, model, imb, scheme))],
                            ignore_index=True)
            res.to_csv(OUT, index=False)     # saved after every config, so you can resume
            a = res[(res.scheme == scheme) & (res.features == fset) &
                    (res.model == model) & (res.imbalance == imb)].auc.mean()
            print(f"{scheme:7}{fset:17}{model:19}{imb:6} AUC={a:.3f}  ({time.time()-t0:.0f}s)")

    cols = ["precision", "recall", "f1", "auc"]
    summ = res.groupby(["scheme", "features", "model", "imbalance"])[cols].agg(["mean", "std"]).round(3)
    summ.to_csv("results/model_summary.csv")
    for scheme in ("pooled", "loro"):
        print("\n" + "=" * 70 + f"\nMAIN COMPARISON ({scheme}, REDUCED features)\n" + "=" * 70)
        print(res[(res.scheme == scheme) & (res.features == "REDUCED")]
              .groupby(["model", "imbalance"])[cols].mean().round(3))
        print(f"\nABLATION ({scheme}, class weights): mean AUC by feature set")
        ab = res[(res.scheme == scheme) & (res.imbalance == "cw") &
                 (res.model.isin(["LogisticRegression", "RandomForest"]))]
        print(ab.pivot_table(index="features", columns="model", values="auc").round(3))
    print("\nLORO AUC per held-out repo (REDUCED, class weights):")
    print(res[(res.scheme == "loro") & (res.features == "REDUCED") & (res.imbalance == "cw")]
          .pivot(index="fold", columns="model", values="auc").round(2))

    stats_report(res)
    grouped_importance(df)


if __name__ == "__main__":
    class Tee:
        def __init__(self, *s): self.s = s
        def write(self, x): [i.write(x) for i in self.s]
        def flush(self): [i.flush() for i in self.s]
    sys.stdout = Tee(sys.__stdout__, open("results/train_report.txt", "a", encoding="utf-8"))
    main()