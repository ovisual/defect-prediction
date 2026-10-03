import sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
SEED, REPEATS, TREES = 42, (1 if QUICK else 3), (100 if QUICK else 200)

REPORT = open("results/beyond_size_report.txt", "w", encoding="utf-8")
class Tee:
    def write(self, x): sys.__stdout__.write(x); REPORT.write(x)
    def flush(self): sys.__stdout__.flush(); REPORT.flush()
sys.stdout = Tee()


def add_features(df):
    d = df.copy()
    L = d["loc"].clip(lower=1)
    C = d["n_commits"].clip(lower=1)
    d["commits_per_loc"] = d["n_commits"] / L
    d["added_per_loc"] = d["lines_added"] / L
    d["deleted_per_loc"] = d["lines_deleted"] / L
    d["fix_ratio"] = d["prior_defects"] / C
    d["authors_per_commit"] = d["n_authors"] / C
    g = d.groupby("repo")
    for c in ["loc", "cc_sum", "n_commits", "lines_added", "lines_deleted",
              "n_authors", "prior_defects", "fan_out"]:
        d["pct_" + c] = g[c].rank(pct=True)
    return d


RATES = ["commits_per_loc", "added_per_loc", "deleted_per_loc", "fix_ratio", "authors_per_commit"]
PCT_CHURN = ["pct_n_commits", "pct_lines_added", "pct_lines_deleted", "pct_n_authors", "pct_prior_defects"]
SETS = {
    "SIZE":              ["loc"],
    "SIZE_REL":          ["pct_loc"],
    "SIZE+RATES":        ["loc"] + RATES,
    "SIZE+PCT_CHURN":    ["loc"] + PCT_CHURN,
    "SIZE+COMPLEXITY":   ["loc", "cc_max", "cc_mean", "comments"],
    "SIZE+FANOUT":       ["loc", "fan_out"],
    "REDUCED":           ["lloc", "cc_max", "comments", "fan_out", "lines_added", "n_authors", "prior_defects"],
    "SIZE+ALL_RELATIVE": ["loc", "pct_cc_sum", "pct_fan_out"] + RATES + PCT_CHURN,
}


def make_model(kind):
    if kind == "LR":
        return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return RandomForestClassifier(n_estimators=TREES, min_samples_leaf=3, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def recall_at(y, s, frac=0.2):
    k = max(1, int(round(frac * len(y))))
    return y[np.argsort(-s)[:k]].sum() / max(1, y.sum())


def strata_auc(y, s, size_pct):
    bins = pd.cut(size_pct, [0, 1 / 3, 2 / 3, 1.0], labels=False, include_lowest=True)
    vals, w = [], []
    for b in range(3):
        m = bins == b
        if y[m].sum() >= 3 and (1 - y[m]).sum() >= 3:
            vals.append(roc_auc_score(y[m], s[m])); w.append(m.sum())
    return np.average(vals, weights=w) if vals else np.nan


def evaluate(df, cols, kind, target, scheme, min_pos=3):
    X, y, sp = df[cols].values, df[target].values, df["pct_loc"].values
    if scheme == "pooled":
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=REPEATS, random_state=SEED)
        splits = [(f"f{i:02d}", tr, te) for i, (tr, te) in enumerate(cv.split(X, y))]
    else:
        splits = [(df.repo.iloc[te[0]].split("/")[1], tr, te)
                  for tr, te in LeaveOneGroupOut().split(X, y, df.repo)]
    rows = []
    for fold, tr, te in splits:
        yt = y[te]
        if yt.sum() < min_pos or yt.sum() == len(yt):
            continue
        s = make_model(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        rows.append(dict(scheme=scheme, model=kind, features=None, fold=fold,
                         auc=roc_auc_score(yt, s), strata_auc=strata_auc(yt, s, sp[te]),
                         recall20=recall_at(yt, s)))
    return rows


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def compare_to_size(res, metric):
    for (scheme, model), g in res.groupby(["scheme", "model"]):
        piv = g.pivot(index="fold", columns="features", values=metric)
        others = [c for c in piv if c != "SIZE"]
        diffs, ps = [], []
        for c in others:
            pair = piv[[c, "SIZE"]].dropna()
            diffs.append((pair[c] - pair["SIZE"]).mean())
            try: ps.append(wilcoxon(pair[c], pair["SIZE"]).pvalue)
            except ValueError: ps.append(1.0)
        print(f"\n[{scheme} / {model}] {metric}: mean SIZE = {piv['SIZE'].mean():.3f}")
        for c, d, p, a in zip(others, diffs, ps, holm(ps)):
            print(f"   {c:18} diff {d:+.3f}   p={p:.4f}   Holm={a:.4f}")


def main():
    df = add_features(pd.read_csv("data/processed/all_repos.csv"))
    all_rows = []
    for fset, cols in SETS.items():
        for kind in ("LR", "RF"):
            for scheme in ("pooled", "loro"):
                r = evaluate(df, cols, kind, "defect_broad", scheme)
                for x in r: x["features"] = fset
                all_rows += r
        print("done", fset)
    res = pd.DataFrame(all_rows)
    res.to_csv("results/beyond_size_folds.csv", index=False)

    metrics = ["auc", "strata_auc", "recall20"]
    for scheme in ("pooled", "loro"):
        print("\n" + "=" * 72 + f"\nSUMMARY ({scheme}, label = defect_broad)\n" + "=" * 72)
        print(res[res.scheme == scheme].groupby(["model", "features"])[metrics].mean().round(3))
    print("\n" + "=" * 72 + "\nPAIRED TESTS AGAINST SIZE ALONE (Holm-corrected within each block)\n" + "=" * 72)
    print("Pooled folds overlap, so pooled p-values are optimistic; LORO has only 8 folds, so low power.")
    for m in metrics:
        compare_to_size(res, m)

    print("\n" + "=" * 72 + "\nLORO AUC PER HELD-OUT REPO (Logistic Regression)\n" + "=" * 72)
    print(res[(res.scheme == "loro") & (res.model == "LR")]
          .pivot(index="fold", columns="features", values="auc").round(2))
    print("\nLORO size-stratified AUC per repo (LR):")
    print(res[(res.scheme == "loro") & (res.model == "LR")]
          .pivot(index="fold", columns="features", values="strata_auc").round(2))

    # robustness: strict label, repos where it works, test repos with >= 10 positives
    print("\n" + "=" * 72 + "\nROBUSTNESS: STRICT LABEL (requests excluded, LORO)\n" + "=" * 72)
    ds = df[df.repo != "psf/requests"].reset_index(drop=True)
    rows = []
    for fset in ("SIZE", "SIZE+RATES", "SIZE+PCT_CHURN", "REDUCED"):
        for kind in ("LR", "RF"):
            for x in evaluate(ds, SETS[fset], kind, "defect_strict", "loro", min_pos=10):
                x["features"] = fset; rows.append(x)
    st = pd.DataFrame(rows)
    print(st.groupby(["model", "features"])[metrics].mean().round(3))
    st.to_csv("results/beyond_size_strict.csv", index=False)
    print("\nReport saved to results/beyond_size_report.txt")


if __name__ == "__main__":
    main()
    REPORT.close()