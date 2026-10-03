import argparse, ast, json, os, subprocess, sys
from collections import defaultdict

import joblib
import pandas as pd
from radon.complexity import cc_visit
from radon.raw import analyze

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mining"))
from mine_repo import EXCLUDE_RE, fan_out, get_log, git, is_bug_fix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("repo_path")
ap.add_argument("--base", help="git ref; only report files changed since it")
ap.add_argument("--top", type=float, default=0.20, help="share of files to flag")
ap.add_argument("--size-only", action="store_true", help="use the size-only baseline")
a = ap.parse_args()

card = json.load(open("models/model_card.json"))
model = joblib.load("models/defect_lr_size_only.joblib" if a.size_only
                    else "models/defect_lr_reduced.joblib")
cols = card["size_features"] if a.size_only else card["features"]

files = [p for p in git(a.repo_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
         if p.endswith(".py") and not EXCLUDE_RE.search(p)]
tops = set()
for p in files:
    parts = p.split("/")
    if parts[0] in ("src", "lib"):
        parts = parts[1:]
    tops.add(parts[0].removesuffix(".py"))

stats = defaultdict(lambda: {"lines_added": 0, "authors": set(), "prior_defects": 0})
for c in get_log(a.repo_path):
    fix = is_bug_fix(c["msg"].split("\n")[0])
    for added, _, p in c["files"]:
        s = stats[p]
        s["lines_added"] += added
        s["authors"].add(c["author"])
        s["prior_defects"] += fix

rows = []
for p in files:
    src = git(a.repo_path, "show", f"HEAD:{p}")
    try:
        tree, raw, blocks = ast.parse(src), analyze(src), cc_visit(src)
    except Exception:
        continue
    s = stats.get(p)
    rows.append({"file": p, "loc": raw.loc, "lloc": raw.lloc, "comments": raw.comments,
                 "cc_max": max([b.complexity for b in blocks] or [0]),
                 "fan_out": fan_out(tree, tops),
                 "lines_added": s["lines_added"] if s else 0,
                 "n_authors": len(s["authors"]) if s else 0,
                 "prior_defects": s["prior_defects"] if s else 0})

df = pd.DataFrame(rows)
df["risk"] = model.predict_proba(df[cols])[:, 1]
df["rank_pct"] = df.risk.rank(pct=True)
df["flag"] = df.rank_pct > 1 - a.top

view = df
if a.base:
    changed = set(git(a.repo_path, "diff", "--name-only", f"{a.base}...HEAD").splitlines())
    view = df[df.file.isin(changed)]

view = view.sort_values("risk", ascending=False)
print(f"Scored {len(df)} files ({'size-only' if a.size_only else 'reduced'} model). "
      f"Flagged = top {a.top:.0%} of files by risk.\n")
print(view[["file", "risk", "rank_pct", "flag"]].round(3).head(25).to_string(index=False))
view.to_csv("results/ci_scores.csv", index=False)