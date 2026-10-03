import os, random, sys
import pandas as pd
from config import REPOS, CLONE_DIR
from mine_repo import get_log, is_bug_fix, CUTOFF, END

OUT = "results/label_audit.csv"

if len(sys.argv) > 1 and sys.argv[1] == "score":
    d = pd.read_csv(OUT).dropna(subset=["is_real_bug_fix"])
    d["ok"] = d.is_real_bug_fix.astype(str).str.lower().str.startswith(("y", "1"))
    print(f"Rated {len(d)} commits. Heuristic precision: {d.ok.mean():.1%}")
    n, p = len(d), d.ok.mean()
    se = (p * (1 - p) / n) ** 0.5
    print(f"Approx 95% CI: {p - 1.96 * se:.1%} to {p + 1.96 * se:.1%}")
    print(d.groupby("repo").ok.agg(["mean", "count"]).round(2))
else:
    random.seed(42)
    rows = []
    for full in REPOS:
        short = full.split("/")[1]
        path = os.path.join(CLONE_DIR, short)
        if not os.path.isdir(path):
            continue
        commits = get_log(path, f"--after={CUTOFF}", f"--before={END}", newest_first=False)
        flagged = [c for c in commits if is_bug_fix(c["msg"].split("\n")[0])]
        for c in random.sample(flagged, min(10, len(flagged))):
            rows.append(dict(repo=short, hash=c["hash"][:10],
                             subject=c["msg"].split("\n")[0][:110], is_real_bug_fix=""))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} commits to {OUT}")