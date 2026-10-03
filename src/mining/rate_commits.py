import os, subprocess
import pandas as pd
from config import CLONE_DIR

OUT = "results/label_audit.csv"
d = pd.read_csv(OUT, dtype=str).fillna("")


def git(repo, *args):
    return subprocess.run(["git", "-C", os.path.join(CLONE_DIR, repo), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


todo = [i for i in d.index
        if d.at[i, "repo"] != "httpx" and d.at[i, "is_real_bug_fix"].strip() == ""]
input(f"{len(todo)} rows to rate.\n"
      "Keys: y = real bug fix, n = not a bug fix, s = can't tell (stays blank), q = save and quit.\n"
      "Press Enter to start...")

for k, i in enumerate(todo, 1):
    r = d.loc[i]
    os.system("cls")
    print(f"[{k}/{len(todo)}]  {r.repo}  {r.hash}")
    print(f"\nSUBJECT: {r.subject}\n")
    print(git(r.repo, "show", "--stat", "--format=", r.hash)[:1200])
    diff = git(r.repo, "show", "--format=", "-U0", r.hash, "--", "*.py")
    changed = [l for l in diff.splitlines()
               if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    print("\n".join(l[:130] for l in changed[:40]))
    if len(changed) > 40:
        print(f"... {len(changed) - 40} more changed lines")
    while True:
        a = input("\n[y/n/s/q] > ").strip().lower()
        if a in ("y", "n", "s", "q"):
            break
    if a == "q":
        break
    if a != "s":
        d.at[i, "is_real_bug_fix"] = a
        d.to_csv(OUT, index=False)      # saved after every answer

d.to_csv(OUT, index=False)
print("Saved. Run: python src\\mining\\label_audit.py score")