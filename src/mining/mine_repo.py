import ast, json, os, re, subprocess, sys
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from github import Github, Auth
from radon.complexity import cc_visit
from radon.raw import analyze
from github import RateLimitExceededException

from config import REPOS, CLONE_DIR

CUTOFF = "2025-10-01"   # history before this date = features
END = "2026-10-01"      # fixes between CUTOFF and END = labels
KEYWORD_RE = re.compile(r"\b(fix(e[sd]?)?|bugs?|bugfix)\b", re.I)

NOISE_RE = re.compile(
    r"\b(typos?|docs?|documentation|readme|changelog|ci|lint\w*|pre-commit|"
    r"dependabot|bump|workflow|actions?|release|spelling|flake8|mypy|ruff|format\w*)\b", re.I)

def is_bug_fix(subject):
    return bool(KEYWORD_RE.search(subject)) and not NOISE_RE.search(subject)

EXCLUDE_RE = re.compile(
    r"(^|/)(tests?|testing|docs?|examples?|benchmarks?|_vendor|vendor|build|scripts|asv_bench)(/|$)"
    r"|(^|/)(setup|conftest)\.py$|(^|/)test_[^/]*\.py$|_test\.py$")

load_dotenv()
gh = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]), timeout=30, retry=None)
def git(path, *args):
    return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def parse_log(text):
    commits = []
    for chunk in text.split("\x01")[1:]:
        head, _, rest = chunk.partition("\x03")
        h, author, msg = head.split("\x02", 2)
        files = []
        for line in rest.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                a, d, p = parts
                files.append((int(a) if a.isdigit() else 0, int(d) if d.isdigit() else 0, p))
        commits.append({"hash": h, "author": author, "msg": msg.strip(), "files": files})
    return commits


def parse_log(text):
    commits = []
    for chunk in text.split("\x01")[1:]:
        head, _, rest = chunk.partition("\x03")
        h, author, msg = head.split("\x02", 2)
        files, toks, i = [], rest.split("\0"), 0
        while i < len(toks):
            parts = toks[i].lstrip("\n").split("\t")
            if len(parts) == 3:
                a, d, p = parts
                a = int(a) if a.isdigit() else 0
                d = int(d) if d.isdigit() else 0
                if p == "":                      # rename: next two tokens are old, new
                    files.append((a, d, toks[i + 2], toks[i + 1]))
                    i += 3
                    continue
                files.append((a, d, p, None))
            i += 1
        commits.append({"hash": h, "author": author, "msg": msg.strip(), "files": files})
    return commits


def get_log(path, *args, newest_first=True):
    fmt = "--format=%x01%H%x02%an%x02%B%x03"
    extra = [] if newest_first else ["--reverse"]
    commits = parse_log(git(path, "log", "HEAD", "--no-merges", "-M", "-z",
                            "--numstat", fmt, *extra, *args))
    alias = {}
    for c in commits:                            # follow renames
        out = []
        for a, d, p, old in c["files"]:
            if newest_first:                     # map old names to the newest name
                cur = alias.get(p, p)
                if old:
                    alias[old] = cur
            else:                                # map new names back to the original name
                cur = alias.get(old, old) if old else alias.get(p, p)
                if old:
                    alias[p] = cur
            out.append((a, d, cur))
        c["files"] = out
    return commits


def fan_out(tree, tops):
    deps = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in tops:
                    deps.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            dots = "." * n.level
            if n.level > 0:
                if n.module:
                    deps.add(dots + n.module)
                else:
                    deps.update(dots + a.name for a in n.names)
            elif n.module and n.module.split(".")[0] in tops:
                deps.add(n.module)
    return len(deps)


def main(short):
    full = next(k for k in REPOS if k.endswith("/" + short))
    cfg = REPOS[full]
    path = os.path.join(CLONE_DIR, short)
    cache_file = os.path.join("data", "raw", f"{short}_issue_cache.json")
    cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
    repo = gh.get_repo(full)

    CLOSE_RE = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*#(\d+)", re.I)

    def labels_of(num):
        if num in cache:
            return cache[num]
        try:
            issue = repo.get_issue(int(num))
            labs = {l.name.lower() for l in issue.labels}
            if issue.pull_request is not None:          # it's a PR: follow its closing links
                for n in set(CLOSE_RE.findall(issue.body or ""))  :
                    labs |= {l.name.lower() for l in repo.get_issue(int(n)).labels}
            cache[num] = sorted(labs)
            
        except RateLimitExceededException:
            raise
        except Exception as e:
            print(f"    API error for #{num}: {e}")
            return []
        return cache[num]

    cutoff_commit = git(path, "rev-list", "-1", f"--before={CUTOFF}", "HEAD").strip()
    print(f"{short}: cutoff commit {cutoff_commit[:8]}")

    # files that exist at the cutoff
    all_py = [p for p in git(path, "ls-tree", "-r", "--name-only", cutoff_commit).splitlines()
              if p.endswith(".py")]
    files = [p for p in all_py if not EXCLUDE_RE.search(p)]
    tops = set()
    for p in files:
        parts = p.split("/")
        if parts[0] in ("src", "lib"):
            parts = parts[1:]
        tops.add(parts[0].removesuffix(".py"))
    print(f"  {len(files)} source files, project packages: {sorted(tops)}")

    # history before cutoff -> churn and prior defects
    stats = defaultdict(lambda: {"n_commits": 0, "lines_added": 0, "lines_deleted": 0,
                                 "authors": set(), "prior_defects": 0})
    for c in get_log(path, f"--before={CUTOFF}"):
        is_fix = is_bug_fix(c["msg"].split("\n")[0])
        for a, d, p in c["files"]:
            s = stats[p]
            s["n_commits"] += 1
            s["lines_added"] += a
            s["lines_deleted"] += d
            s["authors"].add(c["author"])
            s["prior_defects"] += is_fix

    # fixes after cutoff -> labels
    strict, broad = set(), set()
    post = get_log(path, f"--after={CUTOFF}", f"--before={END}", newest_first=False)
    print(f"  {len(post)} commits in label window (checking issue labels via API...)")
    for i, c in enumerate(post, 1):
        refs = sorted(set(re.findall(r"#(\d+)", c["msg"])))[:5]
        verified = any(cfg["bug_label"].lower() in labels_of(n) for n in refs)
        heuristic = is_bug_fix(c["msg"].split("\n")[0])
        for _, _, p in c["files"]:
            if verified:
                strict.add(p)
            if verified or heuristic:
                broad.add(p)
                
        if i % 25 == 0:
            json.dump(cache, open(cache_file, "w"))
        if i % 200 == 0:
            print(f"    {i}/{len(post)}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    json.dump(cache, open(cache_file, "w"))

    # static metrics at the cutoff commit
    rows = []
    for p in files:
        src = git(path, "show", f"{cutoff_commit}:{p}")
        try:
            tree = ast.parse(src)
            raw = analyze(src)
            blocks = cc_visit(src)
        except Exception:
            continue  # unparsable (e.g. old syntax) -> skip
        cc = [b.complexity for b in blocks] or [0]
        s = stats.get(p)
        rows.append({
            "repo": full, "domain": cfg["domain"], "file": p,
            "loc": raw.loc, "sloc": raw.sloc, "lloc": raw.lloc, "comments": raw.comments,
            "cc_sum": sum(cc), "cc_max": max(cc), "cc_mean": sum(cc) / len(cc),
            "n_functions": len(blocks), "fan_out": fan_out(tree, tops),
            "n_commits": s["n_commits"] if s else 0,
            "lines_added": s["lines_added"] if s else 0,
            "lines_deleted": s["lines_deleted"] if s else 0,
            "n_authors": len(s["authors"]) if s else 0,
            "prior_defects": s["prior_defects"] if s else 0,
            "defect_strict": int(p in strict),
            "defect_broad": int(p in broad),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join("data", "labelled"), exist_ok=True)
    out = os.path.join("data", "labelled", f"{short}.csv")
    df.to_csv(out, index=False)
    print(f"  saved {out}: {len(df)} files, "
          f"strict defects {df.defect_strict.mean():.1%}, broad defects {df.defect_broad.mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1])