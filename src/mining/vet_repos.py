import os, time
from dotenv import load_dotenv
from github import Github, Auth

load_dotenv()
gh = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))

CANDIDATES = {
    "pallets/flask": "web",
    "psf/requests": "web",
    "encode/httpx": "web",
    "scrapy/scrapy": "data",
    "scikit-learn/scikit-learn": "data",
    "matplotlib/matplotlib": "data",
    "psf/black": "tools",
    "pytest-dev/pytest": "tools",
    "python-poetry/poetry": "tools",
    "pypa/pip": "tools",
}
BUG_LABELS = ["bug", "type: bug", "kind: bug", "Type: Bug"]

print(f"{'repo':32}{'domain':8}{'stars':>8}  {'last push':12}{'bug label':14}{'closed bugs':>11}")
for name, domain in CANDIDATES.items():
    repo = gh.get_repo(name)
    best_label, best_n = None, 0
    for label in BUG_LABELS:
        n = gh.search_issues(f'repo:{name} is:issue is:closed label:"{label}"').totalCount
        time.sleep(2.5)  # search API rate limit
        if n > best_n:
            best_label, best_n = label, n
    print(f"{name:32}{domain:8}{repo.stargazers_count:>8}  "
          f"{repo.pushed_at:%Y-%m-%d}  {str(best_label):14}{best_n:>11}")