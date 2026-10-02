import os, time
from dotenv import load_dotenv
from github import Github, Auth

load_dotenv()
gh = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))

for name in ["matplotlib/matplotlib", "psf/black", "python-poetry/poetry"]:
    print(f"\n{name}")
    for label in gh.get_repo(name).get_labels():
        if "bug" in label.name.lower() or "defect" in label.name.lower():
            n = gh.search_issues(f'repo:{name} is:issue is:closed label:"{label.name}"').totalCount
            time.sleep(2.5)
            print(f"  {label.name:35}{n:>6}")