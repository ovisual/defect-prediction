import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SKIP = {"httpx.csv", "black.csv"}
files = [f for f in glob.glob("data/labelled/*.csv") if os.path.basename(f) not in SKIP]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

FEATURES = ["loc", "sloc", "lloc", "comments", "cc_sum", "cc_max", "cc_mean",
            "n_functions", "fan_out", "n_commits", "lines_added", "lines_deleted",
            "n_authors", "prior_defects"]

print("Shape:", df.shape)
print("\nMissing values:", int(df.isna().sum().sum()))
print("Duplicate (repo, file) rows:", int(df.duplicated(["repo", "file"]).sum()))

print("\nClass balance per repo:")
print(df.groupby("repo")[["defect_strict", "defect_broad"]].agg(["sum", "mean"]).round(3))
print("\nClass balance per domain:")
print(df.groupby("domain")[["defect_strict", "defect_broad"]].mean().round(3))
print("\nOverall: strict {:.1%}, broad {:.1%}".format(
    df.defect_strict.mean(), df.defect_broad.mean()))

print("\nFeature summary:")
print(df[FEATURES].describe().T[["mean", "50%", "max"]].round(1))

# plot 1: class balance per repo
bal = df.groupby("repo")[["defect_strict", "defect_broad"]].mean()
bal.plot(kind="bar", figsize=(9, 5))
plt.ylabel("Share of defect-prone files")
plt.title("Class balance per repository")
plt.tight_layout()
plt.savefig("results/class_balance.png", dpi=150)
plt.close()

# plot 2: correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[FEATURES + ["defect_broad"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Feature correlations")
plt.tight_layout()
plt.savefig("results/correlations.png", dpi=150)
plt.close()

# plot 3: feature distributions (log scale shows the skew)
fig, axes = plt.subplots(3, 5, figsize=(16, 9))
for ax, col in zip(axes.flat, FEATURES):
    sns.histplot(df[col], bins=40, ax=ax, log_scale=(False, True))
    ax.set_title(col)
plt.tight_layout()
plt.savefig("results/distributions.png", dpi=150)
plt.close()

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/all_repos.csv", index=False)
print("\nSaved data/processed/all_repos.csv")