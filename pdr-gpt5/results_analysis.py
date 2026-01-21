# One-stop Python script to load the CSV, compute summaries, and generate multiple Matplotlib charts.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- CONFIG ----------
CSV_PATH = "results_with_satisfaction.csv"
OUT_DIR = Path("figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LOAD & CLEAN ----------
df = pd.read_csv(CSV_PATH)

# Normalize method/model labels for grouping
def norm_method(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if s in ["adhoc", "ad-hoc", "ad_hoc"]:
        return "Ad Hoc"
    if s in ["pdr"]:
        return "PDR"
    if s in ["pdrcritic", "pdr+critic", "pdr_plus_critic", "pdr_critic"]:
        return "PDR+Critic"
    # fallback to original
    return str(x)

def norm_model(x: str) -> str:
    s = str(x).strip().lower()
    if "4o" in s:
        return "GPT-4o"
    if "5" in s:
        return "GPT-5"
    return str(x)

df["Method"] = df["method"].map(norm_method)
df["Model"] = df["model"].map(norm_model)

# Keep only rows with a recognized method & model
df = df[df["Method"].isin(["Ad Hoc", "PDR", "PDR+Critic"])]
df = df[df["Model"].isin(["GPT-4o", "GPT-5"])]

# Ensure numeric
for col in ["iteration_count", "time_spent_sec", "final_score", "satisfaction_score",
            "perceived_quality", "usability_score", "expert_correctness_score", "expert_style_score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- SUMMARY TABLES ----------
metrics = ["iteration_count", "time_spent_sec", "final_score", "satisfaction_score"]
summary = (df.groupby(["Model", "Method"])[metrics]
             .agg(["mean", "std", "count"])
             .reset_index())
# Flatten columns
summary.columns = ["Model", "Method"] + ["_".join(col).strip() for col in summary.columns[2:]]
summary_path = OUT_DIR / "summary_by_model_method.csv"
summary.to_csv(summary_path, index=False)

# ---------- Helper: grouped bar plot ----------
def grouped_bar(metric_key: str, y_label: str, title: str, filename: str):
    pivot_mean = df.pivot_table(index="Method", columns="Model", values=metric_key, aggfunc="mean")
    pivot_std  = df.pivot_table(index="Method", columns="Model", values=metric_key, aggfunc="std")

    # Order methods
    order = ["Ad Hoc", "PDR", "PDR+Critic"]
    pivot_mean = pivot_mean.reindex(order)
    pivot_std  = pivot_std.reindex(order)

    x = np.arange(len(order))
    models = list(pivot_mean.columns)

    # width split evenly by number of models
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, m in enumerate(models):
        means = pivot_mean[m].values
        errs  = pivot_std[m].values
        ax.bar(x + (i - (len(models)-1)/2)*width, means, width, yerr=errs, capsize=4, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

# ---------- Charts ----------
paths = []

# 1) Quality score grouped bar
paths.append(grouped_bar(
    metric_key="final_score",
    y_label="Final Quality Score",
    title="Quality Scores: Ad Hoc vs PDR vs PDR+Critic",
    filename="quality_grouped_bar.png"
))

# 2) Runtime grouped bar
paths.append(grouped_bar(
    metric_key="time_spent_sec",
    y_label="Runtime (seconds)",
    title="Runtime: Ad Hoc vs PDR vs PDR+Critic",
    filename="runtime_grouped_bar.png"
))

# 3) Iterations grouped bar
paths.append(grouped_bar(
    metric_key="iteration_count",
    y_label="Iterations",
    title="Iterations to Convergence",
    filename="iterations_grouped_bar.png"
))

# 4) Line chart: iterations by method for each model
def line_by_method(metric_key: str, y_label: str, title: str, filename: str):
    means = df.groupby(["Model", "Method"])[metric_key].mean().reset_index()
    # Ensure correct method order
    method_order = {"Ad Hoc": 0, "PDR": 1, "PDR+Critic": 2}
    means["method_order"] = means["Method"].map(method_order)
    fig, ax = plt.subplots(figsize=(12, 7))
    for model, g in means.sort_values("method_order").groupby("Model"):
        ax.plot(g["method_order"], g[metric_key], marker="o", label=model)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Ad Hoc", "PDR", "PDR+Critic"])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    out = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

paths.append(line_by_method("time_spent_sec", "Runtime (s)", "Average Runtime per Task", "runtime_line.png"))
paths.append(line_by_method("final_score", "Final Quality Score", "Final Quality Scores", "quality_line.png"))

# 5) Boxplots for score distribution by method (per model)
def boxplot_by_method(metric_key: str, y_label: str, title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, model in enumerate(["GPT-4o", "GPT-5"]):
        dat = [df[(df["Model"]==model) & (df["Method"]==m)][metric_key].dropna()
               for m in ["Ad Hoc", "PDR", "PDR+Critic"]]
        axes[i].boxplot(dat, labels=["Ad Hoc", "PDR", "PDR+Critic"], showmeans=True)
        axes[i].set_title(model)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle(title)
    for ax in axes:
        ax.set_ylabel(y_label)
    out = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

paths.append(boxplot_by_method("final_score", "Final Quality Score", "Score Distributions by Method (per Model)", "scores_boxplot.png"))
paths.append(boxplot_by_method("time_spent_sec", "Runtime (s)", "Runtime Distributions by Method (per Model)", "runtime_boxplot.png"))

# 6) Task-level heatmap (mean scores per Model/Method/Task)
pivot_task = df.pivot_table(index="task_name", columns=["Model","Method"], values="final_score", aggfunc="mean")
# Sort tasks alphabetically for consistent display
pivot_task = pivot_task.sort_index()

fig, ax = plt.subplots(figsize=(14, max(6, 0.5*len(pivot_task.index))))
im = ax.imshow(pivot_task.values, aspect="auto")
ax.set_yticks(range(len(pivot_task.index)))
ax.set_yticklabels(pivot_task.index)
ax.set_xticks(range(len(pivot_task.columns)))
ax.set_xticklabels([f"{m}\n{meth}" for (m, meth) in pivot_task.columns], rotation=45, ha="right")
ax.set_title("Mean Final Score by Task × Model × Method")
fig.colorbar(im, ax=ax, label="Final Score")
out = OUT_DIR / "task_model_method_heatmap.png"
plt.tight_layout()
plt.savefig(out, dpi=200)
plt.close()
paths.append(out)

# 7) Save a compact textual report
report_path = OUT_DIR / "README.txt"
with open(report_path, "w") as f:
    f.write("Charts generated from results_with_satisfaction.csv\n")
    f.write(f"Summary CSV: {summary_path}\n")
    for p in paths:
        f.write(f"- {p}\n")

print("Saved files:")
for p in [summary_path, *paths, report_path]:
    print(p)
