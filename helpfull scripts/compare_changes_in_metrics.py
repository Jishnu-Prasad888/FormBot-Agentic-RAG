import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# CONFIG
# ==========================
CSV_1 = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\common among removed and 0.6 less questions.csv"
CSV_2 = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june removed and common 0.6 accuracy with chatgpt suggested code changes overall scor is 27 some questions seeding imporment.csv"

METRICS = [
    "Accuracy",
    "Faithfulness",
    "Context Precision",
    "Context Recall",
    "Answer Relevancy"
]

# ==========================
# LOAD FILES
# ==========================
df1 = pd.read_csv(CSV_1)
df2 = pd.read_csv(CSV_2)

# Keep only needed columns
cols = ["Question"] + METRICS

df1 = df1[cols].copy()
df2 = df2[cols].copy()

# Rename metric columns so we know which file they came from
df1 = df1.rename(
    columns={m: f"{m}_old" for m in METRICS}
)

df2 = df2.rename(
    columns={m: f"{m}_new" for m in METRICS}
)

# ==========================
# FIND COMMON QUESTIONS
# ==========================
merged = pd.merge(
    df1,
    df2,
    on="Question",
    how="inner"
)

print(f"Common questions found: {len(merged)}")

# ==========================
# CREATE DELTA COLUMNS
# ==========================
for metric in METRICS:
    merged[f"{metric}_change"] = (
        merged[f"{metric}_new"] -
        merged[f"{metric}_old"]
    )

# ==========================
# PLOT SETTINGS
# ==========================
plt.style.use("ggplot")

for metric in METRICS:

    plot_df = merged[
        ["Question", f"{metric}_change"]
    ].copy()

    # Sort by change
    plot_df = plot_df.sort_values(
        by=f"{metric}_change"
    )

    changes = plot_df[f"{metric}_change"]
    questions = plot_df["Question"]

    colors = [
        "green" if x > 0 else "red"
        for x in changes
    ]

    fig_height = max(6, len(plot_df) * 0.35)

    plt.figure(figsize=(14, fig_height))

    bars = plt.barh(
        questions,
        changes,
        color=colors
    )

    plt.axvline(
        x=0,
        color="black",
        linewidth=1
    )

    # Annotate bars with delta values
    for bar, value in zip(bars, changes):
        plt.text(
            value,
            bar.get_y() + bar.get_height()/2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right"
        )

    plt.title(
        f"{metric}: Improvement / Regression by Question"
    )
    plt.xlabel(
        "Metric Change (New - Old)"
    )
    plt.ylabel("Question")

    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        + "_change.png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("Done.")