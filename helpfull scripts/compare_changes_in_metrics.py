import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/eval 11th june .csv" 
CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"

METRICS = [
    "Accuracy (LLM)",
    "Faithfulness",
    "Context Precision",
    "Context Recall",
    "Answer Relevancy"
]

QUESTION_COL = "Question"

# ==========================
# LOAD FILES
# ==========================
df1 = pd.read_csv(CSV_1)
df2 = pd.read_csv(CSV_2)

# ==========================
# VALIDATE COLUMNS
# ==========================
required_cols = [QUESTION_COL] + METRICS

missing_1 = [c for c in required_cols if c not in df1.columns]
missing_2 = [c for c in required_cols if c not in df2.columns]

if missing_1:
    raise ValueError(
        f"Missing columns in first CSV: {missing_1}"
    )

if missing_2:
    raise ValueError(
        f"Missing columns in second CSV: {missing_2}"
    )

# Keep only required columns
df1 = df1[required_cols].copy()
df2 = df2[required_cols].copy()

# Remove accidental whitespace in questions
df1[QUESTION_COL] = df1[QUESTION_COL].astype(str).str.strip()
df2[QUESTION_COL] = df2[QUESTION_COL].astype(str).str.strip()

# Convert metrics to numeric
for metric in METRICS:
    df1[metric] = pd.to_numeric(df1[metric], errors="coerce")
    df2[metric] = pd.to_numeric(df2[metric], errors="coerce")

# Rename metrics
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
    on=QUESTION_COL,
    how="inner"
)

print(f"Common questions found: {len(merged)}")

if len(merged) == 0:
    raise ValueError(
        "No common questions found between the two CSVs."
    )

# ==========================
# CREATE DELTA COLUMNS
# ==========================
for metric in METRICS:
    merged[f"{metric}_change"] = (
        merged[f"{metric}_new"]
        - merged[f"{metric}_old"]
    )

# ==========================
# PLOT SETTINGS
# ==========================
plt.style.use("ggplot")

for metric in METRICS:

    plot_df = merged[
        [QUESTION_COL, f"{metric}_change"]
    ].copy()

    plot_df = plot_df.dropna()

    plot_df = plot_df.sort_values(
        by=f"{metric}_change"
    )

    changes = plot_df[f"{metric}_change"]
    questions = plot_df[QUESTION_COL]

    colors = [
        "green" if x > 0 else "red"
        for x in changes
    ]

    fig_height = max(
        6,
        len(plot_df) * 0.35
    )

    plt.figure(
        figsize=(14, fig_height)
    )

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

    for bar, value in zip(
        bars,
        changes
    ):
        plt.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left"
            if value >= 0
            else "right"
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
        .replace("(", "")
        .replace(")", "")
        + "_change.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")

print("Done.")