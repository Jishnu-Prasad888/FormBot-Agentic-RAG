import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# CONFIG
# ==========================
CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/eval 11th june .csv"
CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"

QUESTION_ID_COL = "Question No"
QUESTION_COL = "Question"

METRICS = [
    "Accuracy (LLM)",
    "Faithfulness",
    "Context Precision",
    "Context Recall",
    "Answer Relevancy"
]

HEATMAP_FILE = "metric_change_heatmap.png"
MAPPING_FILE = "question_mapping.csv"

# ==========================
# LOAD CSVS
# ==========================
df1 = pd.read_csv(CSV_1)
df2 = pd.read_csv(CSV_2)

# ==========================
# VALIDATE COLUMNS
# ==========================
required_cols = [
    QUESTION_ID_COL,
    QUESTION_COL
] + METRICS

missing_1 = [
    c for c in required_cols
    if c not in df1.columns
]

missing_2 = [
    c for c in required_cols
    if c not in df2.columns
]

if missing_1:
    raise ValueError(
        f"Missing columns in first CSV: {missing_1}"
    )

if missing_2:
    raise ValueError(
        f"Missing columns in second CSV: {missing_2}"
    )

# ==========================
# KEEP REQUIRED COLUMNS
# ==========================
df1 = df1[required_cols].copy()
df2 = df2[required_cols].copy()

# ==========================
# CLEAN DATA
# ==========================
df1[QUESTION_COL] = (
    df1[QUESTION_COL]
    .astype(str)
    .str.strip()
)

df2[QUESTION_COL] = (
    df2[QUESTION_COL]
    .astype(str)
    .str.strip()
)

for metric in METRICS:
    df1[metric] = pd.to_numeric(
        df1[metric],
        errors="coerce"
    )

    df2[metric] = pd.to_numeric(
        df2[metric],
        errors="coerce"
    )

# ==========================
# RENAME METRICS
# ==========================
df1 = df1.rename(
    columns={
        metric: f"{metric}_old"
        for metric in METRICS
    }
)

df2 = df2.rename(
    columns={
        metric: f"{metric}_new"
        for metric in METRICS
    }
)

# ==========================
# MERGE ON QUESTION
# ==========================
merged = pd.merge(
    df1,
    df2,
    on=[
        QUESTION_ID_COL,
        QUESTION_COL
    ],
    how="inner"
)

print(
    f"Common questions found: {len(merged)}"
)

if len(merged) == 0:
    raise ValueError(
        "No common questions found."
    )

# ==========================
# SAVE QUESTION MAPPING
# ==========================
mapping_df = merged[
    [
        QUESTION_ID_COL,
        QUESTION_COL
    ]
].copy()

mapping_df = mapping_df.sort_values(
    by=QUESTION_ID_COL
)

mapping_df.to_csv(
    MAPPING_FILE,
    index=False
)

print(
    f"Saved mapping file: {MAPPING_FILE}"
)

# ==========================
# BUILD HEATMAP DATA
# ==========================
heatmap_df = pd.DataFrame()

for metric in METRICS:
    heatmap_df[metric] = (
        merged[f"{metric}_new"]
        - merged[f"{metric}_old"]
    )

heatmap_df.index = (
    "Q" +
    merged[QUESTION_ID_COL]
    .astype(str)
)

# ==========================
# SORT BY TOTAL IMPROVEMENT
# ==========================
heatmap_df["Total"] = (
    heatmap_df.sum(axis=1)
)

heatmap_df = heatmap_df.sort_values(
    by="Total",
    ascending=False
)

heatmap_df = heatmap_df.drop(
    columns=["Total"]
)

# ==========================
# PLOT HEATMAP
# ==========================
num_questions = len(heatmap_df)

fig_height = max(
    8,
    num_questions * 0.35
)

plt.figure(
    figsize=(12, fig_height)
)

sns.heatmap(
    heatmap_df,
    cmap="RdYlGn",
    center=0,
    annot=False,      # prevents clutter
    linewidths=0.3,
    cbar_kws={
        "label": "Metric Change (New - Old)"
    }
)

plt.title(
    "Metric Improvements by Question",
    fontsize=16,
    pad=20
)

plt.xlabel(
    "Metrics",
    fontsize=12
)

plt.ylabel(
    "Question Number",
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    HEATMAP_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved heatmap: {HEATMAP_FILE}"
)

# ==========================
# SUMMARY
# ==========================
print("\nAverage Metric Changes")

for metric in METRICS:
    avg_change = (
        heatmap_df[metric]
        .mean()
    )

    print(
        f"{metric}: {avg_change:+.4f}"
    )

print("\nDone.")