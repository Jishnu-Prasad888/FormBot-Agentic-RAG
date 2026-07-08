import pandas as pd
from pathlib import Path

# Input and output files
csv_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"
output_file = "formatted_report_rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv.txt"

# Read CSV
df = pd.read_csv(csv_file)

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():

        qno = row.get("Question No", "")
        question = str(row.get("Question", "")).strip()
        expected = str(row.get("Expected Answer", "")).strip()
        generated = str(row.get("Generated Answer", "")).strip()
        context = str(row.get("Retrieved Context", "")).strip()

        f.write("=" * 70 + "\n")
        f.write(f"QUESTION {qno}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Question\n")
        f.write("-" * 8 + "\n")
        f.write(question + "\n\n")

        f.write("Expected Answer\n")
        f.write("-" * 15 + "\n")
        f.write(expected + "\n\n")

        f.write("Generated Answer\n")
        f.write("-" * 16 + "\n")
        f.write(generated + "\n\n")

        f.write("Retrieved Context\n")
        f.write("-" * 17 + "\n")
        f.write(context + "\n\n")

        f.write("Evaluation Metrics\n")
        f.write("-" * 18 + "\n")

        metric_cols = [
            "Accuracy (LLM)",
            "Faithfulness",
            "Context Precision",
            "Context Recall",
            "Answer Relevancy",
            "Exact Match",
            "Semantic Similarity",
            "F1 Score",
            "Accuracy (Combined)",
            "Recall@10",
            "Recall@20",
            "Recall@50",
            "MRR",
            "nDCG@10",
            "Gold Answer Found",
            "Latency (ms)"
        ]

        for col in metric_cols:
            if col in df.columns:
                f.write(f"{col}: {row[col]}\n")

        f.write("\nRationale\n")
        f.write("-" * 9 + "\n")

        rationale_cols = [
            "Accuracy Rationale",
            "Faithfulness Rationale",
            "Context Precision Rationale",
            "Context Recall Rationale",
            "Answer Relevancy Rationale"
        ]

        for col in rationale_cols:
            if col in df.columns and pd.notna(row[col]):
                f.write(f"\n{col.replace(' Rationale', '')}:\n")
                f.write(str(row[col]) + "\n")

        f.write("\n\n")

print(f"Formatted report saved to {output_file}")