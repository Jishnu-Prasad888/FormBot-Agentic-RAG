from pathlib import Path

import pandas as pd

# Input CSV file
input_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/19th june added neo4j and qrant alogn with postgeres sql and elasticsearch overall score 64.9.csv"

# Read CSV
df = pd.read_csv(input_file)

# Filter rows where Accuracy <= 0.6
filtered_df = df[df["Accuracy (LLM)"] <= 0.6]

# Create output filename
input_path = Path(input_file)
output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"

# Save filtered rows
filtered_df.to_csv(output_file, index=False)

print(f"Filtered {len(filtered_df)} rows.")
print(f"Output saved to: {output_file}")
