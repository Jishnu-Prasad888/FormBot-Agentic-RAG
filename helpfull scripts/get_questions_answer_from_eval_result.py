from pathlib import Path
import pandas as pd

# Input CSV file
input_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/helpfull scripts/0.6accuracyandbelowrows_19th june added neo4j and qrant alogn with postgeres sql and elasticsearch overall score 64.9.csv"

# Read CSV
df = pd.read_csv(input_file)

# Keep only required columns
filtered_df = df[["Question No", "Question", "Expected Answer"]].copy()

# Renumber Question No sequentially
filtered_df["Question No"] = range(1, len(filtered_df) + 1)

# Create output filename
input_path = Path(input_file)
output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"

# Save
filtered_df.to_csv(output_file, index=False)

print(f"Saved {len(filtered_df)} rows.")
print(f"Output saved to: {output_file}")