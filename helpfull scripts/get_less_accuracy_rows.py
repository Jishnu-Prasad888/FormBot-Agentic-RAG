import pandas as pd
from pathlib import Path

# Input CSV file
input_file = "eval_number_clean 5th june.csv"

# Read CSV
df = pd.read_csv(input_file)

# Filter rows where Accuracy <= 0.6
filtered_df = df[df["Accuracy"] <= 0.6]

# Create output filename
input_path = Path(input_file)
output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"

# Save filtered rows
filtered_df.to_csv(output_file, index=False)

print(f"Filtered {len(filtered_df)} rows.")
print(f"Output saved to: {output_file}")