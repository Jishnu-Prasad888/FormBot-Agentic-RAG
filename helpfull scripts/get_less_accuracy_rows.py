import pandas as pd
from pathlib import Path

# Input CSV file
input_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"

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