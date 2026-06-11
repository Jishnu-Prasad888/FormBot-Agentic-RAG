import pandas as pd
from pathlib import Path

# Input CSV files
eval_file = r"C:\\Users\\Jishnu\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"
remove_file = r"C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval_number_clean.csv"

# Read CSVs
eval_df = pd.read_csv(eval_file)
remove_df = pd.read_csv(remove_file)

# Questions with Accuracy <= 0.6
low_accuracy_questions = eval_df.loc[
    eval_df["Accuracy"] <= 0.6,
    "Question"
]

# Find matching rows in remove_df
filtered_remove_df = remove_df[
    remove_df["Eval Question"].isin(low_accuracy_questions)
]

# Output file
output_file = (
    f"0.6_accuracy_and_belowrows_common_removed_questions_"
    f"{Path(remove_file).stem}.csv"
)

# Save filtered rows
filtered_remove_df.to_csv(output_file, index=False)

print(f"Filtered {len(filtered_remove_df)} rows.")
print(f"Output saved to: {output_file}")