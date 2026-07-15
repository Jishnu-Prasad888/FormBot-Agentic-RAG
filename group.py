import csv
import json
import math
from uuid import uuid4

INPUT_CSV = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/questions.csv"
OUTPUT_JSON = "questions_grouped.json"

# Read CSV
rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "question_id": row["question_id"],
            "question": row["question"]
        })

# Split into 3 groups (each group gets ~1/3 of records)
total_rows = len(rows)
group_size = math.ceil(total_rows / 3)

groups = []

for i in range(0, total_rows, group_size):
    chunk = rows[i:i + group_size]

    groups.append({
        "group_id": str(uuid4()),  # unique group id
        "questions": chunk
    })

# Write JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(groups, f, indent=2, ensure_ascii=False)

print(f"Created {len(groups)} groups in {OUTPUT_JSON}")