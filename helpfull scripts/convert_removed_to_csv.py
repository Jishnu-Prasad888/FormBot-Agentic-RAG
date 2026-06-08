import re
import csv

input_file = "eval_clean.txt"
output_file = "eval_clean.csv"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split by separator lines
blocks = re.split(r"=+\s*", text.strip())

rows = []

for block in blocks:
    block = block.strip()
    if not block:
        continue

    q_no_match = re.search(r"Question no:\s*(\d+)", block)
    question_match = re.search(r"Eval Question:\s*(.*)", block)
    answer_match = re.search(r"Eval Answer:\s*(.*)", block, re.DOTALL)

    if q_no_match and question_match and answer_match:
        q_no = q_no_match.group(1).strip()
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()

        rows.append([q_no, question, answer])

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question no", "Eval Question", "Eval Answer"])
    writer.writerows(rows)

print("CSV file created:", output_file)