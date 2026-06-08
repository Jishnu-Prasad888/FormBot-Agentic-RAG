import csv

input_file = "eval_clean.csv"
output_file = "eval_number_clean.csv"

rows = []

# Read existing CSV
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)  # skip header if present

    for row in reader:
        if len(row) < 3:
            continue
        # row = [Question no, Eval Question, Eval Answer]
        rows.append([row[1], row[2]])  # drop old number, keep question + answer

# Rewrite with fixed numbering
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question no", "Eval Question", "Eval Answer"])

    for i, (question, answer) in enumerate(rows, start=1):
        writer.writerow([i, question, answer])

print("Fixed CSV written to:", output_file)