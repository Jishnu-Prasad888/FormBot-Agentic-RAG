"""
Convert form-structured CSV to flat eval Q&A format.

Input CSV structure (no fixed header, repeating blocks):
    form name, question 1, question 2, ...
    (blank),   answer 1,   answer 2,   ...

Output CSV structure:
    Question no, Eval Question + form name, Eval Answer

Usage:
    python convert_form_csv.py input.csv output.csv

    # Or with custom delimiter (e.g. tab-separated):
    python convert_form_csv.py input.csv output.csv --delimiter '\t'
"""

import csv
import argparse
import sys


def convert(input_path: str, output_path: str, delimiter: str = ",") -> int:
    """
    Parse the input CSV and write the flattened eval CSV.
    Returns the number of Q&A rows written.
    """
    rows = []
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)

    qa_pairs = []
    i = 0

    while i < len(rows):
        row = rows[i]

        # Skip completely empty rows
        if not any(cell.strip() for cell in row):
            i += 1
            continue

        first_cell = row[0].strip() if row else ""

        # A "form name" row: first cell is non-empty (the form name)
        # and there are questions in the remaining cells.
        if first_cell:
            form_name = first_cell
            questions = [cell.strip() for cell in row[1:]]

            # Look ahead for the answer row (first cell blank, rest are answers)
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                next_first = next_row[0].strip() if next_row else ""
                if not next_first:
                    answers = [cell.strip() for cell in next_row[1:]]
                    i += 2  # consumed both rows
                else:
                    # No answer row follows — treat answers as empty
                    answers = []
                    i += 1
            else:
                answers = []
                i += 1

            # Pair each question with its answer (zip stops at shortest)
            for q, a in zip(questions, answers):
                q = q.strip()
                a = a.strip()
                if q:  # skip blank question slots
                    eval_question = f"{q} ({form_name})" if form_name else q
                    qa_pairs.append((eval_question, a))

        else:
            # Answer row without a preceding form row — skip
            i += 1

    # Write output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question no", "Eval Question + form name", "Eval Answer"])
        for idx, (question, answer) in enumerate(qa_pairs, start=1):
            writer.writerow([idx, question, answer])

    return len(qa_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Convert form-structured CSV to flat eval Q&A CSV."
    )
    parser.add_argument("input", help="Path to the input CSV file")
    parser.add_argument("output", help="Path for the output CSV file")
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter character (default: comma). Use '\\t' for tab.",
    )
    args = parser.parse_args()

    delimiter = args.delimiter.replace("\\t", "\t")

    try:
        count = convert(args.input, args.output, delimiter)
        print(f"Done. Wrote {count} Q&A rows to '{args.output}'.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()