import csv
import json
import os
from pathlib import Path

import pypdf
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=API_KEY)

FORMS_DIR = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\forms"
INPUT_CSV = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval.csv"
OUTPUT_FILE = "eval_clean.txt"

MODEL = "gpt-5"

client = OpenAI(api_key=API_KEY)


def load_all_forms_text(forms_dir: str) -> str:
    """Load and concatenate all PDF text."""
    all_text = []

    pdf_files = sorted(Path(forms_dir).glob("*.pdf"))

    for pdf_file in pdf_files:
        # Keep filename so model knows which form it came from
        all_text.append(pdf_file.name.replace(".pdf",""))

    print(all_text)
    return "\n".join(all_text)


def is_question_relevant(question: str, answer: str, forms_text: str) -> bool:
    """
    Ask the LLM whether this QA pair is relevant
    to someone filling any of the provided SBI forms.
    """

    prompt = f"""
You are evaluating whether a question-answer pair is useful for a user
who is filling SBI banking forms.

Forms content is provided below.

A question should be marked YES if:
- It could reasonably arise while filling any of the forms.
- The answer provides information needed to understand a field,
  term, requirement, process, declaration, or document mentioned
  in the forms.

A question should be marked NO if:
- It is unrelated to the forms.
- It concerns topics not needed to understand or fill the forms.
- The answer provides information that would not help someone
  complete the forms.

Return ONLY valid JSON:

{{"keep": true}}

or

{{"keep": false}}

QUESTION:
{question}

ANSWER:
{answer}

FORMS:
{forms_text}
"""

    response = client.chat.completions.create(
    model=MODEL,
    messages=[
            {"role": "user", "content": prompt}
    ]
    )


    text = response.choices[0].message.content.strip()

    try:
        result = json.loads(text)
        return bool(result.get("keep", False))
    except Exception:
        print("Failed to parse response:")
        print(text)
        return False


def main():
    print("Loading SBI forms...")
    forms_text = load_all_forms_text(FORMS_DIR)
    print(forms_text)
    kept_rows = []

    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            qno = row.get("Question no", "")
            question = row.get("Eval Question", "")
            answer = row.get("Eval Answer", "")

            print(f"Evaluating question {qno}...")

            keep = is_question_relevant(
                question=question,
                answer=answer,
                forms_text=forms_text
            )

            print(f" -> KEEP={keep}")

            if keep:
                kept_rows.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(f"Question no: {row['Question no']}\n")
            f.write(f"Eval Question: {row['Eval Question']}\n")
            f.write(f"Eval Answer: {row['Eval Answer']}\n")
            f.write("\n" + ("=" * 80) + "\n\n")

    print(f"\nDone.")
    print(f"Kept {len(kept_rows)} questions.")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()