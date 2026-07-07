import csv
import hashlib
import json
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------

TXT_FILE = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/formatted_questions.txt"
OUTPUT_CSV = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/questions.csv"

# Replace with your actual list of forms from the system prompt
FORMS = [
"Account Opening Form-Resident Individual",
"Customer Request Form",
"Deposit Slip",
"Withdrawal Slip",
"Agriculture Loan Application Form",
"PM Svanidhi Loan Application Form",
"PMMY Loan Application Form",
"Public Provident Fund Account Closure Form",
"Public Provident Fund Account Deposit Slip",
"Public Provident Fund Account Extension Form",
"Public Provident Fund Account Nomination Form",
"Public Provident Fund Account Opening Form",
"SCSS Account Closure Form",
"SCSS Account Deposit Slip",
"SCSS Account Nomination Change Form",
"SCSS Account Opening Form",
"SSA Account Closure Form",
"SSA Account Opening Form",
"SSA Account Premature Closure",
"SSA Account Withdrawal",
"Auto Loan Application",
"Business Loan Application",
"Education Loan Application",
"Gold Loan Application",
"Home Loan Application",
"Key fact Statement",
"Loan Against Property Application",
"Personal Loan Application"
]

# =====================================================
# OPENAI SETUP
# =====================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =====================================================
# HELPERS
# =====================================================

def get_form_code(form_name: str) -> str:
    """
    Creates a fixed 3-character abbreviation.

    Examples:
        Personal Information Form -> PIF
        Technical Support Form -> TSF
        Survey Form -> SUR
    """

    words = [
        w for w in form_name.split()
        if w.lower() not in {"form", "and", "&"}
    ]

    initials = "".join(word[0].upper() for word in words)

    if len(initials) >= 3:
        return initials[:3]

    cleaned = "".join(
        c.upper()
        for c in form_name
        if c.isalpha()
    )

    return (cleaned + "XXX")[:3]


def create_question_id(form_name: str, question: str) -> str:
    """
    Creates a fixed-length ID:

        PIFA8F3C2D1

    Structure:
        3 chars form code
        8 chars hash

    Total length = 11 chars
    """

    form_code = get_form_code(form_name)

    hash_input = f"{form_name}|{question}"

    hash_part = hashlib.md5(
        hash_input.encode("utf-8")
    ).hexdigest()[:8].upper()

    return f"{form_code}{hash_part}"


def classify_question(question: str) -> str:
    """
    Uses GPT-4o-mini to determine the closest form.
    """

    prompt = f"""
Available forms:

{json.dumps(FORMS, indent=2)}

Question:
{question}

Return ONLY valid JSON:

{{
  "form": "exact form name from the list"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify questions into the closest matching form. "
                    "You must choose exactly one form from the provided list."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
        form_name = result["form"]

        if form_name in FORMS:
            return form_name

    except Exception:
        pass

    return "Unknown Form"


# =====================================================
# MAIN
# =====================================================

def main():

    if not os.path.exists(TXT_FILE):
        raise FileNotFoundError(
            f"Could not find {TXT_FILE}"
        )

    with open(TXT_FILE, "r", encoding="utf-8") as f:
        questions = [
            line.strip()
            for line in f
            if line.strip()
        ]

    rows = []

    total = len(questions)

    for idx, question in enumerate(questions, start=1):

        print(f"[{idx}/{total}] Processing...")

        form_name = classify_question(question)

        question_id = create_question_id(
            form_name,
            question
        )

        rows.append({
            "question_id": question_id,
            "question": question
        })

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "question_id",
                "question"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Questions processed: {len(rows)}")


if __name__ == "__main__":
    main()