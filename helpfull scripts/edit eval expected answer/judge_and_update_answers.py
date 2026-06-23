"""
judge_and_update_answers.py

Reads an eval CSV (Question No, Question, Expected Answer, Generated Answer,
Retrieved Context, ...) and a questions CSV (Question no, Eval Question,
Eval Answer).

For every row in the eval CSV, asks GPT-4o-mini to judge whether the
Expected Answer or the Generated Answer better answers the Question given
the Retrieved Context, and whether the Expected Answer needs to be revised.

If a revision is suggested, the matching row in the questions CSV is updated
-- matched by comparing the actual question TEXT (not just the row number),
so mismatched numbering between the two files doesn't silently corrupt data.
A full audit log CSV is written so every decision can be checked by hand.

Setup:
    pip install pandas python-dotenv openai

    Create a .env file next to this script containing:
        OPENAI_API_KEY=sk-...

Usage:
    python judge_and_update_answers.py \
        --eval-csv eval.csv \
        --questions-csv questions.csv \
        --output-csv questions_updated.csv \
        --log-csv judge_log.csv
"""

import argparse
import difflib
import json
import os
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------
# CONFIG -- adjust these if your column headers differ
# --------------------------------------------------------------------------
EVAL_QUESTION_COL = "Question"
EVAL_EXPECTED_COL = "Expected Answer"
EVAL_GENERATED_COL = "Generated Answer"
EVAL_CONTEXT_COL = "Retrieved Context"
EVAL_NUMBER_COL = "Question No"

Q_NUMBER_COL = "Question no"
Q_QUESTION_COL = "Eval Question"
Q_ANSWER_COL = "Eval Answer"

MODEL = "gpt-4o-mini"
MATCH_THRESHOLD = (
    0.90  # similarity score (0-1) above which two questions are treated as the same
)
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

SYSTEM_PROMPT = (
    "You are a meticulous evaluation judge for question-answering systems. "
    "You compare a gold/expected answer and a model-generated answer against "
    "retrieved context, and judge them strictly on correctness and completeness "
    "relative to that context. Respond ONLY with a valid JSON object and nothing else."
)

USER_PROMPT_TEMPLATE = """Question:
{question}

Expected Answer (gold/reference):
{expected_answer}

Generated Answer (model output):
{generated_answer}

Retrieved Context (source passages):
{retrieved_context}

Task:
1. Decide which answer -- "expected" or "generated" -- better answers the
   Question, using the Retrieved Context as ground truth. Use "tie" if they
   are equally good (or equally bad).
2. Independently assess whether the Expected Answer itself is flawed,
   incomplete, or incorrect given the Retrieved Context, regardless of how
   it compares to the Generated Answer.
3. If the Expected Answer needs correction, write a revised answer that
   correctly and completely answers the Question, grounded only in the
   Retrieved Context.

Respond with ONLY a JSON object, exactly in this shape:
{{
  "better_answer": "expected" | "generated" | "tie",
  "expected_answer_needs_revision": true | false,
  "revised_expected_answer": "<string, or null if no revision is needed>",
  "reasoning": "<1-3 sentence justification>"
}}"""


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def best_match(question: str, candidates: pd.Series):
    """Return (best_index, best_score) of the candidate most similar to `question`."""
    norm_q = normalize(question)
    best_idx, best_score = None, 0.0
    for idx, cand in candidates.items():
        score = difflib.SequenceMatcher(None, norm_q, normalize(cand)).ratio()
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx, best_score


def call_judge(client: OpenAI, question, expected, generated, context) -> dict:
    """Call GPT-4o-mini and return the parsed JSON verdict, with retries."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected,
        generated_answer=generated,
        retrieved_context=context,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [warn] judge call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Judge call failed after {MAX_RETRIES} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, help="Path to the eval CSV")
    parser.add_argument(
        "--questions-csv", required=True, help="Path to the questions CSV"
    )
    parser.add_argument(
        "--output-csv",
        default="questions_updated.csv",
        help="Where to write the updated questions CSV",
    )
    parser.add_argument(
        "--log-csv",
        default="judge_log.csv",
        help="Where to write the full audit log of every decision",
    )
    parser.add_argument(
        "--temp-csv",
        default="temp.csv",
        help="Where to write eval questions that could not be confidently matched in the questions CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help="Similarity threshold (0-1) for matching questions across files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process the first N rows (useful for testing)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not found. Put it in a .env file next to this script.")
    client = OpenAI(api_key=api_key)

    eval_df = pd.read_csv(args.eval_csv)
    questions_df = pd.read_csv(args.questions_csv)

    eval_df.columns = [c.strip() for c in eval_df.columns]
    questions_df.columns = [c.strip() for c in questions_df.columns]

    for col in (
        EVAL_QUESTION_COL,
        EVAL_EXPECTED_COL,
        EVAL_GENERATED_COL,
        EVAL_CONTEXT_COL,
    ):
        if col not in eval_df.columns:
            sys.exit(f"Eval CSV is missing expected column: '{col}'")
    for col in (Q_QUESTION_COL, Q_ANSWER_COL):
        if col not in questions_df.columns:
            sys.exit(f"Questions CSV is missing expected column: '{col}'")

    rows_to_process = eval_df.head(args.limit) if args.limit else eval_df

    log_rows = []
    unmatched_rows = []
    updates_applied = 0
    unmatched = 0

    for i, row in rows_to_process.iterrows():
        question = str(row[EVAL_QUESTION_COL])
        expected = str(row[EVAL_EXPECTED_COL])
        generated = str(row[EVAL_GENERATED_COL])
        context = str(row[EVAL_CONTEXT_COL])
        eval_num = row.get(EVAL_NUMBER_COL, i)

        print(f"[{i + 1}/{len(rows_to_process)}] Judging eval question #{eval_num} ...")

        # Match against questions_csv by TEXT, not by row number.
        match_idx, score = best_match(question, questions_df[Q_QUESTION_COL])

        if match_idx is None or score < args.threshold:
            unmatched += 1
            unmatched_rows.append(
                {
                    "Eval Question No": eval_num,
                    "Eval Question": question,
                    "Expected Answer": expected,
                    "Generated Answer": generated,
                    "Retrieved Context": context,
                    "Best Match Score": round(score, 3),
                    "Closest Question in Questions CSV": (
                        questions_df.loc[match_idx, Q_QUESTION_COL]
                        if match_idx is not None
                        else None
                    ),
                }
            )
            log_rows.append(
                {
                    "Eval Question No": eval_num,
                    "Eval Question": question,
                    "Matched Question No": None,
                    "Match Score": round(score, 3),
                    "Number Mismatch Flag": None,
                    "Better Answer": None,
                    "Needs Revision": None,
                    "Old Expected Answer": expected,
                    "New Expected Answer": None,
                    "Reasoning": "NO MATCH FOUND in questions CSV above threshold -- skipped.",
                }
            )
            print(
                f"  [warn] no confident match in questions CSV (best score {score:.2f}) -- skipped"
            )
            continue

        matched_qnum = (
            questions_df.loc[match_idx, Q_NUMBER_COL]
            if Q_NUMBER_COL in questions_df.columns
            else None
        )
        number_mismatch = matched_qnum is not None and str(matched_qnum) != str(
            eval_num
        )

        try:
            verdict = call_judge(client, question, expected, generated, context)
        except RuntimeError as e:
            print(f"  [error] {e}")
            log_rows.append(
                {
                    "Eval Question No": eval_num,
                    "Eval Question": question,
                    "Matched Question No": matched_qnum,
                    "Match Score": round(score, 3),
                    "Number Mismatch Flag": number_mismatch,
                    "Better Answer": "ERROR",
                    "Needs Revision": "ERROR",
                    "Old Expected Answer": expected,
                    "New Expected Answer": None,
                    "Reasoning": str(e),
                }
            )
            continue

        needs_revision = bool(verdict.get("expected_answer_needs_revision"))
        revised = verdict.get("revised_expected_answer")

        if needs_revision and revised:
            questions_df.loc[match_idx, Q_ANSWER_COL] = revised
            updates_applied += 1

        log_rows.append(
            {
                "Eval Question No": eval_num,
                "Eval Question": question,
                "Matched Question No": matched_qnum,
                "Match Score": round(score, 3),
                "Number Mismatch Flag": number_mismatch,
                "Better Answer": verdict.get("better_answer"),
                "Needs Revision": needs_revision,
                "Old Expected Answer": expected,
                "New Expected Answer": revised if needs_revision else None,
                "Reasoning": verdict.get("reasoning"),
            }
        )

    questions_df.to_csv(args.output_csv, index=False)
    pd.DataFrame(log_rows).to_csv(args.log_csv, index=False)
    pd.DataFrame(unmatched_rows).to_csv(args.temp_csv, index=False)

    print("\nDone.")
    print(f"  Rows processed:     {len(rows_to_process)}")
    print(f"  Answers revised:    {updates_applied}")
    print(f"  Unmatched questions:{unmatched}")
    print(f"  Updated CSV:        {args.output_csv}")
    print(
        f"  Audit log:          {args.log_csv}  <-- review this to verify every match/decision"
    )
    if unmatched:
        print(
            f"  Unmatched rows:     {args.temp_csv}  <-- questions with no confident match in the questions CSV"
        )


if __name__ == "__main__":
    main()
