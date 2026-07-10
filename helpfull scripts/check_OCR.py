import os
import sys
import json
import base64
import argparse
from pathlib import Path
from collections import defaultdict
import fitz
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, APITimeoutError, APIError
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import precision_recall_fscore_support


def log(msg):
    print(msg, flush=True)


# ============================================================
# OpenAI Client
# ============================================================

OPENAI_REQUEST_TIMEOUT = float(
    os.getenv("OPENAI_TIMEOUT", "180")
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    timeout=OPENAI_REQUEST_TIMEOUT,
    max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2"))
)

OCR_PROMPT = """
Perform OCR on this document.

Rules:
- Preserve reading order.
- Preserve Kannada exactly.
- Do not translate.
- Return only extracted text.
"""

VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

USE_RESPONSES_API = hasattr(client, "responses")


# ============================================================
# OCR
# ============================================================

def encode_image_bytes(data):
    return base64.b64encode(data).decode("utf-8")


def extract_text_from_chat_response(response):
    message_content = response.choices[0].message.content

    if isinstance(message_content, str):
        return message_content.strip()

    texts = []

    for part in message_content:
        if isinstance(part, str):
            texts.append(part)
            continue

        text_value = getattr(part, "text", None)

        if text_value is None and isinstance(part, dict):
            text_value = part.get("text")

        if text_value:
            texts.append(text_value)

    return "\n".join(texts).strip()


def ocr_with_chat(image_b64, *, context="image"):
    log(f"[chat] Sending {context} to {VISION_MODEL} (timeout={OPENAI_REQUEST_TIMEOUT}s)...")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        timeout=OPENAI_REQUEST_TIMEOUT
    )

    log(f"[chat] Response received for {context}.")

    return extract_text_from_chat_response(response)


def ocr_with_responses(content, *, context="document"):
    log(f"[responses] Sending {context} to gpt-4.1 (timeout={OPENAI_REQUEST_TIMEOUT}s)...")

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        timeout=OPENAI_REQUEST_TIMEOUT
    )

    log(f"[responses] Response received for {context}.")

    return response.output_text.strip()


def ocr_image(image_path):
    with open(image_path, "rb") as f:
        image_b64 = encode_image_bytes(f.read())

    if USE_RESPONSES_API:
        content = [
            {
                "type": "input_text",
                "text": OCR_PROMPT
            },
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_b64}"
            }
        ]

        return ocr_with_responses(content, context=Path(image_path).name)

    return ocr_with_chat(image_b64, context=Path(image_path).name)


def pdf_to_images(pdf_path, scale=2.0):
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    for page_number, page in enumerate(doc, 1):
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale)
        )

        yield page_number, total_pages, pix.tobytes("png")


def ocr_pdf(pdf_path):
    if USE_RESPONSES_API:
        log("[responses] Uploading PDF for OCR...")

        with open(pdf_path, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="user_data"
            )

        content = [
            {
                "type": "input_file",
                "file_id": uploaded.id
            },
            {
                "type": "input_text",
                "text": OCR_PROMPT
            }
        ]

        return ocr_with_responses(content, context=Path(pdf_path).name)

    page_texts = []

    for page_number, total_pages, image_bytes in pdf_to_images(pdf_path):
        image_b64 = encode_image_bytes(image_bytes)
        page_texts.append(
            ocr_with_chat(
                image_b64,
                context=f"page {page_number}/{total_pages}"
            )
        )

    log("[chat] All pages processed.")

    return "\n".join(page_texts)


# ============================================================
# Ground Truth Extraction
# ============================================================

def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ground_truth(annotation_json):
    rows = []

    for page in annotation_json["pages"]:
        anns = page["annotations"]

        anns = sorted(
            anns,
            key=lambda x: (
                round(x.get("y", 0), 3),
                round(x.get("x", 0), 3)
            )
        )

        for ann in anns:
            text = ann.get("transcript", "").strip()

            if text and text != ".":
                rows.append(text)

    return "\n".join(rows)


def build_ground_truth_per_class(annotation_json):
    result = defaultdict(list)

    for page in annotation_json["pages"]:
        for ann in page["annotations"]:

            text = ann.get("transcript", "").strip()

            if not text:
                continue

            if text == ".":
                continue

            cls = ann.get("class_name", "Unknown")

            result[cls].append(text)

    return {
        k: "\n".join(v)
        for k, v in result.items()
    }


# ============================================================
# Metrics
# ============================================================

def cer(reference, prediction):
    if not reference:
        return 0

    return (
        Levenshtein.distance(reference, prediction)
        / len(reference)
    )


def wer(reference, prediction):
    ref_words = reference.split()
    pred_words = prediction.split()

    if not ref_words:
        return 0

    return (
        Levenshtein.distance(ref_words, pred_words)
        / len(ref_words)
    )


def exact_match(reference, prediction):
    return reference.strip() == prediction.strip()


def precision_recall_f1(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    vocab = sorted(
        set(ref_tokens + pred_tokens)
    )

    idx = {
        token: i
        for i, token in enumerate(vocab)
    }

    y_true = [0] * len(vocab)
    y_pred = [0] * len(vocab)

    for token in set(ref_tokens):
        y_true[idx[token]] = 1

    for token in set(pred_tokens):
        y_pred[idx[token]] = 1

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )
    )

    return precision, recall, f1


def evaluate(reference, prediction):
    p, r, f1 = precision_recall_f1(
        reference,
        prediction
    )

    return {
        "CER": round(cer(reference, prediction), 4),
        "WER": round(wer(reference, prediction), 4),
        "ExactMatch": exact_match(
            reference,
            prediction
        ),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f1, 4),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--document",
        required=True,
        help="Image or PDF"
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="Annotation JSON"
    )

    parser.add_argument(
        "--save-ocr",
        default="ocr_output.txt"
    )

    parser.add_argument(
        "--api-mode",
        choices=["auto", "responses", "chat"],
        default="auto",
        help=(
            "Use 'responses' (single call) or 'chat' (per-page). "
            "Default: auto-detect."
        )
    )

    args = parser.parse_args()

    document_path = Path(args.document)
    annotations_path = Path(args.annotations)

    if not document_path.exists():
        log(f"Document not found: {document_path}")
        sys.exit(1)

    if not annotations_path.exists():
        log(f"Annotation file not found: {annotations_path}")
        sys.exit(1)

    global USE_RESPONSES_API

    if args.api_mode == "chat":
        USE_RESPONSES_API = False
    elif args.api_mode == "responses":
        USE_RESPONSES_API = hasattr(client, "responses")
        if not USE_RESPONSES_API:
            log("Responses API not available in this OpenAI client; falling back to chat.")

    log(f"Using OpenAI vision model: {VISION_MODEL}")
    log(f"Request timeout: {OPENAI_REQUEST_TIMEOUT}s")
    log(f"API mode: {'responses' if USE_RESPONSES_API else 'chat'}")
    log(f"Document: {document_path}")
    log(f"Annotations: {annotations_path}")

    annotation_data = load_annotations(
        annotations_path
    )

    gt_text = build_ground_truth(
        annotation_data
    )

    per_class_gt = build_ground_truth_per_class(
        annotation_data
    )

    ext = document_path.suffix.lower()

    try:
        if ext == ".pdf":
            ocr_text = ocr_pdf(document_path)
        else:
            ocr_text = ocr_image(document_path)
    except (APITimeoutError, APIError) as exc:
        log(f"OCR failed: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        log(f"Unexpected error during OCR: {exc}")
        sys.exit(1)

    with open(
        args.save_ocr,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(ocr_text)

    print("\n")
    print("=" * 70)
    print("LLM OCR RESPONSE")
    print("=" * 70)
    print(ocr_text)
    print("\n")
    print("=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)

    overall = evaluate(
        gt_text,
        ocr_text
    )

    print(
        json.dumps(
            overall,
            indent=2,
            ensure_ascii=False
        )
    )

    print("\n")
    print("=" * 70)
    print("PER CLASS METRICS")
    print("=" * 70)

    for cls, class_gt in per_class_gt.items():

        metrics = evaluate(
            class_gt,
            ocr_text
        )

        print("\n")
        print(f"[{cls}]")

        print(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False
            )
        )

    print("\n")
    print("=" * 70)
    print(f"OCR saved to: {args.save_ocr}")
    print("=" * 70)


if __name__ == "__main__":
    main()
