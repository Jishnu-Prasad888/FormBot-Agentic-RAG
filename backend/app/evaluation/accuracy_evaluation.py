from typing import Any
import re
from difflib import SequenceMatcher
from app.embeddings.openai_client import openai_client
from app.core.logging import get_logger

logger = get_logger("accuracy_evaluation")


class AccuracyEvaluator:
    """Multi-method accuracy evaluation: exact match, semantic similarity, F1."""

    @staticmethod
    def exact_match(generated: str, expected: str) -> float:
        """Exact string match (case-insensitive, normalized)."""
        gen_norm = generated.lower().strip()
        exp_norm = expected.lower().strip()
        return 1.0 if gen_norm == exp_norm else 0.0

    @staticmethod
    def semantic_similarity(generated: str, expected: str) -> float:
        """Token-level semantic similarity using SequenceMatcher."""
        matcher = SequenceMatcher(None, generated.lower(), expected.lower())
        ratio = matcher.ratio()
        return round(ratio, 4)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def f1_score(generated: str, expected: str) -> float:
        """F1 score based on token overlap."""
        gen_tokens = set(AccuracyEvaluator._tokenize(generated))
        exp_tokens = set(AccuracyEvaluator._tokenize(expected))

        if not exp_tokens:
            return 1.0 if not gen_tokens else 0.0

        tp = len(gen_tokens & exp_tokens)
        fp = len(gen_tokens - exp_tokens)
        fn = len(exp_tokens - gen_tokens)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 4)

    @staticmethod
    def combined_accuracy(
        generated: str,
        expected: str,
        weights: dict[str, float] = None,
    ) -> float:
        """Weighted combination of all accuracy methods."""
        if weights is None:
            weights = {
                "exact_match": 0.2,
                "semantic_similarity": 0.3,
                "f1": 0.5,
            }

        exact = AccuracyEvaluator.exact_match(generated, expected)
        semantic = AccuracyEvaluator.semantic_similarity(generated, expected)
        f1 = AccuracyEvaluator.f1_score(generated, expected)

        combined = (
            weights.get("exact_match", 0) * exact +
            weights.get("semantic_similarity", 0) * semantic +
            weights.get("f1", 0) * f1
        )

        return round(combined, 4)


accuracy_evaluator = AccuracyEvaluator()
