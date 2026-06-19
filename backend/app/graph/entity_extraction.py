import json
import logging
from typing import Any, Dict, List

from app.embeddings.openai_client import openai_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Extract banking entities and relationships as JSON with keys 'entities' (list of strings) "
    "and 'relationships' (list of {source, relation, target})."
)


async def extract_entities(text: str) -> Dict[str, Any]:
    """
    GPT-based extractor with graceful fallback. Returns {entities: [], relationships: []}.
    """
    try:
        user_prompt = (
            f"Text:\n{text}\n\nReturn ONLY JSON matching the schema: "
            "{\"entities\": [\"sample\"], \"relationships\": [{\"source\": \"A\", "
            "\"relation\": \"requires\", \"target\": \"B\"}]}"
        )
        raw = await openai_client.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=SYSTEM_PROMPT,
        )
        parsed = json.loads(raw)
        entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
        relationships = parsed.get("relationships", []) if isinstance(parsed, dict) else []
        if not isinstance(entities, list) or not isinstance(relationships, list):
            raise ValueError("Invalid schema from extractor")
        return {"entities": entities, "relationships": relationships}
    except Exception as exc:
        logger.warning("Entity extraction fallback used: %s", exc)
        # Fallback: naive keyword capture of capitalised tokens
        tokens = [t for t in text.split() if t and t[0].isupper()]
        uniq: List[str] = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        return {"entities": uniq[:10], "relationships": []}
