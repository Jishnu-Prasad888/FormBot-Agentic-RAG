import json
import csv
from pathlib import Path



class SynonymExpander:
    def __init__(self, json_path: str = None, csv_path: str = None):
        self.synonyms = {}  # canonical -> [aliases]
        self.reverse_map = {}  # alias -> canonical
        
        if json_path:
            self._load_json(json_path)
        if csv_path:
            self._load_csv(csv_path)

    def _load_json(self, path: str):
        """Load synonyms from JSON format: {topics: [{canonical, aliases}]}"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for topic in data.get("topics", []):
                canonical = topic.get("canonical", "").lower().strip()
                if canonical:
                    aliases = [alias.lower().strip() for alias in topic.get("aliases", [])]
                    self.synonyms[canonical] = aliases
                    for alias in aliases:
                        self.reverse_map[alias] = canonical
        except Exception:
            pass

    def _load_csv(self, path: str):
        """Load synonyms from CSV format: canonical,alias"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    canonical = row.get("canonical", "").lower().strip()
                    alias = row.get("alias", "").lower().strip()
                    if canonical and alias:
                        if canonical not in self.synonyms:
                            self.synonyms[canonical] = []
                        if alias not in self.synonyms[canonical]:
                            self.synonyms[canonical].append(alias)
                        self.reverse_map[alias] = canonical
        except Exception:
            pass

    def expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms and return list of expanded queries"""
        query_lower = query.lower()
        expanded_queries = [query]
        
        for alias, canonical in self.reverse_map.items():
            if alias in query_lower:
                # Replace alias with canonical
                expanded = query_lower.replace(alias, canonical)
                if expanded != query_lower and expanded not in expanded_queries:
                    expanded_queries.append(expanded)
        
        return expanded_queries


# Initialize global synonym expander
_synonym_expander = None


def init_synonym_expander(json_path: str = None, csv_path: str = None):
    """Initialize the global synonym expander"""
    global _synonym_expander
    _synonym_expander = SynonymExpander(json_path, csv_path)
    return _synonym_expander


def get_synonym_expander() -> SynonymExpander:
    """Get the global synonym expander instance"""
    global _synonym_expander
    if _synonym_expander is None:
        backend_dir = Path(__file__).parent.parent.parent
        json_path = backend_dir / "rag_synonym_dictionary.json"
        csv_path = backend_dir / "rag_synonym_dictionary_pairs.csv"
        
        _synonym_expander = SynonymExpander(
            str(json_path) if json_path.exists() else None,
            str(csv_path) if csv_path.exists() else None,
        )
    return _synonym_expander
