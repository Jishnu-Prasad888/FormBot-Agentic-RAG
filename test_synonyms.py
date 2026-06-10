#!/usr/bin/env python3
"""Test synonym expansion functionality"""
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
            print(f"✓ Loaded {len(self.synonyms)} canonical terms from JSON")
        except Exception as e:
            print(f"✗ Error loading JSON synonyms: {e}")

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
            print(f"✓ Loaded synonyms from CSV, total canonical terms: {len(self.synonyms)}")
        except Exception as e:
            print(f"✗ Error loading CSV synonyms: {e}")

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


if __name__ == "__main__":
    backend_dir = Path("C:\\Users\\Jishnu\\Desktop\\SRAG\\backend")
    json_path = backend_dir / "rag_synonym_dictionary.json"
    csv_path = backend_dir / "rag_synonym_dictionary_pairs.csv"
    
    print("Testing Synonym Expansion System")
    print("=" * 50)
    
    expander = SynonymExpander(str(json_path), str(csv_path))
    
    print(f"\nTotal reverse mappings: {len(expander.reverse_map)}")
    print("\nTest Queries:")
    print("-" * 50)
    
    test_queries = [
        "service charges on advances",
        "bank code",
        "policy on general management",
        "advances related service charges",
    ]
    
    for q in test_queries:
        expanded = expander.expand_query(q)
        print(f"\nOriginal: {q}")
        if len(expanded) > 1:
            print(f"Expanded ({len(expanded)} variants):")
            for i, exp in enumerate(expanded[1:], 1):
                print(f"  {i}. {exp}")
        else:
            print("  (no expansions found)")
