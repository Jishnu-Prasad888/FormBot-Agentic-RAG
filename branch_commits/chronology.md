# Project Chronology

Generated 2026-07-13T06:38:34.210636Z from files in `branch_commits/`.

| Date | Commit | Summary | Branches |
| --- | --- | --- | --- |
| 2026-05-25 | 247931b | initial commit with ollama for LLM | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-25 | 3b6abcb | eval dataset | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-25 | 4cc1284 | readme | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-25 | 6c73b78 | added open ai and docs and embedded them | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-25 | 7e761ac | rag eval | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-25 | f5fcde2 | rag logs ignore | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-26 | 53e8147 | gitignore | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-26 | af6be2c | data and prompt changes | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-05-29 | 7cab83d | new embeddings | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-06-02 | 424678b | single ffile for prompt and mutliagent for eval | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-06-02 | 86f6850 | added better system prompt in eval | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-06-02 | a5b57d3 | eval chnages in export | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-06-03 | 9082779 | gitingore adding ignore files | app, feat_live_ai, kag, master, nearest_neighbour, rebase |
| 2026-06-03 | e9839a5 | nearest neighbour in eval only after removing agentic actions | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-08 | 4516739 | query expansion before retreival and llm question back | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-08 | 527f8b5 | cross encoder reranking | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-08 | 745f03b | upload changes | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-08 | ae6d31d | prompts and additional data | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-09 | 65d9c2f | other helpfull scripts | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-09 | 8bc89e6 | chatgpt suggested changes accuracy dropped to 27 percent | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-09 | cac51ef | feat: increase retrieval recall and add banking-grounded answer generation | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-10 | 158cc72 | feat: Implement comprehensive RAG improvements and enhanced evaluation metrics | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-10 | 47cf985 | eval metrics fix | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-10 | 4f77445 | new uploaded docs and chunks | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-10 | 96fba0e | added synonym expansion | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-10 | fc6d21e | added synonym expansion | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-11 | 2860298 | simple nearest neighbour retreival got 42 percent | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-11 | 4d9be84 | simple nearest neighbour retreival with synonym adn acronym dictionary score at 27 | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-11 | 987fdd4 | restart with port 9000 | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-11 | b50ebba | remove logging | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-11 | c12bb79 | remove logging | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-11 | ed9088b | simple nearest neighbour retreival with query expansion and max 2 retry 47 percent | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-12 | 4a27f50 | elastic search data update from json files from crawler | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-12 | 8cceedd | crawled websites | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-12 | b78ce33 | crawled data downloaded | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-12 | c6aa3d1 | chunked text data from crawler | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-15 | 0da125b | Update compare_changes_in_metrics.py | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-15 | 1687fa8 | removing unused files | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-15 | 299dbdd | results and other stuff | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-15 | 578bc6d | frontend unused remove | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-15 | 8887101 | # Change What 1 RRF fusion Replaced _merge_results() (simple dedup) with reciprocal_rank_fusion() from app.rag.rrf — standard 1/(k+rank) scoring across all vector + BM25 ranked lists 2 Intent detection Added _detect_section() — maps query keywords (e.g., "interest rate", "eligibility") to section names — and _extract_product_name() — regex patterns for 16 banking products (kisan-credit-card, home-loan, etc.) 3 Metadata filters Detected section and product_name (mapped to filename) are passed as ChromaDB where filters to vector_rag.retrieve() and as in-memory filters via filter_results() for BM25 4 Neighbor chunk expansion _fetch_neighbor_chunks() groups retrieved chunks by document_id, fetches all chunks for that document from ChromaDB, and adds ±1 adjacent chunks 5 Two-pass reranking First rerank at top_k * 2 → neighbor expansion → second rerank to final top_k 6 use_query_expansion wired up When False, only the original query (not synonym variants) is used 7 Return dict Added detected_section and detected_product keys | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-15 | 8b2a39f | better upload to elastisearch | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-19 | 71e3b35 | adding neo4j qdant and postgress KAG | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-22 | 038b067 | gpt 5.1 | app, feat_live_ai, kag, nearest_neighbour, rebase |
| 2026-06-22 | 8d7f007 | cleanup | app, feat_live_ai, kag, rebase |
| 2026-06-23 | ecf2428 | metrics for 23rd kag run with llm changed answers oeverlal score at 56.1 worse performing | app, feat_live_ai, kag, rebase |
| 2026-06-23 | fa92da6 | new ingested data and Kag and updated answers | app, feat_live_ai, kag, rebase |
| 2026-06-29 | 184ae59 | requirement | app, feat_live_ai, kag, rebase |
| 2026-06-29 | 633da70 | elastic search remove | app, feat_live_ai, kag, rebase |
| 2026-07-01 | 9052bf2 | project started from scratch to a simple cosine similarity | app, feat_live_ai, rebase |
| 2026-07-02 | 8076885 | 68 percent accuracy and ingested data | app, feat_live_ai, rebase |
| 2026-07-03 | 138b357 | qwen as retrever and gpt as generation llm got 67.3 percent accuracy the llm as a judge was gpt only | app, feat_live_ai, rebase |
| 2026-07-03 | 53bac6d | qwen model as eval acuracy provider openai for all other metrics only qwen model for accurayc eval judging | app, feat_live_ai, rebase |
| 2026-07-03 | d98f9e8 | script to upload docs in a folder for ingestion | app, feat_live_ai, rebase |
| 2026-07-03 | f024c99 | ollama instead of openai 68.3 overall score 100% accuracy in retreival related metrics ( need to investigate ) | app, feat_live_ai, rebase |
| 2026-07-07 | 4f4f1af | added image ocr support with openai and google genai | app, feat_live_ai, rebase |
| 2026-07-08 | 962a0d5 | live user interaction with AI RAG | app, feat_live_ai |
| 2026-07-10 | 1e65936 | removing frontend | app |
| 2026-07-10 | 79c5e2d | live call and speak with AI agent | app, feat_live_ai |
| 2026-07-10 | d7a516a | removing all unusable files for app | app |
