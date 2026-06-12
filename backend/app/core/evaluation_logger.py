import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/backend/logs")
LOGS_DIR.mkdir(exist_ok=True)

class EvaluationLogger:
    def __init__(self, eval_id: str = None):
        if not eval_id:
            eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_id = eval_id
        self.log_file = LOGS_DIR / f"evaluation_{eval_id}.txt"
        self.step_counter = 0
        # Create file immediately with header
        with open(self.log_file, "w") as f:
            f.write(f"EVALUATION LOG: {eval_id}\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("="*80 + "\n")
            f.flush()
            os.fsync(f.fileno())
    
    def log(self, step: str, data: str):
        """Log a step with data"""
        self.step_counter += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = f"\n[{timestamp}] STEP {self.step_counter}: {step}\n{data}\n{'-'*80}\n"
        try:
            with open(self.log_file, "a") as f:
                f.write(entry)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_question(self, question: str, question_num: int, total: int):
        self.log(f"QUESTION RECEIVED ({question_num}/{total})", f"Question: {question}")
    
    def log_retrieval(self, strategy: str, top_k: int, results: list):
        self.log(f"RETRIEVAL ({strategy}, top_k={top_k})", 
                f"Retrieved {len(results)} chunks:\n" + 
                "\n".join([f"  - {r.get('filename', '?')}: {r['chunk_text'][:100]}..." for r in results[:5]]))
    
    def log_es_enhancement(self, enhanced_count: int, original_count: int):
        self.log(f"ELASTICSEARCH ENHANCEMENT", 
                f"Original chunks: {original_count}\nEnhanced chunks: {enhanced_count}\nAdded: {enhanced_count - original_count}")
    
    def log_es_search(self, query: str, results: list, attempt: int):
        self.log(f"ELASTICSEARCH SEARCH (Attempt {attempt})",
                f"Query: {query}\nFound {len(results)} results:\n" +
                "\n".join([f"  - {r['content'][:100]}... (score: {r['score']})" for r in results[:3]]))
    
    def log_llm_call(self, step: str, prompt: str, answer: str, latency_ms: float):
        self.log(f"LLM CALL - {step} (latency: {latency_ms}ms)",
                f"Prompt:\n{prompt[:200]}...\n\nAnswer:\n{answer[:200]}...")
    
    def log_metrics(self, metric_name: str, score: float, rationale: str):
        self.log(f"METRIC - {metric_name}", f"Score: {score}\nRationale: {rationale}")
    
    def log_error(self, error: str, question: str = ""):
        self.log(f"ERROR", f"Question: {question}\nError: {error}")
    
    def log_summary(self, summary: dict):
        summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
        self.log(f"EVALUATION SUMMARY", summary_text)

eval_logger = EvaluationLogger()
