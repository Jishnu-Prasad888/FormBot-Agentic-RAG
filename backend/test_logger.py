import sys
sys.path.insert(0, '/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/backend')

from app.core.evaluation_logger import EvaluationLogger

logger = EvaluationLogger("test_manual")
print(f"Log file: {logger.log_file}")
print(f"File exists: {logger.log_file.exists()}")

logger.log("TEST_STEP", "This is a test message")
print("Logged test message")

with open(logger.log_file, 'r') as f:
    content = f.read()
    print(f"File content length: {len(content)}")
    print(f"Content:\n{content}")
