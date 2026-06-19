import csv
import openai
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def check_answer_with_gpt4(question, answer):
    """
    Send question and answer to GPT-4 for evaluation
    Returns: (is_correct, corrected_answer)
    """
    prompt = f"""
You are an evaluator checking if the provided answer correctly answers the question.

Question: {question}

Provided Answer: {answer}

Please evaluate if the answer is correct and complete. If the answer is correct, respond with "CORRECT" only.
If the answer is incorrect or incomplete, provide the corrected answer.

Your response format must be:
- If correct: CORRECT
- If incorrect: CORRECTED: [your corrected answer here]

Do not add any extra text or explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "You are a strict evaluator of answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.startswith("CORRECTED:"):
            corrected_answer = result.replace("CORRECTED:", "").strip()
            return False, corrected_answer
        elif result == "CORRECT":
            return True, answer
        else:
            # Fallback - if response is unexpected, assume incorrect and use the response as correction
            return False, result
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Return False with the original answer to avoid losing data
        return False, answer

def process_csv(input_file, output_file="mistakes.csv", delay=1):
    """
    Process CSV file and create mistakes.csv with corrected answers
    """
    mistakes = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            # Check if columns exist
            expected_columns = ['Question no', 'Eval Question', 'Eval Answer']
            if not all(col in reader.fieldnames for col in expected_columns):
                print(f"Error: CSV must have columns: {expected_columns}")
                return
            
            total_rows = 0
            correct_count = 0
            incorrect_count = 0
            
            for row in reader:
                total_rows += 1
                question_no = row['Question no']
                question = row['Eval Question']
                answer = row['Eval Answer']
                
                print(f"Processing Question {question_no}...")
                
                # Check if answer is correct
                is_correct, corrected_answer = check_answer_with_gpt4(question, answer)
                
                if is_correct:
                    correct_count += 1
                    print(f"  ✓ Question {question_no}: Correct")
                else:
                    incorrect_count += 1
                    print(f"  ✗ Question {question_no}: Incorrect - Adding to mistakes file")
                    mistakes.append({
                        'Question no': question_no,
                        'Eval Question': question,
                        'Eval Answer': corrected_answer
                    })
                
                # Add delay to respect rate limits
                if delay > 0:
                    time.sleep(delay)
        
        # Write mistakes to CSV if any
        if mistakes:
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['Question no', 'Eval Question', 'Eval Answer']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mistakes)
            
            print(f"\n✅ Found {incorrect_count} incorrect answers out of {total_rows} total questions")
            print(f"📝 Mistakes written to: {output_file}")
        else:
            print(f"\n✅ All {total_rows} answers are correct! No mistakes file created.")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Configuration
    INPUT_FILE = "manual_english.csv"  # Change this to your CSV filename
    OUTPUT_FILE = "mistakes.csv"
    API_DELAY = 0.5  # Delay between API calls (in seconds) to avoid rate limits
    
    print("=" * 60)
    print("CSV Answer Evaluator with GPT-4o")
    print("=" * 60)
    
    # Validate API key
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with: OPENAI_API_KEY=your-api-key-here")
        return
    
    print("✅ API key loaded successfully from .env file")
    print(f"📂 Input file: {INPUT_FILE}")
    print(f"📄 Output file: {OUTPUT_FILE}")
    print("=" * 60)
    
    process_csv(INPUT_FILE, OUTPUT_FILE, API_DELAY)

if __name__ == "__main__":
    main()