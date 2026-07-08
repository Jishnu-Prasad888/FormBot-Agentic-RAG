import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

input_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/kag eval sheet from arya.csv"
df = pd.read_csv(input_file)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Correct the following question to make it grammatically correct and clear and also suitable for a person to speak to create a dataset and give only the question back nothing but the corrected question."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()


def write_to_file(formatted_questions, output_file):
    with open(output_file, 'w') as f:
        for question in formatted_questions:
            f.write(question + '\n')



formatted_questions = []

for index, row in df.iterrows():
    form_name = row[df.columns[0]].strip().lower()  # Assuming the form name is in the first column
    question = row[df.columns[1]].strip()  # Assuming the question is in the second column
    # Format the question
    formatted_question = f"For {form_name} {question}"
    formatting_response = ask_llm(formatted_question)
    formatted_questions.append(formatting_response)
    print(f"Formatted question for {form_name}: {formatting_response}")

write_to_file(formatted_questions, "formatted_questions.txt")