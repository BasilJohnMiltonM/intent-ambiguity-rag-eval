import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load questions
with open('data/unambiguous_questions.json', 'r', encoding='utf-8') as f:
    unambiguous = json.load(f)

with open('data/ambiguous_questions.json', 'r', encoding='utf-8') as f:
    ambiguous = json.load(f)

# Read corpus for context
with open('data/corpus/builtins.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()

print(f"Processing {len(unambiguous)} unambiguous questions...")

# Add ground truth to unambiguous questions
for q in unambiguous:
    prompt = f"""Based on the Python documentation, answer this question concisely in 1-2 sentences:
    
Question: {q['question']}

Provide only the answer, no explanations."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    q['ground_truth'] = response.choices[0].message.content.strip()
    print(f"  {q['question_id']}: ?")

print(f"\nProcessing {len(ambiguous)} ambiguous questions...")

# Add ground truth to ambiguous questions
for q in ambiguous:
    answers = []
    for i, interp in enumerate(q.get('interpretations', []), 1):
        prompt = f"""Based on the Python documentation, answer this question considering this specific interpretation:

Question: {q['question']}
Interpretation: {interp}

Provide only the answer, no explanations."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answers.append(response.choices[0].message.content.strip())
    
    q['ground_truth_answers'] = answers
    print(f"  {q['question_id']}: ? ({len(answers)} interpretations)")

# Combine dataset
combined_dataset = unambiguous + ambiguous

# Save
with open('data/questions.json', 'w', encoding='utf-8') as f:
    json.dump(combined_dataset, f, indent=2)

print(f"\n? Dataset finalized with {len(combined_dataset)} questions")
print(f"   - Unambiguous: {len(unambiguous)}")
print(f"   - Ambiguous: {len(ambiguous)}")
print(f"Saved to: data/questions.json")
