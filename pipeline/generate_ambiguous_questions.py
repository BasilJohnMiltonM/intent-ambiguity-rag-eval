import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read the chunks to understand the domain
with open('data/corpus/builtins_chunks.json', 'r') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from corpus")

# System prompt for generating ambiguous questions
system_prompt = """You are an expert at creating ambiguous questions about Python built-in functions.

A question is AMBIGUOUS if:
- The same surface-level question can be interpreted in 2-3 different ways
- Each interpretation leads to a different correct answer
- Both interpretations are plausible and valid

Examples of AMBIGUOUS questions:
- "How do you handle memory in Python?" 
  Interpretation 1: Garbage collection mechanics
  Interpretation 2: Memory optimization for large data
  Interpretation 3: Memory leaks and reference counting

- "What is the best way to iterate in Python?"
  Interpretation 1: Using for loops vs while loops
  Interpretation 2: Using iterators vs generators
  Interpretation 3: Performance considerations for different data types

Generate exactly 25 ambiguous questions about Python built-in functions.
For each question, provide 2-3 different plausible interpretations.

Return them as a JSON array with this format:
[
  {
    "question_id": "A01",
    "question": "...",
    "ambiguity_label": 1,
    "interpretations": ["interpretation 1", "interpretation 2", "interpretation 3"]
  },
  ...
]
"""

print("Generating 25 ambiguous questions...")

response = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=3000,
    messages=[
        {
            "role": "user",
            "content": system_prompt + "\n\nGenerate 25 ambiguous questions about Python built-in functions. Return as JSON array only."
        }
    ]
)

# Extract the response
response_text = response.choices[0].message.content

# Parse JSON from response
try:
    # Clean up the response (remove markdown code blocks if present)
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    questions = json.loads(response_text)
    print(f"Generated {len(questions)} ambiguous questions")
    
    # Save to file
    with open('data/ambiguous_questions.json', 'w') as f:
        json.dump(questions, f, indent=2)
    
    print("Ambiguous questions saved to data/ambiguous_questions.json")
    
    # Print first 2 questions
    print("\nFirst 2 questions:")
    for q in questions[:2]:
        print(f"\n  {q['question_id']}: {q['question']}")
        print(f"  Interpretations:")
        for i, interp in enumerate(q.get('interpretations', []), 1):
            print(f"    {i}. {interp}")

except json.JSONDecodeError as e:
    print(f"Error parsing response: {e}")
    print(f"Response: {response_text[:500]}")
