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

# System prompt for generating unambiguous questions
system_prompt = """You are an expert at creating clear, unambiguous questions about Python built-in functions.

A question is UNAMBIGUOUS if:
- It has a single, clear interpretation
- The answer is verifiable and objective
- There is no confusion about what is being asked

Examples of UNAMBIGUOUS questions:
- "What does the len() function return?"
- "How many parameters does the zip() function take?"
- "What is the return type of the type() function?"

Generate exactly 25 unambiguous questions about Python built-in functions.
Return them as a JSON array with this format:
[
  {
    "question_id": "U01",
    "question": "...",
    "ambiguity_label": 0
  },
  ...
]
"""

print("Generating 25 unambiguous questions...")

response = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=2000,
    messages=[
        {
            "role": "user",
            "content": system_prompt + "\n\nGenerate 25 unambiguous questions about Python built-in functions. Return as JSON array only."
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
    print(f"Generated {len(questions)} questions")
    
    # Save to file
    with open('data/unambiguous_questions.json', 'w') as f:
        json.dump(questions, f, indent=2)
    
    print("Unambiguous questions saved to data/unambiguous_questions.json")
    
    # Print first 3 questions
    print("\nFirst 3 questions:")
    for q in questions[:3]:
        print(f"  {q['question_id']}: {q['question']}")

except json.JSONDecodeError as e:
    print(f"Error parsing response: {e}")
    print(f"Response: {response_text[:500]}")
