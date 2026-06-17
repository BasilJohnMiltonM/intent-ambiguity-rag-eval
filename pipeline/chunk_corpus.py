import os
import json
from pathlib import Path

# Simple tokenizer (split by words)
def tokenize(text):
    """Split text into tokens (words)."""
    return text.split()

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks.
    chunk_size: number of tokens per chunk
    overlap: number of overlapping tokens between chunks
    """
    tokens = tokenize(text)
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        if chunk_tokens:  # Only add non-empty chunks
            chunk_text = ' '.join(chunk_tokens)
            chunks.append({
                'chunk_id': len(chunks),
                'text': chunk_text,
                'token_count': len(chunk_tokens)
            })
        if i + chunk_size >= len(tokens):
            break
    
    return chunks

# Read the corpus
corpus_path = 'data/corpus/builtins.txt'
with open(corpus_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total text length: {len(text)} characters")
print(f"Approximate tokens: {len(tokenize(text))}")

# Create chunks
chunks = chunk_text(text, chunk_size=400, overlap=50)

# Save chunks to JSON
output_path = 'data/corpus/builtins_chunks.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2)

print(f"Created {len(chunks)} chunks")
print(f"Chunks saved to: {output_path}")
