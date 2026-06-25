import json
import os

def load_corpus_chunks(chunks_path='data/corpus/builtins_chunks.json'):
    """
    Load pre-chunked corpus from JSON.
    
    Args:
        chunks_path: Path to the chunks JSON file
    
    Returns:
        List of chunk dictionaries with 'chunk_id' and 'text'
    """
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return chunks

def validate_chunks(chunks):
    """
    Validate that chunks have required fields.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        Boolean indicating if chunks are valid
    """
    if not chunks:
        print("No chunks found!")
        return False
    
    for chunk in chunks:
        if 'chunk_id' not in chunk or 'text' not in chunk:
            print(f"Invalid chunk: {chunk}")
            return False
    
    return True

def print_corpus_stats(chunks):
    """Print statistics about the corpus."""
    total_chunks = len(chunks)
    total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
    
    print(f"\n{'='*60}")
    print(f"Corpus Statistics")
    print(f"{'='*60}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total approximate tokens: {total_tokens}")
    print(f"Average tokens per chunk: {total_tokens // total_chunks if total_chunks > 0 else 0}")
    print(f"\nFirst chunk preview:")
    print(f"  ID: {chunks[0]['chunk_id']}")
    print(f"  Text: {chunks[0]['text'][:100]}...")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Load chunks
    print("Loading corpus chunks...")
    chunks = load_corpus_chunks()
    
    # Validate
    if not validate_chunks(chunks):
        print("Chunk validation failed!")
        exit(1)
    
    # Print stats
    print_corpus_stats(chunks)
    
    # Save metadata for pipeline
    metadata = {
        'total_chunks': len(chunks),
        'domain': 'Python builtins',
        'chunk_size': 400,
        'chunk_overlap': 50
    }
    
    with open('data/corpus_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("? Ingestion complete. Metadata saved to data/corpus_metadata.json")
