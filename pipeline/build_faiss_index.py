import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_path='data/corpus/builtins_chunks.json'):
    """Load chunks from JSON."""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

def build_embeddings(chunks, model_name='all-mpnet-base-v2'):
    """
    Embed all chunks using sentence-transformers.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Sentence-transformers model to use
    
    Returns:
        Tuple of (embeddings array, model)
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Encoding {len(chunks)} chunks...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, model

def build_faiss_index(embeddings):
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: NumPy array of shape (n_chunks, embedding_dim)
    
    Returns:
        FAISS index object
    """
    print("Building FAISS index...")
    
    # Create L2 index (Euclidean distance)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Add vectors to index
    index.add(embeddings.astype(np.float32))
    
    print(f"Index created with {index.ntotal} vectors")
    return index

def save_index_and_metadata(index, chunks, output_dir='data/corpus'):
    """Save FAISS index and mapping of chunk IDs."""
    
    # Save index
    index_path = os.path.join(output_dir, 'faiss_index.bin')
    faiss.write_index(index, index_path)
    print(f"Index saved to: {index_path}")
    
    # Save chunk metadata (ID -> text mapping)
    metadata = {
        'chunk_id_to_text': {
            str(chunk['chunk_id']): chunk['text'] 
            for chunk in chunks
        },
        'total_chunks': len(chunks),
        'embedding_model': 'all-MiniLM-L6-v2',
        'index_type': 'L2'
    }
    
    metadata_path = os.path.join(output_dir, 'index_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    print("="*60)
    print("Building FAISS Index")
    print("="*60 + "\n")
    
    # Load chunks
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Build embeddings
    embeddings, model = build_embeddings(chunks)
    
    # Build index
    index = build_faiss_index(embeddings)
    
    # Save
    save_index_and_metadata(index, chunks)
    
    print("\n" + "="*60)
    print("? FAISS index built and saved successfully!")
    print("="*60)
