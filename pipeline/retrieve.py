import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    """
    Retriever for RAG pipeline.
    Loads FAISS index and returns top-k chunks for a query.
    """
    
    def __init__(self, 
                 index_path='data/corpus/faiss_index.bin',
                 metadata_path='data/corpus/index_metadata.json',
                 model_name='all-MiniLM-L6-v2'):
        """
        Initialize retriever.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to index metadata
            model_name: Sentence-transformers model
        """
        # Load model
        self.model = SentenceTransformer(model_name)
        
        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.chunk_id_to_text = self.metadata['chunk_id_to_text']
    
    def retrieve(self, query, k=3):
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve (default: 3)
        
        Returns:
            List of tuples: (chunk_id, chunk_text, similarity_score)
        """
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Format results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = 1 / (1 + dist)  # Normalized similarity (0-1)
            
            chunk_text = self.chunk_id_to_text[str(idx)]
            results.append({
                'rank': i + 1,
                'chunk_id': int(idx),
                'text': chunk_text,
                'distance': float(dist),
                'similarity': float(similarity)
            })
        
        return results

def format_context(retrieved_chunks):
    """Format retrieved chunks into context string for LLM."""
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}")
    
    return "\n\n".join(context_parts)

if __name__ == "__main__":
    # Test retrieval
    print("="*60)
    print("Testing Retrieval Function")
    print("="*60 + "\n")
    
    try:
        retriever = RAGRetriever()
        print("? Retriever initialized\n")
        
        # Test query
        test_query = "What does the len() function return?"
        print(f"Query: {test_query}\n")
        
        results = retriever.retrieve(test_query, k=3)
        
        print(f"Retrieved {len(results)} chunks:\n")
        for result in results:
            print(f"Rank {result['rank']} (Chunk {result['chunk_id']})")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Text: {result['text'][:150]}...")
            print()
        
        print("="*60)
        print("? Retrieval test successful!")
        print("="*60)
        
    except Exception as e:
        print(f"? Error: {e}")
        import traceback
        traceback.print_exc()
