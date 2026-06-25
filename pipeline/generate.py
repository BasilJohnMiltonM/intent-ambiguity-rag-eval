import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGGenerator:
    """
    Generator for RAG pipeline.
    Takes context and question, generates answer with GPT-4o.
    """
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        """
        Initialize generator.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for generation (0.0 = deterministic)
        """
        self.model = model
        self.temperature = temperature
    
    def generate(self, question, context):
        """
        Generate answer given question and context.
        
        Args:
            question: User question
            context: Retrieved context chunks (formatted string)
        
        Returns:
            Generated answer string
        """
        # Fixed prompt template
        prompt = f"""You are a precise question-answering assistant.
Use only the provided context to answer the question.
If the question is ambiguous, state the interpretation you are using before answering.

Context: {context}

Question: {question}

Answer:"""
        
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()

def format_context(retrieved_chunks):
    """Format retrieved chunks into context string for LLM."""
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}")
    
    return "\n\n".join(context_parts)

if __name__ == "__main__":
    # Test generation
    print("="*60)
    print("Testing Generation Function")
    print("="*60 + "\n")
    
    try:
        from retrieve import RAGRetriever
        
        # Initialize retriever and generator
        retriever = RAGRetriever()
        generator = RAGGenerator(model="gpt-4o-mini")
        print("? Retriever and generator initialized\n")
        
        # Test query
        test_question = "What does the len() function return?"
        print(f"Question: {test_question}\n")
        
        # Retrieve context
        retrieved_chunks = retriever.retrieve(test_question, k=3)
        context = format_context(retrieved_chunks)
        
        print(f"Retrieved {len(retrieved_chunks)} chunks\n")
        
        # Generate answer
        print("Generating answer...")
        answer = generator.generate(test_question, context)
        
        print(f"\nAnswer:\n{answer}")
        
        print("\n" + "="*60)
        print("? Generation test successful!")
        print("="*60)
        
    except Exception as e:
        print(f"? Error: {e}")
        import traceback
        traceback.print_exc()
