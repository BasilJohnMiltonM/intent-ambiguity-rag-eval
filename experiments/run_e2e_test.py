import os
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.retrieve import RAGRetriever, format_context
from pipeline.generate import RAGGenerator

def run_e2e_test(num_unambiguous=3, num_ambiguous=2):
    """
    Run end-to-end test on sample questions.
    
    Args:
        num_unambiguous: Number of unambiguous questions to test
        num_ambiguous: Number of ambiguous questions to test
    """
    print("="*60)
    print("End-to-End RAG Pipeline Test")
    print("="*60 + "\n")
    
    # Load questions
    with open('data/questions.json', 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    # Separate unambiguous and ambiguous
    unambiguous = [q for q in all_questions if q['ambiguity_label'] == 0][:num_unambiguous]
    ambiguous = [q for q in all_questions if q['ambiguity_label'] == 1][:num_ambiguous]
    
    test_questions = unambiguous + ambiguous
    
    print(f"Selected {len(test_questions)} test questions")
    print(f"  - Unambiguous: {len(unambiguous)}")
    print(f"  - Ambiguous: {len(ambiguous)}\n")
    
    # Initialize retriever and generator
    print("Initializing retriever and generator...")
    retriever = RAGRetriever()
    generator = RAGGenerator(model="gpt-4o-mini")
    print("? Ready\n")
    
    # Run pipeline on each question
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}/{len(test_questions)}: {question['question_id']}")
        
        # Retrieve
        retrieved = retriever.retrieve(question['question'], k=3)
        context = format_context(retrieved)
        
        # Generate
        answer = generator.generate(question['question'], context)
        
        # Store result
        result = {
            'question_id': question['question_id'],
            'question': question['question'],
            'ambiguity_label': question['ambiguity_label'],
            'retrieved_chunks': [r['chunk_id'] for r in retrieved],
            'generated_answer': answer,
            'ground_truth': question.get('ground_truth', question.get('ground_truth_answers', []))
        }
        
        results.append(result)
        
        print(f"  Answer: {answer[:80]}...\n")
    
    # Save results
    output_path = 'results/e2e_test_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("="*60)
    print(f"? E2E test complete! Results saved to {output_path}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = run_e2e_test(num_unambiguous=3, num_ambiguous=2)
