## Overview
A controlled evaluation study examining whether standard RAG evaluation metrics (RAGAS) are sensitive to intent ambiguity in user queries.

## Research Question
Do standard RAG evaluation metrics overestimate answer quality when user queries are ambiguous?

## Dataset
- 50 questions total
- 25 unambiguous (clear single interpretation)
- 25 ambiguous (multiple valid interpretations)
- Domain: Python documentation

## Experiments
- **E1**: Compare RAGAS metrics on ambiguous vs. unambiguous queries
- **E2**: Prompt intervention—does asking the model to state its interpretation change behavior?
