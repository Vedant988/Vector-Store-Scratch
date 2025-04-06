# Vector-Store-Scratch

This repository contains implementations of various vector store approaches, built from scratch to demonstrate the mathematical concepts and fundamentals behind different embedding and retrieval techniques.

## Repository Structure

```
Vector-Store-Scratch/
├── _pycache_/
├── RAG-Engine-ProtoTypes/
├── SentenceTransformer-Vector-Store/
├── Simple-Vector-Store/
│   ├── _pycache_/
│   ├── app.py
│   └── vector_store.py
├── TFIDF-Vector-Store/
│   └── ....
├── Word2Vec-Vector-Store/
│   └── ....
├── .gitattributes
└── README.md
```

## Project Overview

This repository contains a progressive implementation of vector stores, starting with simple approaches and gradually incorporating more sophisticated and complex (current-time which we use) techniques. Each folder represents a different implementation, with increasing complexity and capabilities.

### Implementation Progression

1. **Simple-Vector-Store**
   - Basic vector store implementation
   - Fundamental vector operations and similarity calculations
   - Simple storage and retrieval mechanisms

2. **TFIDF-Vector-Store**
   - Implementation of Term Frequency-Inverse Document Frequency
   - Text vectorization using statistical methods
   - Improved relevance scoring

3. **Word2Vec-Vector-Store**
   - Word embeddings using Word2Vec approach
   - Semantic similarity calculations
   - Context-aware vector representations

4. **SentenceTransformer-Vector-Store**
   - Sentence-level embeddings using transformer architecture
   - Advanced semantic understanding
   - Improved contextual representations

5. **RAG-Engine-ProtoTypes**
   - Retrieval Augmented Generation implementations
   - Custom RAG pipeline development
   - Integration of retrieval mechanisms with generation capabilities

## Getting Started

### Prerequisites
- Python 3.7+
- Not such required as everything is coded using numpy / scratch

### Installation
```bash
git clone https://github.com/Vedant988/Vector-Store-Scratch.git
cd Vector-Store-Scratch
```

## Learning Purpose

This repository was created for self understanding purposes to understand:
- Mathematical fundamentals behind vector representations
- Progression of embedding techniques from simple to complex
- Implementation details of vector storage and retrieval
- Building custom RAG pipelines from scratch

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. If you find any issues, please fork the repository and create a pull request with your fix. Thank you!
