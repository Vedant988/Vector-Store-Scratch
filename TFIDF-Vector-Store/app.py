"""
why TF-IDF?
- TF-IDF downweights common words and highlights unique/rare term that carry more sementic meaning!!
- it allows better sementic retrieval than raw frequency count

"""

from tfidf_vector_store import TfidfVectorStore

docs = [
"machine learning is fun",
"deep learning is the part of machine learning",
"natural language processing uses machine learning"
]

query="machine learning"
TFIDF = TfidfVectorStore()

for doc in docs:
    TFIDF.add_document(doc)

TFIDF.build()

results=TFIDF.query(query,tok_k=3)
print("\nTop Matching Documents:\n")
for idx,sim in results:
    print(f"Doc {idx} -> Similarity: {sim:.4f} -> \"{docs[idx]}\"")