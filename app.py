from vector_store import VectorStore
import numpy as np

vs=VectorStore()
sentences=[
    "I am vedant from IIIT Nagpur",
    "Fruits are good for Health",
    "Vegetables are more healthier than junk food"]

vocabulary = set()
tokenized_sentences = []
for sentence in sentences:
    print("\nsentence:",sentence)
    tokens=sentence.lower().split()
    print("tokens:",tokens)
    tokenized_sentences.append(tokens)
    vocabulary.update(tokens)

# this will sort the vocabulary in alphabetic Order - for Ease to debugging
word_to_index = {word: idx for idx,word in enumerate(sorted(vocabulary))}
# word_to_index = {"am":0, "are":1, "food":2, "for":3, etc...}
# len(word_to_index) = 17

def senetence_to_vector(tokens):
    vec=np.zeros(len(word_to_index))
    # vec=[0,0,0,0...] # 17 zeros
    for token in tokens:
        if token in word_to_index:
            # for input "query" tokens, if the word is present in our vocabulary then in its value vec[index]=1
            vec[word_to_index[token]]+=1
        # else:
            # Teated as OOV - Out Of Vocabulary
            # for improvement we generally use Sementic Search (using BERT, word2vec,etc) to handle OOV words
    return vec

for i, tokens in enumerate(tokenized_sentences):
    vector=senetence_to_vector(tokens)
    vs.add_vector(f"sentence_{i}",vector)

print(f"\n vs.vector_data \n{vs.vector_data},\n\n vs.vector_index\n{vs.vector_index}")

query = ""
query_tokens=query.lower().split()
query_vector=senetence_to_vector(query_tokens)

# FINDING COSINE SIMILARITY
similarity_score=vs.find_similarity_vectors(query_vector,num_results=2)

print("\nQuery:",query)
print("\nTop Similar Sentence:")
for sentence_id,similarity in similarity_score:
    print(f"{sentence_id} --> similarity: {similarity:.4f}")

