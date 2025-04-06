import numpy as np
from collections import defaultdict
import math

class TfidfVectorStore:
    def __init__(self):
        self.documents=[]
        self.tokenized_docs=[]
        self.vocab=set()
        self.word_to_index={}
        self.tfidf_matrix=None
        self.idf_vector=None
    
    def _tokenize(self,text):
        return text.lower().split()
    
    def add_document(self,doc):
        self.documents.append(doc)
        tokens=self._tokenize(doc)
        self.tokenized_docs.append(tokens)
        self.vocab.update(tokens)

    def build(self):
        self.vocab=sorted(list(self.vocab))
        self.word_to_index={word:i for i,word in enumerate(self.vocab)}
        N=len(self.documents)
        V=len(self.vocab)
    
        tf_matrix=np.zeros((N,V))
        for i,tokens in enumerate(self.tokenized_docs):
            for word in tokens:
                if word in self.word_to_index:
                    tf_matrix[i][self.word_to_index[word]]+=1
            tf_matrix[i]/=len(tokens)

        # Document Frequency
        df=np.zeros(V)
        for j,word in enumerate(self.vocab):
            for doc_tokens in self.tokenized_docs:
                if word in doc_tokens:
                    df[j]+=1

        self.idf_vector=np.log((N)/(1+df))
        self.tfidf_matrix=tf_matrix*self.idf_vector

    def _vectorize_query(self,query):
        tokens=self._tokenize(query)
        vec=np.zeros(len(self.vocab))
        for word in tokens:
            if word in self.word_to_index:
                vec[self.word_to_index[word]]=+1
        if len(tokens)>0:
            vec/=len(tokens)
        return vec*self.idf_vector
    
    def query(self,query_text,tok_k=2):
        query_vector=self._vectorize_query(query_text)
        similarities=[]

        for i, doc_vector in enumerate(self.tfidf_matrix):
            sim=self._cosine_similarity(query_vector,doc_vector)
            similarities.append((i,sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:tok_k]
    
    def _cosine_similarity(self,a,b):
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        if denom==0:
            return 0.0
        return np.dot(a,b)/denom


