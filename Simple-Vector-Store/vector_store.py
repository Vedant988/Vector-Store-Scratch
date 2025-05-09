import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data = {}
        self.vector_index = {}
    
    def add_vector(self,vector_id,vector):
        self.vector_data[vector_id]=vector
        self.update_vector(vector_id,vector)
    
    def get_vector(self,vector_id):
        return self.vector_data.get(vector_id)
    
    def update_vector(self,vector_id,vector):
        for existing_id,existing_vector in self.vector_data.items():
            similarity = np.dot(vector,existing_vector)/(np.linalg.norm(vector)*np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id]={}
            self.vector_index[existing_id][vector_id]=similarity
    
    def find_similarity_vectors(self,query_vector,num_results=5):
        results=[]
        for vector_id,vector in self.vector_data.items():
            similarity=np.dot(query_vector,vector)/(np.linalg.norm(query_vector)*np.linalg.norm(vector))
            results.append((vector_id,similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"\nresult: {results}")
        return results[:num_results]

