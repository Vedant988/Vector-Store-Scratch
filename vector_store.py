import numpy as np

class vector_store:
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
            