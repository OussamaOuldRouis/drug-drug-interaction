import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

class DrugInteractionRetriever:
    def __init__(self, index_path='index_data/drug_interactions.index', 
                 df_path='index_data/drug_interactions_df.pkl',
                 model_name='all-MiniLM-L6-v2'):
        """Initialize the retriever with the FAISS index and data."""
        self.index = faiss.read_index(index_path)
        self.df = pd.read_pickle(df_path)
        self.model = SentenceTransformer(model_name)
        
    def query(self, drug1, drug2, k=3):
        """Query the index for interactions between two drugs."""
        # Create query text
        query_text = f"Interaction between {drug1} and {drug2}"
        
        # Create embedding for the query
        query_embedding = self.model.encode([query_text])[0].astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.df):  # Ensure the index is valid
                result = {
                    'drug1': self.df.iloc[idx]['drug1'],
                    'drug2': self.df.iloc[idx]['drug2'],
                    'description': self.df.iloc[idx]['description'],
                    'similarity_score': 1 / (1 + distance)  # Convert distance to similarity score
                }
                results.append(result)
        
        return results

def main():
    # Example usage
    retriever = DrugInteractionRetriever()
    
    # Example query
    drug1 = "aspirin"
    drug2 = "warfarin"
    
    print(f"\nSearching for interactions between {drug1} and {drug2}...")
    results = retriever.query(drug1, drug2)
    
    print("\nFound interactions:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Interaction between {result['drug1']} and {result['drug2']}")
        print(f"Description: {result['description']}")
        print(f"Similarity Score: {result['similarity_score']:.2f}")

if __name__ == "__main__":
    main() 