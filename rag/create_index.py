import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

def load_data(csv_path):
    """Load and preprocess the drug interaction data."""
    df = pd.read_csv(csv_path)
    # Combine drug1 and drug2 with description for better context
    df['combined_text'] = df.apply(
        lambda x: f"Interaction between {x['drug1']} and {x['drug2']}: {x['description']}", 
        axis=1
    )
    return df

def create_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Create embeddings using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    """Create and return a FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def save_index_and_data(index, df, output_dir='index_data'):
    """Save the FAISS index and the dataframe."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the FAISS index
    faiss.write_index(index, os.path.join(output_dir, 'drug_interactions.index'))
    
    # Save the dataframe
    df.to_pickle(os.path.join(output_dir, 'drug_interactions_df.pkl'))
    
    print(f"Index and data saved in {output_dir}")

def main():
    # Load and process data
    print("Loading data...")
    df = load_data('db_drug_interactions.csv')
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(df['combined_text'].values)
    
    # Create FAISS index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)
    
    # Save index and data
    print("Saving index and data...")
    save_index_and_data(index, df)
    
    print("Done! Index created successfully.")

if __name__ == "__main__":
    main() 