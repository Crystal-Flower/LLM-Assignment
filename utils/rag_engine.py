import os
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self, df):
        self.df = df
        self.vector_db = None
        # Use llama3.2 for embeddings (or another local model if preferred)
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        """
        Creates a FAISS vector database from student profiles.
        """
        documents = []
        for _, row in self.df.iterrows():
            # Create a rich text representation of the student profile
            content = f"""
            Name: {row['Name']}
            Department: {row['Department']}
            Year: {row['Year']}
            Technical Skills: {row['Technical Skills']}
            Research Areas: {row['Research Areas']}
            Career Path: {row['Intended Career Path']}
            """
            doc = Document(page_content=content, metadata={"name": row['Name']})
            documents.append(doc)
        
        self.vector_db = FAISS.from_documents(documents, self.embeddings)

    def search_students(self, query, k=5):
        """
        Performs similarity search to find students matching a query.
        """
        if not self.vector_db:
            return []
        
        results = self.vector_db.similarity_search(query, k=k)
        return results

if __name__ == "__main__":
    # Test stub
    try:
        df_test = pd.read_csv(os.path.join("data", "student_data.csv"))
        engine = RAGEngine(df_test)
        matches = engine.search_students("students interested in Computer Vision and Deep Learning")
        for doc in matches:
            print(f"Match: {doc.metadata['name']}")
    except Exception as e:
        print(f"Error testing RAGEngine: {e}")
