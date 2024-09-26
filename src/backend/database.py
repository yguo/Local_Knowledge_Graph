from chromadb import Client
from chromadb.config import Settings
import uuid
class ChromaDatabase:
    def __init__(self, path='./chroma_db', embedding_dim=4096, collection_name='graph_knowledge'):
        self.collection_name = collection_name
        self.client = Client(Settings(persist_directory=path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_dim = embedding_dim
        self.embeddings = []  
        self. texts = []                   

    def insert_data(self, text, embedding, is_question, id = None   ):
        unique_id = str(uuid.uuid4()) if id is None else id
        self.collection.add(
            ids=[unique_id],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{"is_question": is_question}]
        )
        self.embeddings.append(embedding)
        self.texts.append(text)                

    def clear_database(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            print(f"Error clearing database collection {self.collection_name}: {e}")

    def get_all_embeddings(self):
        return self.collection.get()

    def find_similar(self, query_embedding, top_k=5):
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], 
                                        n_results=top_k, 
                                        where={"is_question": True})
        processed_results = []
        for i, (distance, text, metadata) in enumerate(zip(results['distances'][0], 
                                                          results['documents'][0], 
                                                          results['metadatas'][0])):
            similarity = 1 - distance
            is_question = metadata.get("is_question", False)
            processed_results.append({"id": i, "text": text, "similarity": similarity, "is_question": is_question})

        return processed_results