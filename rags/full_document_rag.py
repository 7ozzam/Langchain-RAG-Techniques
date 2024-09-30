from rags.base_rag import BaseRAG

class FullDocumentRAG(BaseRAG):
    def __init__(self, chunk_size=256, chunk_overlap=20, vector_store_path="full_document"):
        super().__init__()
        self.chunk_method = 'FullDocument'
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
    def load_and_chunk_file(self, file_path):
        # Custom document chunking for Chroma
        pass

    def create_embeddings(self, chunks, storing_path="full_document"):
        # Custom embedding logic for Chroma
        pass

    def restore_vector_store(self, storing_path="full_document"):
        # Restore Chroma vector store logic
        pass

    def ask_and_get_answer(self, vector_store, query, k=3):
        # Custom retrieval logic for Chroma
        pass
