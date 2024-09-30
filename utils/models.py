from dataclasses import dataclass


# Define a model for Vector Store metadata
@dataclass
class VectorStoreMetadata:
    store_name: str
    path: str
    chunk_size: int = None
    chunk_count: int = None
    document_count: int = None
    chunk_method: str = None

# Define a model for Embedding information
@dataclass
class EmbeddingInfo:
    embedding_model: str
    vector_count: int
    dimension: int

# Define a model for stored files
@dataclass
class SavedFile:
    file_name: str
    file_path: str
    file_size: int
    upload_date: str