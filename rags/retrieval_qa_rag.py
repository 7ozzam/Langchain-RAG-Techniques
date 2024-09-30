import os
from typing import List, Optional
import streamlit as st
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rags.base_rag import BaseRAG
from unstructured.partition.auto import partition
from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate



class RetrievalQaRAG(BaseRAG):
    """Implementation of the RAG system using FAISS for vector storage and retrieval."""

    def __init__(self, chunk_size, chunk_overlap, vector_store_path="faiss"):
        super().__init__()
        self.chunk_method = 'RetrievalQA'
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path

    def load_and_chunk_file(self, file_path: str, chunk_size: int = 512) -> List[Document]:
        """Loads and chunks the file for FAISS embedding."""
        print("Loading file: ", file_path)
        try:
            with open(file_path, "rb") as f:
                elements = partition(file=f, include_page_breaks=True, strategy="auto")

                if elements:
                    chunked_elements = chunk_by_title(elements, max_characters=chunk_size, overlap=self.chunk_overlap, combine_text_under_n_chars=50)
                    return [Document(page_content=e.text, metadata=e.metadata.to_dict()) for e in chunked_elements]
                return []
        except Exception as e:
            logging.error(f"Error chunking file: {e}")
            st.error(f"Error chunking file: {e}")
            return []

    def create_embeddings(self, chunks: List[Document], vector_store_save_path: str) -> Optional[FAISS]:
        """Creates FAISS embeddings and saves them locally."""
        try:
            current_embed_model = self.get_embed_model()
            vector_store = FAISS.from_documents(chunks, current_embed_model)
            vector_store.save_local(vector_store_save_path)
            return vector_store
        except Exception as e:
            logging.error(f"Error creating embeddings: {e}")
            st.error(f"Error creating embeddings: {e}")
            return None

    def restore_vector_store(self, storing_path: str) -> Optional[FAISS]:
        """Restores the FAISS vector store from a given path."""
        if os.path.exists(storing_path):
            current_embed_model = self.get_embed_model()
            return FAISS.load_local(storing_path, current_embed_model, allow_dangerous_deserialization=True)
        else:
            logging.error(f"Vector store not found at: {storing_path}")
            st.error(f"Vector store not found at: {storing_path}")
            return None

    def ask_and_get_answer(self, vector_store: FAISS, query: str, k: int = 3) -> Optional[dict]:
        """Retrieves the answer to a query from the FAISS vector store."""
        
        # try:
        llm = self.get_llm()
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
        print("Retriever: ", retriever)
        print("Query: ", query)
        print("LLM: ", llm)
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        print("Chain: ", chain)
        return chain.invoke(query)
        # except Exception as e:
        #     logging.error(f"Error during query answering: {e}")
        #     st.error(f"Error during query answering: {e}")
        #     return None
