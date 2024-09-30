from abc import ABC, abstractmethod
from typing import List, Optional, cast

import streamlit as st
import os
import logging
import json
import utils.models as models
import sys


class BaseRAG(ABC):
    """Base class for RAG system, follows the SOLID principles for extendability."""
    
    def __init__(self, vector_store_path: str = 'general', chunk_method: str = 'default'):
        self.chunk_method = chunk_method
        self.vector_store = None
        self.answer = None
        self.vector_store_path = vector_store_path
        self.k = None
        self.chunk_size = None
        self.load_initial_state()

    def load_initial_state(self):
        """Initializes environment variables and session states."""
        # st.session_state.setdefault('ollama_model', '')
        # st.session_state.setdefault('huggingface_model', '')
        # st.session_state.setdefault('huggingface_api_key', '')
        st.session_state.setdefault('selected_stores', [])
        st.session_state.setdefault('available_stores', [])

    def get_current_llm_source(self) -> str:
        """Returns the current LLM source from session state."""
        return st.session_state.get('llm_source', '')

    def get_current_embed_model_source(self) -> str:
        """Returns the current LLM source from session state."""
        return st.session_state.get('embed_model_source', '')
    
    def get_current_llm_model(self) -> str:
        """Returns the current LLM model based on the selected source."""
        llm_source = self.get_current_llm_source()
        return st.session_state.get(f'{llm_source}_model', '')
    
    def get_current_embed_model(self) -> str:
        """Returns the current LLM model based on the selected source."""
        llm_source = self.get_current_embed_model_source()
        return st.session_state.get(f'{llm_source}_embed_model', '')

    def get_current_huggingface_api_key(self) -> str:
        """Returns the current API key based on the selected source."""
        return st.session_state.get('huggingface_api_key', '')

    def reload_vector_stores(self):
        """Reloads vector stores from the storage path."""
        st.session_state['available_stores'] = self.load_vector_stores()

    def clear_history(self):
        """Clears session history."""
        st.session_state.pop('history', None)

    def file_upload_ui(self):
        """File upload interface."""
        return st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt', 'md'])

    def get_llm(self):
        """Gets the current LLM model based on the selected source."""
        llm_source = self.get_current_llm_source()
        current_model = self.get_current_llm_model()
        current_api_key = self.get_current_huggingface_api_key()
        
        if llm_source == 'huggingface':
            from langchain_huggingface import HuggingFaceEndpoint
            return HuggingFaceEndpoint(repo_id=current_model, huggingfacehub_api_token=current_api_key, task="text-generation", max_new_tokens=8096, do_sample=False, temperature=0.01, trust_remote_code=True)
        elif llm_source == 'ollama':
            from langchain_ollama.llms import OllamaLLM
            return OllamaLLM(model=current_model)
        return None
    
    def get_embed_model(self):
        """Gets the current LLM model based on the selected source."""
        embed_model_source = self.get_current_embed_model_source()
        current_embed_model = self.get_current_embed_model()
        current_api_key = self.get_current_huggingface_api_key()
        
        if embed_model_source == 'huggingface':
            # from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
            # from langchain_huggingface import HuggingFaceEndpointEmbeddings

            # return HuggingFaceEndpointEmbeddings(repo_id=current_embed_model, huggingfacehub_api_token=current_api_key)
            # return HuggingFaceInferenceAPIEmbeddings(api_key=current_api_key, model_name=current_embed_model)
            return HuggingFaceEndpointEmbeddings(repo_id=current_embed_model, task="feature-extraction", huggingfacehub_api_token=current_api_key)
            
        elif embed_model_source == 'ollama':
            from langchain_ollama.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(model=current_embed_model)
        return None

    @abstractmethod
    def load_and_chunk_file(self, file_path: str, chunk_size: int = 256, chunk_overlap: int = 20):
        """To be implemented by subclasses to load and chunk files."""
        pass

    @abstractmethod
    def create_embeddings(self, chunks: List[str]):
        """To be implemented by subclasses to create embeddings."""
        pass

    def create_meta_file(self, store_name: str, path: str, chunk_size: int, chunk_count: int, document_count: int):
        """Creates a metadata file for a vector store."""
        meta_file_path = os.path.join(path, 'meta.json')
        meta_data = {
            'store_name': store_name,
            'path': path,
            'chunk_size': chunk_size,
            'chunk_count': chunk_count,
            'chunk_method': self.chunk_method,
            'documents_inside': document_count
        }
        with open(meta_file_path, 'w') as meta_file:
            json.dump(meta_data, meta_file)

    def load_vector_stores(self) -> List[models.VectorStoreMetadata]:
        """Loads the available vector stores and reads metadata."""
        vector_stores = []
        try:
            if os.path.exists(self.vector_store_path):
                for store_name in os.listdir(self.vector_store_path):
                    store_path = os.path.join(self.vector_store_path, store_name)
                    if os.path.isdir(store_path):
                        meta_file_path = os.path.join(store_path, 'meta.json')
                        if os.path.exists(meta_file_path):
                            with open(meta_file_path, 'r') as meta_file:
                                try:
                                    metadata = json.load(meta_file)
                                    store_metadata = models.VectorStoreMetadata(
                                        store_name=metadata.get('store_name', store_name),
                                        path=store_path,
                                        chunk_size=metadata.get('chunk_size', 256),
                                        chunk_count=metadata.get('chunk_count', 256),
                                        chunk_method=metadata.get('chunk_method', 'default'),
                                        document_count=metadata.get('documents_inside', 0)
                                    )
                                    vector_stores.append(store_metadata)
                                except json.JSONDecodeError:
                                    logging.error(f"Failed to decode JSON in {meta_file_path}")
                        else:
                            vector_stores.append(models.VectorStoreMetadata(store_name=store_name, path=store_path))
            return vector_stores
        except Exception as e:
            logging.error(f"Error loading vector stores: {e}")
            return []

    @abstractmethod
    def restore_vector_store(self, storing_path: str):
        """To be implemented by subclasses to restore vector store."""
        pass

    @abstractmethod
    def ask_and_get_answer(self, vector_store, query: str, k: int = 3):
        """To be implemented by subclasses to handle query answering."""
        pass

    def handle_file_upload(self, uploaded_file, chunk_size: int):
        """Handles file upload and embedding creation."""
        try:
            with st.spinner('Processing the file...'):
                file_path = self.save_uploaded_file(uploaded_file)
                chunks = self.load_and_chunk_file(file_path, chunk_size)
                if chunks:
                    from datetime import datetime
                    timestamp = str(round(datetime.now().timestamp()))
                    file_unique_name = f"{os.path.splitext(uploaded_file.name)[0]}_{timestamp}"
                    vector_store_save_path = os.path.join(self.vector_store_path, file_unique_name)
                    vector_store = self.create_embeddings(chunks, vector_store_save_path)
                    self.create_meta_file(store_name=file_unique_name, path=vector_store_save_path, chunk_size=chunk_size, chunk_count=len(chunks), document_count=1)
                    if vector_store:
                        st.session_state.vs = vector_store
                        st.success('File uploaded and embedded successfully.')
                        self.reload_vector_stores()
        except Exception as e:
            logging.error(f"Error during file upload: {e}")
            st.error(f"Error during file upload: {e}")

    def save_uploaded_file(self, uploaded_file):
        """Saves the uploaded file locally."""
        file_path = os.path.join('./uploads/', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        return file_path

    def handle_restore_data(self):
        """Restores the selected vector stores."""
        for store in cast(List[models.VectorStoreMetadata], st.session_state['selected_stores']):
            vector_store = self.restore_vector_store(store.path)
            
            if vector_store:
                st.session_state.vs = vector_store
                st.success(f"Successfully restored vector store: {store.store_name}")

    def sidebar_ui(self):
        """UI for the sidebar interactions."""
        uploaded_file = self.file_upload_ui()
        self.chunk_size = st.number_input('Chunk size:', min_value=100, max_value=4096, value=1024, on_change=self.clear_history)
        
        add_data = st.button('Add Data', on_click=self.clear_history)
        if uploaded_file and add_data:
            self.handle_file_upload(uploaded_file, self.chunk_size)

        st.session_state['available_stores'] = self.load_vector_stores()
        # Assuming available_stores and selected_stores are initialized in session state
        if 'available_stores' not in st.session_state:
            st.session_state['available_stores'] = []  # Populate with actual VectorStoreMetadata objects
        if 'selected_stores' not in st.session_state:
            st.session_state['selected_stores'] = None
        # Create a list of store names to display in the radio button
        store_names = [store.store_name for store in cast(List[models.VectorStoreMetadata], st.session_state['available_stores'])]

        

        # Find the selected store from the list
        if len(store_names) > 0:
            
            st.write("Available Vector Stores:")

           
            # Use st.radio to allow selecting only one store at a time
            selected_store_name = st.radio("Select a store", store_names)
            selected_store = next(store for store in st.session_state['available_stores'] if store.store_name == selected_store_name)

            # Display selected store information
            st.code(f"Store Name: {selected_store.store_name}\nChunk Method: {selected_store.chunk_method or 'Unknown'}\n"
                    f"Chunk Size: {selected_store.chunk_size or 'Unknown'}\nChunk Count: {selected_store.chunk_count or 'Unknown'}\n"
                    f"Document Count: {selected_store.document_count or 'Unknown'}")

            # Update session state to track the selected store
            st.session_state['selected_stores'] = [selected_store]
            self.handle_restore_data()

        
        # for store in cast(List[models.VectorStoreMetadata], st.session_state['available_stores']):
        #     checkbox = st.checkbox(store.store_name, key=store.store_name)
        #     if checkbox:
        #         st.code(f"Store Name: {store.store_name}\nChunk Method: {store.chunk_method or 'Unknown'}\n"
        #                 f"Chunk Size: {store.chunk_size or 'Unknown'}\nChunk Count: {store.chunk_count or 'Unknown'}\n"
        #                 f"Document Count: {store.document_count or 'Unknown'}")
        #     if checkbox and store not in st.session_state['selected_stores']:
        #         st.session_state['selected_stores'].append(store)
        #     elif not checkbox and store in st.session_state['selected_stores']:
        #         st.session_state['selected_stores'].remove(store)

        # restore_data = st.button('Restore Embedded Data')

   
        
        # if restore_data:
        #     self.handle_restore_data()

    def run(self):
        """Main entry point for the app."""
        
        # with st.sidebar:
        #     self.sidebar_ui()

        col1, col2 = st.columns(2)
        with col1:
            query = st.text_input('Ask a question:')
            if query and 'vs' in st.session_state:
                vector_store = st.session_state.vs
                self.k = st.number_input('Number of results (k):', min_value=1, max_value=50, value=3, on_change=self.clear_history)
                self.answer = self.ask_and_get_answer(vector_store, query, k=self.k)

            if self.answer and 'result' in self.answer:
                self.display_answer(query)

        with col2:
            if self.answer and 'source_documents' in self.answer:
                for chunk in self.answer['source_documents']:
                    st.text_area('RAG Answer:', value=chunk.page_content)
                    st.divider()

    def display_answer(self, query):
        """Displays the answer in the UI."""
        st.text_area('LLM Answer:', value=self.answer["result"], height=800)
        st.divider()

        if 'history' not in st.session_state:
            st.session_state.history = ''

        value = f'Q: {query} \nA: {self.answer["result"]}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        st.text_area('Chat History', value=st.session_state.history, height=400)


