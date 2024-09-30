import streamlit as st
from rags.retrieval_qa_rag import RetrievalQaRAG
from rags.full_document_rag import FullDocumentRAG  # Import other RAG techniques as needed
from utils.config import EnvManager
import sys


def load_rag_implementation(technique="faiss"):
    if technique == "faiss":
        return RetrievalQaRAG(chunk_size=512, chunk_overlap=1, vector_store_path="./faiss")
    elif technique == "full_document":
        return FullDocumentRAG(chunk_size=512, chunk_overlap=30)
    else:
        st.error(f"Unknown RAG technique: {technique}")
        return None

def load_env_config():
        """Loads environment variables and returns the API key."""
        env_manager = st.session_state['env_manager'] = EnvManager()
        config = st.session_state['config'] = env_manager.config
        print(config)
        return config
        
    
    
def main():
    config = load_env_config()
    st.set_page_config(page_title="Woovl | RAG Techniques Demo",layout="wide")
    st.title("RAG Techniques Demo")
                            
                        
    if config.get('enable_ollama') == 'false' and config.get('enable_huggingface') == 'true':
        llm_source = "huggingface"
        st.session_state.setdefault('llm_source', "huggingface")
        embedding_model_source = "huggingface"
        st.session_state.setdefault('embedding_model_source', 'huggingface')
    elif config.get('enable_ollama') == 'true' and config.get('enable_huggingface') == 'false':
        llm_source = "ollama"
        st.session_state.setdefault('llm_source', "ollama")
        embedding_model_source = "ollama"
        st.session_state.setdefault('embedding_model_source', 'ollama')
    # Option to select RAG technique
    with st.sidebar:
        playground_tab, config_tab = st.tabs(["Playground", "Configuration"])
        
            
        with config_tab:
                st.subheader('LLM RAG Configuration')
                rag_technique = st.selectbox("Choose RAG Technique", ["faiss", "full_document"], disabled=True)
                
                # Load the appropriate RAG implementation
                rag_instance = load_rag_implementation(technique=rag_technique)
                # llm_tab, embedding_tab = st.tabs(["LLM", "Embedding"])
                # with llm_tab:
                options = ["huggingface", "ollama"]
                if config.get('enable_ollama') == 'false' and config.get('enable_huggingface') == 'true':
                    options = ["huggingface"]
                elif config.get('enable_ollama') == 'true' and config.get('enable_huggingface') == 'false':
                    options = ["ollama"]
                    
                llm_source = st.radio(key="llm_source",label="LLM Source", options=options)
                
                if (llm_source == "huggingface"):
                    huggingface_model = st.text_input("Huggingface Model", type="default", value="meta-llama/Llama-3.1-70B-Instruct", key="huggingface_model")
                    
                elif (llm_source == "huggingface"):
                    ollama_model = st.text_input("Ollama Model", type="default", value="llama3.1:8b-instruct-q5_K_M", key="ollama_model")
                    
                
                st.divider()
                # with embedding_tab:

                embedding_model_source = st.radio(key="embed_model_source",label="Embed Model Source", options=options)

                    
                
                    
                if (embedding_model_source == "huggingface"):
                    hf_embed_model = st.text_input("HF Embed Model", type="default", value="mixedbread-ai/mxbai-embed-large-v1", key="huggingface_embed_model", disabled=True)
                    
                elif (embedding_model_source == "ollama"):
                    ollama_embed_model = st.text_input("Ollama Embed Model", type="default", value="nomic-embed-text", key="ollama_embed_model")
                    
                
                st.divider()
                if ((st.session_state.get('llm_source') ==  'huggingface' or st.session_state.get('embed_model_source') == 'huggingface')):
                    hf_token = st.text_input("Huggingface Token", type="password", value=config['huggingface_api_key'], key="huggingface_api_key")
    
    with playground_tab:
        st.subheader('LLM RAG Playground')
        rag_instance.sidebar_ui()
        st.divider()
    if rag_instance:
        rag_instance.run()

if __name__ == "__main__":
    main()
