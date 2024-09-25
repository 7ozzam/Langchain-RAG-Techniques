import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import tiktoken


# Utility to load environment variables
def load_env_vars():
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key


# Load and process document based on file extension
def load_document_elements(file_path):
    from unstructured.partition.auto import partition
    
    _, extension = os.path.splitext(file_path)

    with open(file_path, "rb") as f:
        elements = partition(file=f, include_page_breaks=True)
    return elements;


# def document_to_elements(file):
#     from unstructured.partition.auto import partition
#     elements = partition(file=file)

# Split document into manageable chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain_core.documents import Document
    from unstructured.chunking.title import chunk_by_title
    
    elements = data
    chunked_elements = chunk_by_title(elements)

    documents = []
    for element in chunked_elements:
        metadata = element.metadata.to_dict()
        documents.append(Document(page_content=element.text,
                                metadata=metadata))

    return documents


# Generate embeddings using Ollama's embedding model and store using FAISS
def create_embeddings(chunks, storing_path="vectorstore"):
    try:
        vector_store = FAISS.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"))
        vector_store.save_local(storing_path)
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None


# Restore vector store from a local storage path
def restore_vector_store(storing_path="vectorstore"):
    try:
        if os.path.exists(storing_path):
            vector_store = FAISS.load_local(storing_path, OllamaEmbeddings(model="nomic-embed-text"), allow_dangerous_deserialization = True)
            st.success('Vector store restored successfully.')
            return vector_store
        else:
            st.error(f"Vector store not found at path: {storing_path}")
            return None
    except Exception as e:
        st.error(f"Error restoring vector store: {e}")
        return None


# Retrieve answers using a vector store and Ollama LLM
def ask_and_get_answer(vector_store, query, k=3):
    try:
        llm = OllamaLLM(model="llama3.1:8b-instruct-q5_K_M")
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
        results = vector_store.similarity_search(query,k=k)
        retrieved_docs = retriever.get_relevant_documents(query)
        
        from langchain.chains import RetrievalQA
        
        # chain = llm

        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        print(retrieved_docs)
        return chain.invoke(query)
    except Exception as e:
        st.error(f"Error retrieving answer: {e}")
        return None


# Calculate embedding cost based on token count
# def calculate_embedding_cost(texts):
#     enc = tiktoken.get_encoding("cl100k_base")
#     total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
#     cost = total_tokens / 1000 * 0.0004
#     return total_tokens, cost


# Clear session history in Streamlit
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Streamlit app logic
# Streamlit app logic
def main():
    load_env_vars()
    st.set_page_config(layout="wide")
    st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    answer = None
    
    with st.sidebar:
        # File uploader and input widgets
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt', 'md'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=4096, value=4096, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=50, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)
        restore_data = st.button('Restore Embedded Data')

        if uploaded_file and add_data:
            try:
                with st.spinner('Processing the file...'):
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(uploaded_file.read())

                    data = load_document_elements(file_name)
                    if data:
                        chunks = chunk_data(data, chunk_size=chunk_size)
                        st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                        # tokens, embedding_cost = calculate_embedding_cost(chunks)
                        # st.write(f'Embedding cost: ${embedding_cost:.4f}')
                        vector_store = create_embeddings(chunks)
                        st.session_state.vs = vector_store
                        st.success('File uploaded and embedded successfully.')
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        # Restore vector store from saved embeddings
        if restore_data:
            vector_store = restore_vector_store()
            if vector_store:
                st.session_state.vs = vector_store

    col1, col2 = st.columns(2)
    with col1:
        # Ask a question
        q = st.text_input('Ask a question about the content of your file:')
        if q and 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store, q, k)
            
        if answer is not None and 'result' in answer:
            st.text_area('LLM Answer:', value=answer["result"])
            st.divider()
                    
            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Q: {q} \nA: {answer["result"]}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)

    with col2:
        chunks_view = ''
        if answer is not None and 'source_documents' in answer:
            for chunk in answer["source_documents"]:
                chunks_view += chunk.page_content + '\n\n\n--------------------------------------------------\n\n\n'
                st.text_area('RAG Answer:', value=chunk.page_content)
                st.divider()

        
if __name__ == "__main__":
    main()
