# Langchain-RAG-Techniques

## Introduction

"Langchain-RAG-Techniques" is a sophisticated application designed to explore Retrieval-Augmented Generation (RAG) techniques using Langchain. This application utilizes OpenAI's language models, ChromaDB, and the Langchain library to enable users to query and interact with various document formats such as PDF, DOCX, and TXT files through an intuitive Streamlit interface.

## Features

- **RAG-Powered Chat Interface**: Combines Retrieval-Augmented Generation with natural language processing capabilities using OpenAI's models.
- **Document Processing**: Efficiently loads, processes, and chunks documents for enhanced interaction.
- **Vector Storage with FAISS**: Leverages FAISS for storing and retrieving document embeddings, ensuring rapid response times.
- **Environment Configuration**: Utilizes a simple `.env` setup for managing API keys.
- **Interactive Web Application**: Built with Streamlit, providing a user-friendly interface for seamless document interaction.

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/sowole-aims/langchain-rag-techniques.git
   ```

   ```bash
   cd langchain-rag-techniques
   ```
2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Configuration**: Create a `.env` file in the project root and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Start the Application**:

   ```bash
   streamlit run langchain_rag_app.py
   ```

   Open your browser and navigate to `http://localhost:8501` to interact with the application.
2. **Upload Documents**: Use the file uploader to upload documents (PDF, DOCX, TXT) that you want to query.
3. **Interact with the Chat**:

   - Enter your questions related to the uploaded documents in the chat interface.
   - Adjust the chunk size and number of retrieved documents (`k`) from the sidebar as needed.
   - Click "Add Data" to process and embed the uploaded document.
4. **View Results**: The application will display the generated answers along with the relevant chunks from the documents.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project utilizes Langchain, OpenAI, FAISS, and Streamlit to demonstrate effective RAG techniques and document processing capabilities.
