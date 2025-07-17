# RAG Knowledge Assistant

The RAG Knowledge Assistant is a Retrieval-Augmented Generation (RAG) web application built using Streamlit for the user interface, LangChain for the RAG pipeline, and ChromaDB as the vector store. It supports dynamic switching between OpenAI and Ollama for both language model and embedding providers. It also includes source tracking for document-based answers.

## Features

- Document Upload: Upload PDF and TXT files to build your knowledge base.
- Semantic Search: Performs semantic search over your documents to retrieve relevant information.
- Conversational Interactions: Engage in multi-turn conversations about your uploaded documents.
- Dynamic LLM Backend Switching:
  - OpenAI: Uses OpenAI's chat models and embeddings (requires API key).
  - Ollama: Runs local Ollama models like LLaMA2 or Mistral.
- Source Tracking: Displays specific document chunks and their sources (file name, page number) that contributed to the assistant's response.

## Technologies Used

- Streamlit – For creating the interactive web application
- LangChain – For building the RAG pipeline (loading, splitting, retrieval, memory)
- ChromaDB – For storing document embeddings in-memory
- OpenAI – For language models and embedding services (API-based)
- Ollama – For running local open-source LLMs
- PyPDF – For loading PDF documents

## To run
Install the depenedecies in venv
```bash
pip install streamlit langchain langchain-community pypdf openai chromadb
```
Set your OpenAI API key if you want to use ChatGPT, or ensure Ollama is installed and running if you prefer to use a local LLM like Llama.

Start the application:
```bash
streamlit run app.py
```
