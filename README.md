# RAG Knowledge Assistant

## Overview

**RAG Knowledge Assistant** is a document-aware chatbot built using Retrieval-Augmented Generation (RAG). It enables users to upload internal company documents (PDFs) and ask natural language questions, receiving accurate, context-based answers powered by large language models. This assistant is designed to simulate a real-world internal support tool for HR, onboarding, policy reference, or customer-facing document Q&A.

## Features

- Upload and process internal documents (PDF)
- Embed documents using OpenAIEmbeddings
- Store vectors persistently using ChromaDB
- Retrieve top-k document chunks via semantic search
- Answer questions using GPT-3.5-turbo or LLaMA
- Maintain conversational context using memory buffers
- Display source documents supporting the response
- Log and review all user interactions and answers
- Deployable on Streamlit Cloud with zero setup


## Tech Stack

- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Embeddings**: OpenAI Embeddings
- **LLM Backend**: GPT-3.5 (via OpenAI API), Ollama:2b
- **Vector Store**: ChromaDB with disk persistence
- **Document Parsing**: LangChain PyPDFLoader

