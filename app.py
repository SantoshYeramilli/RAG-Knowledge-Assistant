import streamlit as st 
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain 
from langchain_community.chat_models import ChatOpenAI, ChatOllama 
from langchain.memory import ConversationBufferMemory


# --- Configuration ---
# Set your OpenAI API key here or as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Define a default Ollama host, which can be overridden by an environment variable
# When running locally, if OLLAMA_HOST is not set, it defaults to localhost:11434,
# which is where Ollama typically runs.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Functions for RAG Pipeline ---

def load_documents(uploaded_files):
    """
    Loads documents from uploaded files (PDF or TXT).
    Temporarily saves files to disk to allow LangChain loaders to access them.
    """
    documents = []
    for uploaded_file in uploaded_files:
        #st.write(f"DEBUG: Processing uploaded file: {uploaded_file.name}") # DEBUG PRINT
        
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        #st.write(f"DEBUG: Detected file extension: '{file_extension}'") # DEBUG PRINT
        
        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            #st.write("DEBUG: Assigned PyPDFLoader.") # DEBUG PRINT
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
            #st.write("DEBUG: Assigned TextLoader.") # DEBUG PRINT
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            if os.path.exists(file_path):
                os.remove(file_path)
            continue

        try:
            documents.extend(loader.load())
            #st.write(f"DEBUG: Successfully loaded {uploaded_file.name}") # DEBUG PRINT
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    return documents

def split_documents(documents):
    """
    Splits documents into smaller, overlapping chunks.
    This helps the LLM process information efficiently and accurately.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk will aim for 1000 characters
        chunk_overlap=200, # Chunks will overlap by 200 characters to maintain context
        length_function=len, # Use standard Python len() for length calculation
    )
    return text_splitter.split_documents(documents)

def create_vector_store(texts, llm_provider): # Changed 'llm' to 'llm_provider'
    embeddings = None # Renamed 'embedding' to 'embeddings' for clarity
    if llm_provider == "OpenAI":
        try:
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            st.error(f"Error initializing OpenAI Embeddings. Make sure OPENAI_API_KEY is set. Error: {e}")
            return None
    elif llm_provider == "Ollama":
        try:
            embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=st.session_state.ollama_model)
        except Exception as e:
            st.error(f"Error initializing Ollama Embeddings. Ensure Ollama is running and model '{st.session_state.ollama_model}' is available at {OLLAMA_HOST}. Error: {e}")
            return None
    else:
        st.error("Invalid LLM provider selected for embeddings.")
        return None
    if embeddings: # Check for 'embeddings'
        vectorstore = Chroma.from_documents(texts, embeddings)
        return vectorstore
    return None

def get_conversation_chain(vectorstore, llm_provider, ollama_model_name):
    """
    Initializes the conversational retrieval chain based on the selected LLM provider.
    This chain connects the LLM, the retriever (vector store), and conversational memory.
    """
    llm = None
    
    # You can remove these debug prints now if you want, as they've served their purpose
    # st.write(f"DEBUG (inside get_conversation_chain): Received llm_provider: '{llm_provider}'")
    # st.write(f"DEBUG (inside get_conversation_chain): Repr of llm_provider: {repr(llm_provider)}")
    # st.write(f"DEBUG (inside get_conversation_chain): Does llm_provider == 'Ollama'? {llm_provider == 'Ollama'}")

    if llm_provider == "OpenAI":
        try:
            llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        except Exception as e:
            st.error(f"Error initializing OpenAI Chat Model. Make sure OPENAI_API_KEY is set. Error: {e}")
            return None
    elif llm_provider == "Ollama":
        try:
            llm = ChatOllama(base_url=OLLAMA_HOST, model=ollama_model_name, temperature=0.7)
        except Exception as e:
            st.error(f"Error initializing Ollama Chat Model. Ensure Ollama is running and model '{ollama_model_name}' is available at {OLLAMA_HOST}. Error: {e}")
            return None
    else:
        st.error(f"Internal Error: Unexpected LLM provider value: '{llm_provider}'. This should not happen.")
        raise ValueError(f"Unexpected LLM provider value in get_conversation_chain: '{llm_provider}'")

    if llm:
        # --- NEW CODE HERE: Add output_key to ConversationBufferMemory ---
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer' # Explicitly tell memory which key holds the answer
        )
        # --- END NEW CODE ---

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True
        )
        return conversation_chain
    return None

# --- Main Streamlit Application Function ---    

def main():
    st.set_page_config(page_title="RAG Knowledge Assistant", page_icon=":books:")
    st.header("RAG Knowledge Assistant :books:")

    # --- Initialize session state variables ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "OpenAI"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "llama2"

    # --- Sidebar Section ---
    with st.sidebar:
        st.subheader("Configuration")

        llm_provider_options = ("OpenAI", "Ollama")
        
        try:
            current_index = llm_provider_options.index(st.session_state.llm_provider)
        except ValueError:
            current_index = 0

        llm_provider_selection = st.radio(
            "Choose LLM Provider:",
            llm_provider_options,
            index=current_index,
            key="llm_provider_radio"
        )

        if st.session_state.llm_provider != llm_provider_selection:
            st.session_state.llm_provider = llm_provider_selection
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.llm_provider == "Ollama":
            ollama_model_input = st.text_input(
                "Ollama Model Name (e.g., llama2, mistral):",
                value=st.session_state.ollama_model,
                key="ollama_model_input"
            )
            if ollama_model_input != st.session_state.ollama_model:
                st.session_state.ollama_model = ollama_model_input
                st.session_state.conversation = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF or TXT files here and click 'Process'",
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )
        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
                return

            with st.spinner("Processing..."):
                raw_documents = load_documents(uploaded_files)

                if not raw_documents:
                    st.error("No documents were loaded successfully. Please check file formats or content.")
                    return
                st.success(f"Loaded {len(raw_documents)} documents.")

                text_chunks = split_documents(raw_documents)
                st.success(f"Split into {len(text_chunks)} text chunks.")

                # DEBUG PRINT HERE: What is the llm_provider *right now*?
                #st.write(f"DEBUG: LLM Provider for embeddings: {st.session_state.llm_provider}")

                vectorstore = create_vector_store(text_chunks, st.session_state.llm_provider)
                if vectorstore is None:
                    st.error("Failed to create vector store. Please check your API key or Ollama setup.")
                    return
                st.success("Vector store created.")

                # DEBUG PRINT HERE: What is the llm_provider *right before* chain creation?
                #st.write(f"DEBUG: LLM Provider for chat model: {st.session_state.llm_provider}")

                st.session_state.conversation = get_conversation_chain(
                    vectorstore,
                    st.session_state.llm_provider,
                    st.session_state.ollama_model
                )
                if st.session_state.conversation is None:
                    st.error("Failed to initialize conversation chain. Please check LLM provider settings.")
                    return
                st.success("RAG pipeline ready!")
                st.session_state.chat_history = []


    st.write("Ask a question about your documents:")
    user_question = st.text_input("Your question:")

    if user_question:
        if st.session_state.conversation:
            with st.spinner("Generating response...."):
                try:
                    response = st.session_state.conversation({'question': user_question})
                    # The actual answer from the LLM is usually the content of the last message in chat_history
                    answer = response['chat_history'][-1].content
                    
                    # Extract source documents (if return_source_documents=True was set)
                    source_documents = response.get('source_documents', [])

                    # Store the interaction in chat_history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer,
                        "sources": source_documents
                    })
                except Exception as e:
                    st.error(f"Error generating response: {e}. Please ensure your LLM is running and accessible.")
        else:
            st.warning("Please process documents first to enable the RAG assistant.")
    # This loop iterates through the stored chat_history and displays each turn
    if st.session_state.chat_history:
        st.subheader("Chat History")
        # Display in reverse order to show most recent at the top
        for i, chat in enumerate(reversed(st.session_state.chat_history)): # Iterate in reverse
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Assistant:** {chat['answer']}")
            
            # Display Source Tracking
            # In main() function, inside the chat history loop:
            if chat['sources']:
                with st.expander(f"Sources for this response"):
                    for j, source_doc in enumerate(chat['sources']):
                        source_name = source_doc.metadata.get('source', 'Unknown Source')
                        page_number = source_doc.metadata.get('page', 'N/A')
                        st.markdown(f"**Source {j+1}:** {source_name} (Page: {page_number})")
                        st.text_area( # Change from st.code to st.text_area
                            f"Content from Source {j+1}",
                            source_doc.page_content, # Display full content
                            height=150, # Set a fixed height
                            key=f"source_content_{i}_{j}" # Unique key for each text_area
                        )
            st.markdown("---") # Separator between chat turns
if __name__ == "__main__":
    main()