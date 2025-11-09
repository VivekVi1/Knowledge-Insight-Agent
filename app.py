import streamlit as st
import os
import sys
import tempfile
import warnings
import json


try:
    from decoupled.raptor_rag import RAPTORSystem, process_files
except ImportError:
    st.error(
        "Could not import RAPTOR modules. "
        "Please ensure 'app.py' is in the same directory as the 'decoupled' folder."
    )
    sys.exit(1)

# ---
# Page Configuration
# ---
st.set_page_config(
    page_title="Multimodal RAPTOR RAG",
    page_icon="📚",
    layout="wide"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# ---
# Helper Function to save uploaded files to a temp directory
# ---
def save_uploaded_files(uploaded_files):
    """Saves uploaded files to a temporary directory and returns their paths."""
    # Create a persistent temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_paths = []
    
    for uploaded_file in uploaded_files:
        # Get file bytes
        bytes_data = uploaded_file.getvalue()
        # Create a path in the temp directory
        temp_path = os.path.join(temp_dir.name, uploaded_file.name)
        # Write the file
        with open(temp_path, "wb") as f:
            f.write(bytes_data)
        temp_paths.append(temp_path)
        
    # Return the list of paths and the directory object
    # We must return temp_dir so it doesn't get garbage-collected
    return temp_paths, temp_dir

# ---
# Session State Initialization
# ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "raptor_system" not in st.session_state:
    # Initialize the RAPTORSystem class once and store it in state
    st.session_state.raptor_system = RAPTORSystem()
    
if "index_built" not in st.session_state:
    st.session_state.index_built = False
    
if "temp_dir" not in st.session_state: 
    # This will hold the TemporaryDirectory object to prevent it
    # from being deleted while the app is running
    st.session_state.temp_dir = None

# ---
# Main UI
# ---
st.title("📚 Multimodal RAPTOR RAG System")
st.markdown("""
This application uses the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval)
technique to index and query your documents. It understands text, tables, images, audio and video.
""")

# ---
# Sidebar for Setup (Upload or Load)
# ---
with st.sidebar:
    st.header("Setup")

    # --- Option 1: Process New Documents ---
    with st.expander("1. Process New Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, DOCX, PPTX, TXT, MP3, WAV, M4A, MP4, MOV, AVI, MKV)",
            type=["pdf", "docx", "pptx", "txt", "mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True
        )
        process_button = st.button("Build New RAPTOR Index")

    # --- Option 2: Load Existing Index ---
    with st.expander("2. Load Existing Index"):
        index_path = st.text_input("Index Path", "raptor_output/raptor_index")
        load_button = st.button("Load Index")

    # --- Logic for Processing ---
    if process_button and uploaded_files:
        if st.session_state.index_built:
            st.warning("Index is already active. Please refresh to build a new one.")
        else:
            # 1. Save files to a temporary location
            with st.spinner("Saving uploaded files..."):
                temp_paths, temp_dir_obj = save_uploaded_files(uploaded_files)
                # Store the temp_dir object in session state to keep it alive
                st.session_state.temp_dir = temp_dir_obj
                st.success(f"Saved {len(temp_paths)} files to a temporary location.")

            # 2. Process Documents
            with st.status("Processing documents...", expand=True) as status:
                all_texts = process_files(temp_paths)
                status.update(label="Document processing complete!", state="complete")

            # 3. Build RAPTOR Index
            with st.status("Building RAPTOR Index (this may take a while)...", expand=True) as status:
                status.write(f"Preparing {len(all_texts)} text segments for RAPTOR indexing...")
                
                # Get the RAPTOR system from session state
                raptor_system = st.session_state.raptor_system
                
                # Build the hierarchical index
                raptor_system.build_index(all_texts)
                
                # Update session state
                st.session_state.index_built = True
                
                status.update(label="RAPTOR Index built successfully!", state="complete")

            # 4. Save Index
            with st.spinner("Saving index for future use..."):
                output_dir = "raptor_output"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "raptor_index")
                
                raptor_system.save_index(save_path)
                st.success(f"Index saved to: {save_path}")

    elif process_button and not uploaded_files:
        st.error("Please upload at least one file.")
        
    # --- Logic for Loading ---
    if load_button:
        if st.session_state.index_built:
            st.warning("Index is already active.")
        else:
            if not os.path.exists(index_path):
                st.error(f"Error: Index path not found: {index_path}")
            else:
                with st.spinner(f"Loading index from {index_path}..."):
                    try:
                        # Get the RAPTOR system from state
                        raptor_system = st.session_state.raptor_system
                        # Load the index
                        raptor_system.load_index(index_path)
                        # Update state
                        st.session_state.index_built = True
                        st.success("Index loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading index: {e}")
                        st.error("Please ensure the 'vectorstore' folder and 'tree_levels.pkl' are present.")

# ---
# Main Chat Interface
# ---
st.header("2. Query Your Documents")

if not st.session_state.index_built:
    st.info("Please process new documents or load an existing index to start querying.")
else:
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show retrieval details if they exist
            if "details" in message:
                with st.expander("See retrieval details"):
                    st.json(message["details"])

    # Chat input box
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Get the RAPTOR system from state
                    raptor_system = st.session_state.raptor_system
                    
                    # Process query
                    result = raptor_system.query(prompt, show_sources=True)
                    
                    if result:
                        answer = result.get('answer', 'No answer found.')
                        details = {
                            "sources_retrieved": result.get('sources', 0),
                            "levels_used": result.get('levels_used', [])
                        }
                        message_placeholder.markdown(answer)
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "details": details
                        })
                    else:
                        message_placeholder.markdown("No relevant information found for your query.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "No relevant information found for your query."
                        })
                        
                except Exception as e:
                    error_msg = f"Error processing query: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"An error occurred: {e}"
                    })