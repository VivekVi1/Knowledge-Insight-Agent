1. Overview
-----------

This application is a highly advanced, end-to-end RAG (Retrieval-Augmented Generation) system designed to ingest, understand, and query complex, multimodal documents (PDFs, DOCX, PPTX).

Its core innovation lies in two areas:
1.  Multimodal Processing: It doesn't just extract text. It also extracts tables and images, and critically, it uses a multimodal AI (Amazon Bedrock Nova) to generate rich, textual descriptions of every image.
2.  Advanced Indexing (RAPTOR): Instead of a simple "flat" vector store, it implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) technique. This builds a hierarchical, tree-structured index of document summaries, allowing for multi-resolution retrieval that can find both granular details and high-level concepts.

The result is a powerful query engine that can reason over the *entire* content of a document—including its text, tables, and visual elements—to provide comprehensive answers.

2. Key Features
---------------

* Multimodal Ingestion: Natively supports PDF (.pdf), Microsoft Word (.docx), Microsoft PowerPoint (.pptx), and plain text (.txt) files.
* Comprehensive Content Extraction:
    * Text: Extracts all text and splits it into semantic chunks.
    * Tables: Identifies and extracts tables from PDFs into a string format.
    * Images: Extracts all embedded images from PDFs, DOCX, and PPTX files.
* AI-Powered Image Understanding: Each extracted image is passed to the amazon.nova-pro-v1:0 multimodal model to generate a detailed textual description, making visual data searchable.
* Hierarchical RAPTOR Indexing:
    * Level 0: The base of the tree, containing the original text chunks, table strings, and AI-generated image descriptions.
    * Higher Levels (L1, L2, ...): Recursively clusters the nodes from the level below and generates AI-powered summaries for each cluster. This creates a "pyramid" of information, from fine-grained to highly abstract.
* Unified Vector Store: All nodes from all levels of the RAPTOR tree (both raw chunks and generated summaries) are embedded using amazon.titan-embed-text-v2:0 and stored in a single, unified FAISS vector store.
* Smart Retrieval: Uses a ContextualCompressionRetriever to intelligently fetch the most relevant documents from the vector store and then uses an LLM to "compress" them, ensuring only the most salient information is passed to the final answer-generation step.
* Interactive CLI: Provides a simple, user-friendly command-line interface to process new documents and ask questions.
* Persistent Index: Automatically saves the complete RAPTOR index (tree structure and FAISS vector store) to disk, allowing you to load it later without reprocessing.

3. System Architecture & Pipeline
--------------------------------

The application runs in a sequential pipeline orchestrated by main.py.

1.  Start (User Input)
    * main.py starts and prompts the user to enter one or more file paths, separated by commas.

2.  Step 1: Document Processing (document_processor.py)
    * For each file, main.py calls load_and_process_document.
    * This function first creates an output_<filename>/ directory to store all extracted assets.
    * It uses the appropriate loader (PyMuPDFLoader, Docx2txtLoader, etc.) to extract raw text, which is then chunked.
    * It calls specialized functions (extract_pdf_elements, extract_docx_images, etc.) to find and save images and tables.
    * Crucially: For every image, it calls generate_image_description. This function converts the image to base64, sends it to the Bedrock nova-pro model (using the image_description.txt prompt), and receives a textual description back.
    * All extracted content (text chunks, table strings, image descriptions, metadata) is saved to disk and returned as a unified list.

3.  Step 2: Index Building (raptor_rag.py)
    * main.py initializes the RAPTORSystem.
    * It passes the complete list of text items (from all documents) to raptor.build_index().
    * Level 0: RAPTORSystem embeds all these original texts and stores them as Level 0 of its tree.
    * Clustering: It uses GaussianMixture to cluster the Level 0 embeddings.
    * Level 1: It generates summaries for each cluster (using the cluster_summary.txt prompt) and embeds them. These summaries become Level 1.
    * Recursion: This process repeats—clustering Level 1 to create Level 2, and so on—until the number of nodes is too small to cluster further.
    * Unified FAISS Store: All embeddings (from L0, L1, L2...) are loaded into a single FAISS vector store.
    * Retriever Setup: A ContextualCompressionRetriever is created, using the context_compression.txt prompt to filter results.

4.  Step 3: Interactive Query (main.py & raptor_rag.py)
    * main.py enters an interactive query loop.
    * The user enters a query.
    * raptor.query() is called.
    * The ContextualCompressionRetriever searches the *entire* FAISS store, pulling back the most relevant chunks (which could be L0 details, L1 summaries, etc.).
    * These retrieved contexts are combined and passed to the Bedrock nova-pro LLM along with the final_answer.txt prompt.
    * The final, synthesized answer is printed to the user.

5.  Step 4: Save Index (main.py)
    * When the user quits, main.py calls raptor.save_index().
    * This saves the tree_levels (a dictionary of pandas DataFrames) as tree_levels.pkl and the FAISS index to the raptor_output/raptor_index/ directory.

4. Module Breakdown
-------------------

main.py
* Purpose: The main entry point and orchestrator.
* Key Functions:
    * main(): Controls the entire application flow from start to finish: prompts for files, calls processing, calls indexing, runs the query loop, and calls the save function.

decoupled/document_processor.py
* Purpose: Handles all file ingestion, parsing, and multimodal content extraction.
* Key Functions:
    * load_and_process_document(): The main entry point for this module. Selects the correct loader and extractor based on the file extension.
    * generate_image_description(): The core multimodal function. Takes image bytes, converts them to base64 PNG, and uses the Bedrock nova-pro model to generate a description.
    * extract_pdf_elements(): Uses pymupdf (Fitz) to get images and tabula-py to get tables from PDFs.
    * extract_docx_images(): Uses python-docx to find and extract images from Word documents.
    * extract_pptx_images(): Uses python-pptx to find and extract images from PowerPoint slides.

decoupled/raptor_rag.py
* Purpose: Implements the RAPTOR indexing logic and the RAG query pipeline.
* Key Class: RAPTORSystem
    * __init__(): Initializes the Bedrock LLM (nova-pro) and embedding model (titan-embed-text-v2:0).
    * build_index(): The main indexing logic. Orchestrates the recursive clustering (cluster_with_gmm), summarization (generate_summaries_for_clusters), and embedding creation. It finishes by building the unified FAISS store and the compression retriever.
    * query(): Takes a user's string query, retrieves relevant context using the ContextualCompressionRetriever, and generates a final answer using the LLM.
    * save_index() / load_index(): Handles saving the tree state (as a pickle file) and the FAISS index to disk, and loading them back.

decoupled/utils.py
* Purpose: A simple utility module.
* Key Functions:
    * load_prompt(): A helper function to load prompt text from the prompts/ directory at the project root.

5. Setup & Installation
-----------------------

Follow these steps to set up and run the application.

1. Directory Structure
Ensure your project has the following structure:

/your-project-root
├── .env
├── main.py
├── requirements.txt
├── decoupled/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── raptor_rag.py
│   └── utils.py
└── prompts/
    ├── cluster_summary.txt
    ├── context_compression.txt
    ├── final_answer.txt
    └── image_description.txt

2. AWS Bedrock Configuration
This application requires access to AWS Bedrock.

1.  AWS Account: You must have an AWS account with access to Bedrock enabled.
2.  Model Access: You must request and receive access to the following models in the us-east-1 region:
    * amazon.nova-pro-v1:0 (for summarization, image description, and final answers)
    * amazon.titan-embed-text-v2:0 (for generating embeddings)
3.  AWS Credentials: Configure your AWS credentials so that boto3 can find them. The recommended way is by installing the AWS CLI and running:
    
    aws configure
    
    Enter your AWS Access Key ID, Secret Access Key, and set your default region to us-east-1.

3. Python Environment & Dependencies
1.  Python: This project is compatible with Python 3.8 or newer.
2.  .env File: The code attempts to load a BEDROCK_API key from a .env file, although standard boto3 authentication (from Step 2) is typically sufficient. To be safe, create a .env file in the project root:
    
    BEDROCK_API="dummy_key"

3.  requirements.txt: Create a requirements.txt file with the following content:
    
    numpy
    pandas
    scikit-learn
    langchain
    langchain-aws
    langchain-community
    boto3
    python-dotenv
    faiss-cpu
    pymupdf
    tabula-py
    python-docx
    python-pptx
    pillow

4.  Install Dependencies:
    
    pip install -r requirements.txt
    
    Note on tabula-py: tabula-py is a wrapper for a Java library. You must have Java installed on your system for it to work.

4. Prompts
Create the prompts/ directory and add the four required .txt files. The content of these files will define the AI's behavior.

* prompts/image_description.txt: (e.g., "Describe this image in detail. What is it? What components does it have? What information is it trying to convey?")
* prompts/cluster_summary.txt: (e.g., "You will be given a set of related text chunks. Summarize them into a single, cohesive paragraph that captures the main ideas. Text: {text}")
* prompts/context_compression.txt: (e.g., "Given the following context and question, extract only the parts of the context that are relevant to answering the question. Context: {context} Question: {question}")
* prompts/final_answer.txt: (e.g., "Using the provided context, answer the following question. Context: {context} Question: {question} Answer:")

6. How to Run
-------------

1.  Ensure all setup steps (dependencies, AWS credentials, .env, and prompts/ folder) are complete.
2.  Open your terminal and navigate to the project's root directory.
3.  Run the main application:
    
    python main.py
    
4.  Step 1: The application will ask for file paths.
    
    📁 Enter file paths: documents/my_report.pdf, slides/presentation.pptx
    
5.  Step 2: Wait while the system processes the documents and builds the RAPTOR index. You will see detailed logs in your terminal.
6.  Step 3: Once the index is built, the query interface will appear.
    
    🔍 Enter your query: What was the main conclusion from the diagram on slide 10?
    
7.  The system will retrieve context (potentially from the AI-generated description of the diagram) and generate an answer.
8.  Ask more questions, or type 'quit' or 'exit' to end the session.
9.  Step 4: Upon quitting, the system will save the index to the raptor_output/ directory for future use.

7. Application Output
---------------------

This application generates two types of output:

1.  Per-File Processing Output (e.g., output_my_report/):
    * images/: Contains all raw image files extracted from the document.
    * text/: Contains .txt files for each text chunk and each AI-generated image description.
    * other/tables/: Contains .txt files of the extracted table data.
    * other/metadata/: Contains a .txt file with the document's metadata (author, title, etc.).

2.  Global Index Output (raptor_output/):
    * raptor_index/: A folder containing the saved FAISS vector store (index.faiss, index.pkl).
    * raptor_index/tree_levels.pkl: The pickled Python object (a dictionary of DataFrames) representing the RAPTOR hierarchical tree.
    * processing_summary.txt: A high-level summary of the items processed and the final RAPTOR tree structure.