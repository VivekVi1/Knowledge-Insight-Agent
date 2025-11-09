import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
import pickle
import boto3
from dotenv import load_dotenv

from sklearn.mixture import GaussianMixture  
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

# ---
# Import the new prompt loader
# ---
from decoupled.utils import load_prompt


# ---
# 1. BEDROCK & CLIENT INITIALIZATIONS
# ---

load_dotenv()
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.getenv("BEDROCK_API", "")

try:
    boto_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    llm = ChatBedrockConverse(
        model="amazon.nova-pro-v1:0",
        client=boto_client,
        max_tokens=1024,
        temperature=0.0
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=boto_client
    )
    
    print("RaptorRAG: Bedrock LLM and Embeddings initialized successfully.")

except Exception as e:
    print(f"Error initializing Boto3 client, LLM, or Embeddings in RaptorRAG: {e}")
    exit()

# ---
# Load prompts ONCE at the module level
# ---
try:
    PROMPT_CLUSTER_SUMMARY = load_prompt("cluster_summary.txt")
    PROMPT_CONTEXT_COMPRESSION = load_prompt("context_compression.txt")
    PROMPT_FINAL_ANSWER = load_prompt("final_answer.txt")
except Exception as e:
    exit(1) # Exit if prompt loading fails


# ---
# 2. RAPTOR RAG SYSTEM CLASS
# ---

class RAPTORSystem:
    """
    RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) System
    Builds hierarchical document representations for improved RAG.
    """
    
    def __init__(self):
        # Use the models initialized at the module level
        self.embeddings = embeddings
        self.llm = llm
        self.tree_levels = {}
        self.vectorstore = None
        self.compression_retriever = None
        
    def create_embeddings_batch(self, texts_list: List[str]):
        """
        Convert texts to embeddings - batched for efficiency.
        """
        print(f" Creating embeddings for {len(texts_list)} texts...")
        
        # Ensure all items are strings
        str_texts = [str(text) for text in texts_list]
        
        # Generate embeddings in a single batch call
        all_embeddings = self.embeddings.embed_documents(str_texts)
        
        print(f" Created {len(all_embeddings)} embeddings")
        return all_embeddings 

    def cluster_with_gmm(self, embeddings_array, n_clusters):
        """Cluster embeddings using Gaussian Mixture Models."""
        # Ensure we don't try to create more clusters than data points
        n_clusters = min(n_clusters, len(embeddings_array))
        if n_clusters < 2:
            return np.zeros(len(embeddings_array))  # Single cluster

        print(f" Clustering into {n_clusters} groups...")
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        return gm.fit_predict(embeddings_array)

    def generate_summaries_for_clusters(self, texts, cluster_labels, level_number):
        """Generate summaries for each cluster."""
        print(f" Generating summaries for Level {level_number}...")
        summaries = []
        metadata_list = []
        
        # ---
        # Use the loaded prompt
        # ---
        prompt = ChatPromptTemplate.from_template(PROMPT_CLUSTER_SUMMARY)
        chain = prompt | self.llm

        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            # Get texts in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]

            # Combine and summarize
            combined_text = "\n\n".join(cluster_texts)
            
            # FIX: Increased context limit from 4000 to 100000.
            truncated_text = combined_text[:100000]
            if len(combined_text) > 100000:
                print(f"   - Warning: Cluster {cluster_id} text truncated for summarization.")

            # Invoke the chain
            result = chain.invoke({"text": truncated_text}) 

            summary = result.content if hasattr(result, 'content') else str(result)
            summaries.append(summary)

            # Create metadata
            metadata = {
                "level": level_number,
                "cluster_id": int(cluster_id),
                "source_docs": len(cluster_texts),
                "id": f"summary_L{level_number}_C{cluster_id}"
            }
            metadata_list.append(metadata)

        return summaries, metadata_list

    def create_dataframe_for_level(self, texts, embeddings_list, metadata_list, cluster_labels=None):
        """Create organized DataFrame for each tree level."""
        df = pd.DataFrame({
            'text': texts,
            'embedding': embeddings_list,
            'metadata': metadata_list
        })
        if cluster_labels is not None:
            df['cluster'] = cluster_labels
        return df
    
    def build_index(self, texts: List[str]):
        """
        Build the complete RAPTOR hierarchical index from input texts.
        """
        print("\n" + "="*80)
        print("BUILDING RAPTOR HIERARCHICAL INDEX")
        print("="*80)
        
        # Build Level 0 - Original documents
        print("\n Building Level 0 (Original Documents)...")

        # Create embeddings for all original texts
        embeddings_list = self.create_embeddings_batch(texts)

        # Create metadata for original documents
        level_0_metadata = []
        for i, text in enumerate(texts):
            metadata = {
                "level": 0,
                "origin": "original",
                "id": f"doc_{i}",
                "text_length": len(text)
            }
            level_0_metadata.append(metadata)

        # Create Level 0 DataFrame
        level_0_df = self.create_dataframe_for_level(
            texts, 
            embeddings_list,  # Use the list directly
            level_0_metadata
        )

        print(f" Level 0 created with {len(level_0_df)} documents")
        print(f" Sample metadata: {level_0_df['metadata'].iloc[0]}")

        # Store in our tree structure
        self.tree_levels[0] = level_0_df

        # Convert to numpy array for clustering (we need numpy array for GMM)
        embeddings_array = np.array(embeddings_list)
        
        if len(embeddings_array) == 0:
            print("No text was provided to build_index. Aborting.")
            return

        # Cluster Level 0 documents
        optimal_clusters = min(8, max(2, len(texts) // 3))  # Smart cluster sizing
        cluster_labels = self.cluster_with_gmm(embeddings_array, optimal_clusters)

        # Add clusters to Level 0
        level_0_df['cluster'] = cluster_labels

        print(" Clustering Results:")
        for cluster_id in np.unique(cluster_labels):
            count = sum(cluster_labels == cluster_id)
            print(f"   Cluster {cluster_id}: {count} documents")

        # Generate Level 1 summaries
        level_1_summaries, level_1_metadata = self.generate_summaries_for_clusters(
            texts, cluster_labels, level_number=1
        )

        # Create embeddings for summaries
        level_1_embeddings = self.create_embeddings_batch(level_1_summaries)

        # Create Level 1 DataFrame
        level_1_df = self.create_dataframe_for_level(
            level_1_summaries,
            level_1_embeddings,
            level_1_metadata
        )

        # Add to tree
        self.tree_levels[1] = level_1_df

        print(f" Level 1 created with {len(level_1_df)} summaries")
        if level_1_summaries:
            print(f" Sample summary: {level_1_summaries[0][:150]}...")

        # Continue building levels until we have few enough summaries
        current_level = 1
        current_texts = level_1_summaries
        current_embeddings = level_1_embeddings

        # Convert to numpy array for clustering
        current_embeddings_array = np.array(current_embeddings)

        while len(current_texts) > 3 and current_level < 4:  # Prevent infinite loops
            print(f"\n Building Level {current_level + 1}...")
            
            if len(current_embeddings_array) == 0:
                 print(f" No embeddings for Level {current_level}, stopping.")
                 break

            # Cluster current level
            n_clusters = min(6, max(2, len(current_texts) // 2))
            cluster_labels = self.cluster_with_gmm(current_embeddings_array, n_clusters)

            # Add clusters to current level DataFrame (this is safe now)
            if current_level in self.tree_levels:
                self.tree_levels[current_level]['cluster'] = cluster_labels
                print(f" Added cluster info to Level {current_level}")

            # Generate summaries for next level
            next_level = current_level + 1
            next_summaries, next_metadata = self.generate_summaries_for_clusters(
                current_texts, cluster_labels, next_level
            )

            # Check if we got any summaries
            if not next_summaries:
                print(f" No summaries generated for Level {next_level}, stopping.")
                break

            # Create embeddings for next level
            next_embeddings = self.create_embeddings_batch(next_summaries)

            # Create next level DataFrame
            next_df = self.create_dataframe_for_level(
                next_summaries,
                next_embeddings,
                next_metadata
            )

            # Add to tree
            self.tree_levels[next_level] = next_df

            print(f" Level {next_level} created with {len(next_df)} summaries")

            # Prepare for next iteration
            current_level = next_level
            current_texts = next_summaries
            current_embeddings = next_embeddings
            current_embeddings_array = np.array(current_embeddings)

        print(f"\n Final tree structure:")
        for level, df in self.tree_levels.items():
            content_type = "original documents" if level == 0 else f"level-{level} summaries"
            print(f"   Level {level}: {len(df)} {content_type}")

        # Build unified FAISS vectorstore from all tree levels
        print(" Building unified FAISS vectorstore...")

        all_documents = []
        level_distribution = {}

        # Collect documents from all levels
        for level, df in self.tree_levels.items():
            print(f" Adding Level {level} to vectorstore...")

            for _, row in df.iterrows():
                doc = Document(
                    page_content=str(row['text']),
                    metadata=row['metadata']
                )
                all_documents.append(doc)

            level_distribution[level] = len(df)
            
        if not all_documents:
            print("No documents found in any tree level. Cannot build vectorstore.")
            return

        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_documents(all_documents, self.embeddings)

        print(f" Vectorstore created with {len(all_documents)} total documents!")
        print(f" Distribution across levels: {level_distribution}")

        # Create contextual compression retriever
        print(" Creating contextual compression retriever...")

        # Base retriever
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        # ---
        # Use the loaded prompt
        # ---
        compression_prompt = ChatPromptTemplate.from_template(PROMPT_CONTEXT_COMPRESSION)

        # Create the extractor
        extractor = LLMChainExtractor.from_llm(self.llm, prompt=compression_prompt)

        # Create compression retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=extractor,
            base_retriever=base_retriever
        )

        print(" Smart retriever created!")
        print("\n" + "="*80)
        print("RAPTOR INDEX BUILD COMPLETE")
        print("="*80)
    
    def query(self, query: str, show_sources: bool = True) -> Dict[str, Any]:
        """Test RAPTOR with a query and show detailed results."""
        if not self.compression_retriever:
            print("Error: RAPTOR index not built. Call build_index() first.")
            return None
            
        print(f"\n Testing query: '{query}'")
        print("=" * 60)

        # Retrieve documents
        retrieved_docs = self.compression_retriever.invoke(query)

        if show_sources:
            print(f"📄 Retrieved {len(retrieved_docs)} relevant documents:")
            level_counts = {}

            for i, doc in enumerate(retrieved_docs):
                level = doc.metadata.get('level', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1

                print(f"\n   {i+1}. Level {level} | {doc.page_content[:120]}...")

            print(f"\n Level distribution: {level_counts}")

        # Generate answer
        if retrieved_docs:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # ---
            # Use the loaded prompt
            # ---
            answer_prompt = ChatPromptTemplate.from_template(PROMPT_FINAL_ANSWER)

            chain = answer_prompt | self.llm
            result = chain.invoke({"context": context, "question": query})
            answer = result.content if hasattr(result, 'content') else str(result)

            print(f"\n RAPTOR Answer:")
            print("-" * 40)
            print(answer)

            return {
                "answer": answer,
                "sources": len(retrieved_docs),
                "levels_used": list(level_counts.keys()) if show_sources else []
            }
        else:
            print(" No relevant documents found.")
            return None
    
    def save_index(self, path: str):
        """Save the RAPTOR index to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save tree levels
        with open(os.path.join(path, "tree_levels.pkl"), 'wb') as f:
            pickle.dump(self.tree_levels, f)
        
        # Save vectorstore
        if self.vectorstore:
            self.vectorstore.save_local(os.path.join(path, "vectorstore"))
        
        print(f"✓ Index saved to {path}")
    
    def load_index(self, path: str):
        """Load a previously saved RAPTOR index."""
        # Load tree levels
        with open(os.path.join(path, "tree_levels.pkl"), 'rb') as f:
            self.tree_levels = pickle.load(f)
        
        # Load vectorstore
        self.vectorstore = FAISS.load_local(
            os.path.join(path, "vectorstore"),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Recreate compression retriever
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
        
        # ---
        # Use the loaded prompt
        # ---
        compression_prompt = ChatPromptTemplate.from_template(PROMPT_CONTEXT_COMPRESSION)
        
        extractor = LLMChainExtractor.from_llm(self.llm, prompt=compression_prompt)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=extractor,
            base_retriever=base_retriever
        )
        
        print(f"✓ Index loaded from {path}")