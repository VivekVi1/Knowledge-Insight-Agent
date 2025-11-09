import os
import sys
import warnings

# Import the main function and class from your decoupled package
from decoupled.document_processor import load_and_process_document
from decoupled.raptor_rag import RAPTORSystem

# Suppress warnings
warnings.filterwarnings('ignore')

def main():

    print("\n" + "="*80)
    print("RAPTOR RAG SYSTEM - COMPLETE PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Document Processing
    # ========================================================================
    print("\n[STEP 1] DOCUMENT PROCESSING")
    print("-" * 80)
    
    # ---
    # Ask the user for file paths
    # ---
    print("Please enter the file paths you want to process.")
    print("You can enter multiple files by separating them with a comma (,)")
    
    files_to_process = []
    while not files_to_process:
        try:
            # Ask the user for input
            raw_input = input("\n📁 Enter file paths: ").strip()
            
            if not raw_input:
                print("⚠ Please enter at least one file path.")
                continue
                
            # Split the input string by commas
            files_to_process = [path.strip() for path in raw_input.split(',')]
            
            # Remove any empty strings (e.g., from a trailing comma)
            files_to_process = [path for path in files_to_process if path]
            
            if not files_to_process:
                print("⚠ No valid paths entered. Please try again.")

        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Process cancelled by user. Exiting.")
            sys.exit(0)
    # ---
    # End of input section
    # ---
    
    all_processed_items = []
    
    print("\nStarting processing...")
    for filepath in files_to_process:
        if os.path.exists(filepath):
            print(f"\nProcessing: {filepath}")
            items = load_and_process_document(filepath)
            all_processed_items.extend(items)
            print(f"✓ Extracted {len(items)} items from {filepath}")
        else:
            print(f"\n⚠ Warning: File '{filepath}' not found. Skipping.")
    
    if not all_processed_items:
        print("\n❌ No items were processed. Exiting.")
        return
    
    # Display processing summary
    print("\n" + "="*80)
    print("DOCUMENT PROCESSING SUMMARY")
    print("="*80)
    print(f"Total items processed: {len(all_processed_items)}")
    print(f"   - Text chunks: {sum(1 for i in all_processed_items if i['type'] == 'text')}")
    print(f"   - Images: {sum(1 for i in all_processed_items if i['type'] == 'image_description')}")
    print(f"   - Tables: {sum(1 for i in all_processed_items if i['type'] == 'table')}")
    print(f"   - Metadata: {sum(1 for i in all_processed_items if i['type'] == 'metadata')}")
    
    # ========================================================================
    # STEP 2: Build RAPTOR Index
    # ========================================================================
    print("\n[STEP 2] BUILDING RAPTOR HIERARCHICAL INDEX")
    print("-" * 80)
    
    # Combine all text content for RAPTOR
    all_texts = []
    for item in all_processed_items:
        if item['type'] in ['text', 'image_description', 'table', 'metadata']:
            # Add type prefix for context
            text_with_context = f"[{item['type'].upper()}] {item['text']}"
            all_texts.append(text_with_context)
    
    print(f"Preparing {len(all_texts)} text segments for RAPTOR indexing...")
    
    # Initialize RAPTOR system
    raptor = RAPTORSystem()
    
    # Build the hierarchical index
    raptor.build_index(all_texts)
    
    # ========================================================================
    # STEP 3: Interactive Query Interface
    # ========================================================================
    print("\n[STEP 3] QUERY INTERFACE")
    print("-" * 80)
    
    if not raptor.compression_retriever:
        print("\n❌ RAPTOR index was not built successfully. Cannot start query interface.")
        return

    print("\nRAPTOR system is ready for queries!")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    # Pre-defined test queries (optional)
    test_queries = [
        "What is this document about?",
        "Summarize the key points",
        "What images or diagrams are described?",
    ]
    
    print("Suggested queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"   {i}. {query}")
    
    print("\n" + "-"*80)
    
    # Interactive query loop
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠ Please enter a valid query.")
                continue
            
            # Process query through RAPTOR
            result = raptor.query(query, show_sources=True)
            
            if result:
                print("\n" + "="*80)
                print("QUERY RESULTS")
                print("="*80)
                print(f"\n📊 Retrieved {result['sources']} sources from levels {result['levels_used']}")
                print(f"\n💡 Answer:\n{result['answer']}")
                print("\n" + "="*80)
            else:
                print("\n⚠ No relevant information found for your query.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            continue
    
    # ========================================================================
    # STEP 4: Save Results (Optional)
    # ========================================================================
    print("\n[STEP 4] SAVING RESULTS")
    print("-" * 80)
    
    # Save RAPTOR index for future use
    output_dir = "raptor_output"
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "raptor_index")
    raptor.save_index(index_path)
    print(f"✓ RAPTOR index saved to: {index_path}")
    
    # Save processing summary
    summary_path = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("DOCUMENT PROCESSING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total items processed: {len(all_processed_items)}\n")
        f.write(f"   - Text chunks: {sum(1 for i in all_processed_items if i['type'] == 'text')}\n")
        f.write(f"   - Images: {sum(1 for i in all_processed_items if i['type'] == 'image_description')}\n")
        f.write(f"   - Tables: {sum(1 for i in all_processed_items if i['type'] == 'table')}\n")
        f.write(f"   - Metadata: {sum(1 for i in all_processed_items if i['type'] == 'metadata')}\n\n")
        
        f.write("RAPTOR TREE STRUCTURE\n")
        f.write("="*80 + "\n\n")
        for level, df in raptor.tree_levels.items():
            content_type = "original documents" if level == 0 else f"level-{level} summaries"
            f.write(f"Level {level}: {len(df)} {content_type}\n")
    
    print(f"✓ Processing summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("You can reload the RAPTOR index later using raptor.load_index()")


if __name__ == "__main__":
    main()