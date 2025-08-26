 #!/usr/bin/env python3
"""
Script to show full detailed chunks retrieved for "What is Karma Yoga" query.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator

load_dotenv()

def show_full_chunk(chunk, chunk_num):
    """Display a single chunk with full details."""
    print(f"\n{'='*100}")
    print(f"CHUNK #{chunk_num}")
    print(f"{'='*100}")
    
    # Metadata
    metadata = chunk.get('metadata', {})
    print(f"SOURCE: {metadata.get('source', 'Unknown')}")
    print(f"CHAPTER: {metadata.get('chapter', 'N/A')}")
    print(f"VERSE: {metadata.get('verse', 'N/A')}")
    print(f"RELEVANCE SCORE: {chunk.get('relevance_score', 'N/A'):.4f}")
    print(f"CONTENT LENGTH: {len(chunk.get('content', ''))} characters")
    
    # Full content
    content = chunk.get('content', '')
    print(f"\nFULL CONTENT:")
    print("-" * 100)
    print(content)
    print("-" * 100)

def main():
    query = "What is Karma Yoga"
    print(f"🕉️ RETRIEVING FULL CHUNKS FOR: '{query}'")
    print("=" * 100)
    
    # Initialize answer generator
    print("\n⏳ Initializing Answer Generator...")
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    # Get direct access to dual-source retriever to show intermediate results
    dual_source_retriever = generator.retriever.dual_source_retriever
    
    # Retrieve chunks with intermediate results
    print(f"\n🔍 Retrieving chunks for: '{query}'")
    start_time = time.time()
    
    # Get raw dual-source results (10 chunks: 5 verses + 5 explanations)
    print("\n📊 STEP 1: Fetching from both collections...")
    raw_result = dual_source_retriever.retrieve(query=query, k=5)
    
    # Get final processed chunks (merged and ranked to 5)
    retrieval_result = generator.retrieve_relevant_chunks(
        query=query,
        k=5,
        use_mmr=True
    )
    
    end_time = time.time()
    print(f"⏱️ Retrieval completed in {end_time - start_time:.2f} seconds")
    
    # Display intermediate results - this shows the "fetch 10, select 5" process
    print(f"\n📊 INTERMEDIATE RESULTS (10 CHUNKS FETCHED):")
    print(f"Verses from clean_verses collection: {len(raw_result.clean_verses)}")
    print(f"Explanations from spiritual_texts collection: {len(raw_result.explanations)}")
    
    # Show verses collection results - FULL CONTENT
    print(f"\n📜 VERSES COLLECTION RESULTS (5 chunks) - FULL CONTENT:")
    for i, verse_doc in enumerate(raw_result.clean_verses, 1):
        verse_content = verse_doc.page_content if hasattr(verse_doc, 'page_content') else str(verse_doc)
        print(f"\n{'='*120}")
        print(f"VERSE CHUNK #{i}")
        print(f"{'='*120}")
        print(f"Length: {len(verse_content)} characters")
        if hasattr(verse_doc, 'metadata'):
            metadata = verse_doc.metadata
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Chapter: {metadata.get('chapter', 'N/A')}")
            print(f"Verse: {metadata.get('verse', 'N/A')}")
        print(f"\nFULL CONTENT:")
        print("-" * 120)
        print(verse_content)
        print("-" * 120)
    
    # Show explanations collection results - FULL CONTENT
    print(f"\n📝 EXPLANATIONS COLLECTION RESULTS (5 chunks) - FULL CONTENT:")
    for i, explanation_doc in enumerate(raw_result.explanations, 1):
        explanation_content = explanation_doc.page_content if hasattr(explanation_doc, 'page_content') else str(explanation_doc)
        print(f"\n{'='*120}")
        print(f"EXPLANATION CHUNK #{i}")
        print(f"{'='*120}")
        print(f"Length: {len(explanation_content)} characters")
        if hasattr(explanation_doc, 'metadata'):
            metadata = explanation_doc.metadata
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Chapter: {metadata.get('chapter', 'N/A')}")
            print(f"Verse: {metadata.get('verse', 'N/A')}")
        print(f"\nFULL CONTENT:")
        print("-" * 120)
        print(explanation_content)
        print("-" * 120)
    
    # Display final results
    chunks = retrieval_result.get("chunks", [])
    query_info = retrieval_result.get("query_info", {})
    
    print(f"\n📊 FINAL MERGED RESULTS (5 CHUNKS SELECTED):")
    print(f"Query: '{query}'")
    print(f"Total chunks retrieved: {len(chunks)}")
    print(f"Dual source used: {query_info.get('dual_source_used', False)}")
    print(f"Query type: {query_info.get('query_type', 'Unknown')}")
    print(f"Verses found: {query_info.get('verses_found', 0)}")
    print(f"Explanations found: {query_info.get('explanations_found', 0)}")
    
    if not chunks:
        print("❌ No chunks retrieved!")
        return
    
    # Show all final chunks in detail
    print(f"\n📄 FINAL SELECTED CHUNKS (AFTER MERGING & RANKING):")
    for i, chunk in enumerate(chunks, 1):
        show_full_chunk(chunk, i)
    
    # Summary statistics
    print(f"\n📈 CHUNK ANALYSIS:")
    scores = [c.get('relevance_score', 0) for c in chunks if c.get('relevance_score')]
    if scores:
        print(f"Average relevance score: {sum(scores)/len(scores):.4f}")
        print(f"Highest relevance score: {max(scores):.4f}")
        print(f"Lowest relevance score: {min(scores):.4f}")
    
    total_chars = sum(len(c.get('content', '')) for c in chunks)
    print(f"Total content characters: {total_chars}")
    print(f"Average chunk size: {total_chars//len(chunks) if chunks else 0} characters")
    
    # Content type analysis
    verse_count = 0
    explanation_count = 0
    
    for chunk in chunks:
        content = chunk.get('content', '').lower()
        if any(indicator in content for indicator in ['sanskrit', 'कर्म', 'योग', 'श्लोक']):
            verse_count += 1
        else:
            explanation_count += 1
    
    print(f"\nContent Distribution:")
    print(f"Verse-related chunks: {verse_count}")
    print(f"Explanation chunks: {explanation_count}")

if __name__ == "__main__":
    main()
