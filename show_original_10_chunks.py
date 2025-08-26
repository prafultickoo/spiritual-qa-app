#!/usr/bin/env python3
"""
Show the original 10 chunks (5 verses + 5 explanations) before merging.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from answer_generator import AnswerGenerator

load_dotenv()

def show_original_10_chunks():
    """Show the exact position of chunks in the original 10 intermediate results."""
    
    print("📋 ORIGINAL 10 CHUNKS (BEFORE MERGING)")
    print("=" * 100)
    
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    dual_source_retriever = generator.retriever.dual_source_retriever
    raw_result = dual_source_retriever.retrieve(query="What is Karma Yoga", k=5)
    
    print("📜 VERSES COLLECTION (5 chunks):")
    for i, verse_doc in enumerate(raw_result.clean_verses, 1):
        content = verse_doc.page_content
        print(f"\nVERSE #{i}:")
        print(f"Length: {len(content)} chars")
        print(f"Content: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        # Check for the target snippet
        if "yoga communities" in content:
            print(f"🎯 THIS IS THE 'yoga communities' CHUNK - VERSE #{i}")
    
    print("\n📝 EXPLANATIONS COLLECTION (5 chunks):")
    for i, explanation_doc in enumerate(raw_result.explanations, 1):
        content = explanation_doc.page_content
        print(f"\nEXPLANATION #{i}:")
        print(f"Length: {len(content)} chars")
        print(f"Content: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        # Check for the target snippet
        if "yoga communities" in content:
            print(f"🎯 THIS IS THE 'yoga communities' CHUNK - EXPLANATION #{i}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Total verses: {len(raw_result.clean_verses)}")
    print(f"Total explanations: {len(raw_result.explanations)}")
    print(f"Total intermediate chunks: {len(raw_result.clean_verses) + len(raw_result.explanations)}")
    
    # Find the exact position in the combined list
    target_snippet = "yoga communities"
    combined_position = 0
    
    # Search verses first
    for i, verse_doc in enumerate(raw_result.clean_verses, 1):
        combined_position += 1
        if target_snippet in verse_doc.page_content:
            print(f"\n🎯 EXACT POSITION: Chunk #{combined_position} in the combined list of 10")
            print(f"   Collection: VERSES")
            print(f"   Position in verses: #{i}")
            break
    else:
        # Search explanations if not found in verses
        for i, explanation_doc in enumerate(raw_result.explanations, 1):
            combined_position += 1
            if target_snippet in explanation_doc.page_content:
                print(f"\n🎯 EXACT POSITION: Chunk #{combined_position} in the combined list of 10")
                print(f"   Collection: EXPLANATIONS") 
                print(f"   Position in explanations: #{i}")
                break

if __name__ == "__main__":
    show_original_10_chunks()
