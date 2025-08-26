#!/usr/bin/env python3
"""
Debug the mismatch between shown 10 chunks and interleaving content.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from answer_generator import AnswerGenerator

load_dotenv()

def debug_chunk_mismatch():
    """Debug where the 'yoga communities' snippet is coming from."""
    
    print("🔍 DEBUGGING CHUNK MISMATCH")
    print("=" * 100)
    
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    dual_source_retriever = generator.retriever.dual_source_retriever
    raw_result = dual_source_retriever.retrieve(query="What is Karma Yoga", k=5)
    
    print("📜 RAW VERSES COLLECTION RESULTS:")
    for i, verse_doc in enumerate(raw_result.clean_verses, 1):
        content = verse_doc.page_content
        print(f"\nVERSE #{i}:")
        print(f"Length: {len(content)} chars")
        print(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")
        
        # Check if this matches the problematic snippet
        if "yoga communities" in content:
            print("🚨 FOUND THE 'yoga communities' SNIPPET HERE!")
            print(f"Full content: {content}")
    
    print("\n📝 RAW EXPLANATIONS COLLECTION RESULTS:")
    for i, explanation_doc in enumerate(raw_result.explanations, 1):
        content = explanation_doc.page_content
        print(f"\nEXPLANATION #{i}:")
        print(f"Length: {len(content)} chars") 
        print(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")
        
        # Check if this matches the problematic snippet
        if "yoga communities" in content:
            print("🚨 FOUND THE 'yoga communities' SNIPPET HERE!")
            print(f"Full content: {content}")
    
    print("\n🔍 SEARCHING ALL COLLECTIONS FOR THE SNIPPET:")
    target_snippet = "yoga communities, karma is often spoken of"
    
    # Check verses collection
    verse_found = False
    for i, verse_doc in enumerate(raw_result.clean_verses, 1):
        if target_snippet in verse_doc.page_content:
            print(f"✅ Found in VERSES collection - Verse #{i}")
            verse_found = True
            break
    
    if not verse_found:
        print("❌ NOT found in verses collection")
    
    # Check explanations collection
    explanation_found = False
    for i, explanation_doc in enumerate(raw_result.explanations, 1):
        if target_snippet in explanation_doc.page_content:
            print(f"✅ Found in EXPLANATIONS collection - Explanation #{i}")
            explanation_found = True
            break
    
    if not explanation_found:
        print("❌ NOT found in explanations collection")
    
    print(f"\n🤔 ANALYSIS:")
    if verse_found:
        print("The snippet IS in the verses collection")
    elif explanation_found:
        print("The snippet IS in the explanations collection")
    else:
        print("🚨 MYSTERY: The snippet is NOT in either collection!")
        print("This suggests a bug in the retrieval or display process")

if __name__ == "__main__":
    debug_chunk_mismatch()
