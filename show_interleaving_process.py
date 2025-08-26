#!/usr/bin/env python3
"""
Show the actual interleaving process with real verse and explanation content.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from answer_generator import AnswerGenerator

load_dotenv()

def show_interleaving_with_actual_content():
    """Show the interleaving algorithm with actual verse and explanation content."""
    
    print("🔄 INTERLEAVING ALGORITHM - ACTUAL CONTENT")
    print("=" * 100)
    
    # Get the raw results
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    dual_source_retriever = generator.retriever.dual_source_retriever
    raw_result = dual_source_retriever.retrieve(query="What is Karma Yoga", k=5)
    
    # Extract the actual content
    verses = []
    explanations = []
    
    for verse_doc in raw_result.clean_verses:
        verses.append(verse_doc.page_content)
    
    for explanation_doc in raw_result.explanations:
        explanations.append(explanation_doc.page_content)
    
    print(f"📊 INPUT DATA:")
    print(f"Verses available: {len(verses)}")
    print(f"Explanations available: {len(explanations)}")
    
    # Simulate the interleaving algorithm
    print(f"\n🔄 INTERLEAVING PROCESS:")
    merged = []
    max_len = max(len(verses), len(explanations))
    
    for i in range(max_len):
        print(f"\n{'='*80}")
        print(f"ROUND {i+1}")
        print(f"{'='*80}")
        
        round_additions = []
        
        # Add verse if available
        if i < len(verses):
            merged.append(("VERSE", verses[i]))
            round_additions.append(f"VERSE #{i+1}")
            print(f"✅ ADD VERSE #{i+1}:")
            print(f"   Content: {verses[i][:150]}{'...' if len(verses[i]) > 150 else ''}")
            print(f"   Length: {len(verses[i])} characters")
        
        # Add explanation if available and not at limit
        if i < len(explanations) and len(merged) < 5:
            merged.append(("EXPLANATION", explanations[i]))
            round_additions.append(f"EXPLANATION #{i+1}")
            print(f"✅ ADD EXPLANATION #{i+1}:")
            print(f"   Content: {explanations[i][:150]}{'...' if len(explanations[i]) > 150 else ''}")
            print(f"   Length: {len(explanations[i])} characters")
        
        # Check if we've reached the limit
        print(f"\n📊 ROUND {i+1} SUMMARY:")
        print(f"   Added: {', '.join(round_additions)}")
        print(f"   Total chunks so far: {len(merged)}")
        
        if len(merged) >= 5:
            print(f"   🛑 STOPPING - Reached target of 5 chunks")
            break
    
    # Show final merged result
    print(f"\n🎯 FINAL INTERLEAVED RESULT:")
    print(f"Total chunks: {len(merged)}")
    
    for i, (chunk_type, content) in enumerate(merged, 1):
        print(f"\n--- CHUNK {i} ({chunk_type}) ---")
        print(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")
        print(f"Length: {len(content)} characters")
    
    # Analyze the distribution
    verse_count = sum(1 for chunk_type, _ in merged if chunk_type == "VERSE")
    explanation_count = sum(1 for chunk_type, _ in merged if chunk_type == "EXPLANATION")
    
    print(f"\n📈 FINAL DISTRIBUTION:")
    print(f"Verses: {verse_count}/5")
    print(f"Explanations: {explanation_count}/5")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"• Interleaving ensures balanced content types")
    print(f"• Algorithm stops exactly at k=5 limit")
    print(f"• Highest scoring chunks from each collection get priority")
    print(f"• This creates diverse candidate pool for final selection")

if __name__ == "__main__":
    show_interleaving_with_actual_content()
