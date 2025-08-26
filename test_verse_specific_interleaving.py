#!/usr/bin/env python3
"""
Test interleaving logic with verse-specific query.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from answer_generator import AnswerGenerator

load_dotenv()

def test_verse_specific_interleaving():
    """Test if interleaving algorithm adapts for verse-specific queries."""
    
    print("🔍 TESTING VERSE-SPECIFIC INTERLEAVING")
    print("=" * 100)
    
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        llm_model="o3-mini",
        enable_dual_source=True
    )
    
    # Test with explicit verse query
    verse_query = "Explain Chapter 3 Verse 5"
    print(f"Query: '{verse_query}'")
    
    dual_source_retriever = generator.retriever.dual_source_retriever
    raw_result = dual_source_retriever.retrieve(query=verse_query, k=5)
    
    print(f"\n📊 QUERY ANALYSIS:")
    print(f"Query Type: {raw_result.query_type}")
    print(f"Chapter: {raw_result.chapter}")
    print(f"Verse: {raw_result.verse}")
    
    print(f"\n📜 VERSES RETRIEVED: {len(raw_result.clean_verses)}")
    for i, verse_doc in enumerate(raw_result.clean_verses, 1):
        content = verse_doc.page_content[:100] + "..." if len(verse_doc.page_content) > 100 else verse_doc.page_content
        print(f"  Verse #{i}: {content}")
    
    print(f"\n📝 EXPLANATIONS RETRIEVED: {len(raw_result.explanations)}")
    for i, explanation_doc in enumerate(raw_result.explanations, 1):
        content = explanation_doc.page_content[:100] + "..." if len(explanation_doc.page_content) > 100 else explanation_doc.page_content
        print(f"  Explanation #{i}: {content}")
    
    print(f"\n🔄 FINAL MERGED RESULT: {len(raw_result.merged_context)}")
    for i, merged_doc in enumerate(raw_result.merged_context, 1):
        content = merged_doc.page_content[:100] + "..." if len(merged_doc.page_content) > 100 else merged_doc.page_content
        print(f"  Final #{i}: {content}")
    
    # Count distribution in final result
    verse_count = 0
    explanation_count = 0
    
    # We need to check which collection each final chunk came from
    final_contents = [doc.page_content for doc in raw_result.merged_context]
    verse_contents = [doc.page_content for doc in raw_result.clean_verses]
    explanation_contents = [doc.page_content for doc in raw_result.explanations]
    
    for content in final_contents:
        if content in verse_contents:
            verse_count += 1
        elif content in explanation_contents:
            explanation_count += 1
    
    print(f"\n📈 FINAL DISTRIBUTION:")
    print(f"Verses: {verse_count}/5")
    print(f"Explanations: {explanation_count}/5")
    
    print(f"\n🤔 ANALYSIS:")
    if raw_result.query_type == 'chapter_verse':
        print("✅ Query detected as chapter_verse - should prioritize explanations")
        if explanation_count > verse_count:
            print("✅ Algorithm DID adapt - more explanations than verses")
        else:
            print("❌ Algorithm did NOT adapt - still using rigid pattern")
    else:
        print(f"⚠️  Query detected as '{raw_result.query_type}' - not chapter_verse")
    
    # Test another verse query format
    print(f"\n" + "="*100)
    print("TESTING DIFFERENT VERSE QUERY FORMAT")
    print("="*100)
    
    verse_query2 = "What does Bhagavad Gita Chapter 2 Verse 47 mean?"
    print(f"Query: '{verse_query2}'")
    
    raw_result2 = dual_source_retriever.retrieve(query=verse_query2, k=5)
    
    print(f"\n📊 QUERY ANALYSIS:")
    print(f"Query Type: {raw_result2.query_type}")
    print(f"Chapter: {raw_result2.chapter}")
    print(f"Verse: {raw_result2.verse}")
    
    # Count distribution in second result
    verse_count2 = 0
    explanation_count2 = 0
    
    final_contents2 = [doc.page_content for doc in raw_result2.merged_context]
    verse_contents2 = [doc.page_content for doc in raw_result2.clean_verses]
    explanation_contents2 = [doc.page_content for doc in raw_result2.explanations]
    
    for content in final_contents2:
        if content in verse_contents2:
            verse_count2 += 1
        elif content in explanation_contents2:
            explanation_count2 += 1
    
    print(f"\n📈 FINAL DISTRIBUTION:")
    print(f"Verses: {verse_count2}/5")
    print(f"Explanations: {explanation_count2}/5")

if __name__ == "__main__":
    test_verse_specific_interleaving()
