#!/usr/bin/env python3
"""
Debug script to test semantic analysis for "Summarize the main points"
to understand why it's still being classified as needs_context=False
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.semantic_analyzer import SemanticAnalyzer

def test_semantic_analysis():
    """Test semantic analysis for problematic query"""
    
    print("=" * 80)
    print("🔍 DEBUG SEMANTIC ANALYSIS")
    print("=" * 80)
    
    analyzer = SemanticAnalyzer()
    
    # Test the problematic query
    query = "Summarize the main points"
    conversation_history = [
        {'role': 'user', 'content': 'Explain the concept of dharma'},
        {'role': 'assistant', 'content': 'Dharma is your life\'s purpose and righteous duty according to your nature, circumstances, and stage of life. It represents the moral and ethical principles that guide right living. Dharma is not the same for everyone - a teacher\'s dharma differs from a soldier\'s or a parent\'s. Following your dharma means acting in harmony with your true nature while contributing to the greater good of society and the universe.'}
    ]
    
    print(f"Query: '{query}'")
    print(f"Has conversation history: {len(conversation_history)} items")
    
    # Test linguistic completeness analysis
    print("\n--- STEP 1: Linguistic Completeness Analysis ---")
    completeness = analyzer.analyze_linguistic_completeness(query)
    print(f"Is Complete: {completeness['is_complete']}")
    print(f"Missing Elements: {completeness['missing_elements']}")
    print(f"Confidence: {completeness['confidence']}")
    print(f"Analysis: {completeness['analysis']}")
    
    # Test full ambiguity analysis
    print("\n--- STEP 2: Full Ambiguity Analysis ---")
    ambiguity = analyzer.analyze_query_ambiguity(query, conversation_history)
    print(f"Is Ambiguous: {ambiguity['is_ambiguous']}")
    print(f"Needs Context: {ambiguity['needs_context']}")
    print(f"Confidence: {ambiguity['confidence']}")
    print(f"Analysis Details: {ambiguity['analysis_details']}")
    
    print("\n" + "=" * 80)
    print("EXPECTED: needs_context=True (but getting False)")
    print("=" * 80)

if __name__ == "__main__":
    test_semantic_analysis()
