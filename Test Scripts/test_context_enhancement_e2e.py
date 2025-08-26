#!/usr/bin/env python3
"""
End-to-End Test for Context Enhancement Pipeline
Tests the full flow from API request through context enhancement to final answer
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_document_retriever import EnhancedDocumentRetriever
from answer_generator import AnswerGenerator
from utils.conversation_context import ConversationContextProcessor
from utils.semantic_analyzer import SemanticAnalyzer
from utils.query_classifier import LLMQueryClassifier


def test_context_enhancement_e2e():
    """Test the complete context enhancement pipeline end-to-end."""
    
    print("=" * 70)
    print("CONTEXT ENHANCEMENT PIPELINE - END-TO-END TEST")
    print("=" * 70)
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # Initialize OpenAI client for Stage 3 LLM classification
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI client initialized")
    
    # Initialize enhanced retriever with context enhancement
    retriever = EnhancedDocumentRetriever(
        vector_store_dir="./vector_store",
        enable_dual_source=True,
        enable_context_enhancement=True
    )
    print("✅ Enhanced Document Retriever initialized")
    
    # Initialize answer generator
    generator = AnswerGenerator(
        vector_store_dir="./vector_store",
        enable_dual_source=True
    )
    print("✅ Answer Generator initialized")
    print()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "1. Direct Query (No Context Needed)",
            "conversation_history": [],
            "query": "What is the meaning of dharma in Hinduism?",
            "expected_behavior": "Should retrieve directly without enhancement"
        },
        {
            "name": "2. Simple Follow-up with Pronoun",
            "conversation_history": [
                {"role": "user", "content": "What is karma yoga?"},
                {"role": "assistant", "content": "Karma Yoga is the path of selfless action..."}
            ],
            "query": "Tell me more about this concept",
            "expected_behavior": "Should enhance query with context about karma yoga"
        },
        {
            "name": "3. Reformatting Request",
            "conversation_history": [
                {"role": "user", "content": "Explain the concept of moksha"},
                {"role": "assistant", "content": "Moksha is liberation from the cycle of rebirth..."}
            ],
            "query": "Can you summarize that in bullet points?",
            "expected_behavior": "Should identify as reformatting (no RAG needed)"
        },
        {
            "name": "4. Perspective Application",
            "conversation_history": [
                {"role": "user", "content": "What are the yamas in yoga?"},
                {"role": "assistant", "content": "The yamas are ethical principles including ahimsa..."}
            ],
            "query": "How can I apply these in modern workplace?",
            "expected_behavior": "Should apply workplace perspective to yamas"
        }
    ]
    
    # Run tests
    for scenario in test_scenarios:
        print(f"🧪 TEST: {scenario['name']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Context: {len(scenario['conversation_history'])} previous exchanges")
        print(f"   Expected: {scenario['expected_behavior']}")
        print()
        
        # Test retrieval with context
        start_time = time.time()
        
        try:
            # Use retrieve_chunks_with_context method
            result = retriever.retrieve_chunks_with_context(
                query=scenario['query'],
                conversation_history=scenario['conversation_history'],
                k=5,
                use_mmr=True,
                llm_client=openai_client,  # Pass actual OpenAI client for Stage 3
                model="gpt-4o-mini"  # Test with specific model
            )
            
            retrieval_time = time.time() - start_time
            
            # Analyze results
            query_info = result.get('query_info', {})
            
            print(f"   ✅ Retrieval completed in {retrieval_time:.2f}s")
            print(f"   📊 Results:")
            print(f"      - Status: {result.get('status')}")
            print(f"      - Original query: {query_info.get('original_query')}")
            print(f"      - Processed query: {query_info.get('processed_query')}")
            print(f"      - Context enhanced: {query_info.get('context_enhanced', False)}")
            print(f"      - Pipeline stage: {query_info.get('pipeline_stage', 0)}")
            print(f"      - Action: {query_info.get('action', 'N/A')}")
            print(f"      - Intent: {query_info.get('intent', 'N/A')}")
            print(f"      - Needs RAG: {query_info.get('action') != 'reformat_previous'}")
            print(f"      - Chunks retrieved: {len(result.get('chunks', []))}")
            
            # Check if behavior matches expectation
            if query_info.get('context_enhanced'):
                print(f"   🎯 Query was enhanced with context")
            elif query_info.get('pipeline_stage', 0) > 0:
                print(f"   🎯 Query went through {query_info.get('pipeline_stage')} pipeline stages")
            else:
                print(f"   🎯 Query processed directly (no context needed)")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            
        print("-" * 70)
        print()
    
    # Test full answer generation with context
    print("🔍 TESTING FULL ANSWER GENERATION WITH CONTEXT")
    print("=" * 70)
    
    # Create a conversation context string
    conversation_context = """User: What is the concept of dharma?
Assistant: Dharma is a fundamental concept in Hindu philosophy that encompasses righteousness, moral law, and duty. It represents the cosmic order and the right way of living that sustains harmony in the universe and society."""
    
    query = "How does this relate to karma?"
    
    print(f"Query: '{query}'")
    print("Conversation context provided: Yes")
    print()
    
    try:
        # Generate answer with context
        result = generator.generate_answer_with_context(
            query=query,
            conversation_context=conversation_context,
            max_docs=5,
            k=10,
            use_mmr=True
        )
        
        if result.get('status') == 'success':
            print("✅ Answer generation successful!")
            print(f"📝 Answer preview: {result.get('answer', '')[:200]}...")
            print(f"📚 Sources used: {len(result.get('sources', []))}")
            print(f"📖 Verses referenced: {len(result.get('verses', []))}")
        else:
            print(f"❌ Answer generation failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error generating answer: {str(e)}")
    
    print()
    print("=" * 70)
    print("✅ END-TO-END TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_context_enhancement_e2e()
