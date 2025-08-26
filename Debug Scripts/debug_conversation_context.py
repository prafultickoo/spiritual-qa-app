"""
Debug script to identify why conversation context is not working properly.

The issue: Follow-up questions are getting fallback responses instead of using 
conversation history to provide proper contextual answers.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator
import json

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_conversation_context():
    """Debug why conversation context is failing"""
    print("=" * 80)
    print("🔍 DEBUGGING CONVERSATION CONTEXT ISSUE")
    print("=" * 80)
    
    # Initialize answer generator
    generator = AnswerGenerator(
        vector_store_dir="vector_store",
        llm_model="gpt-4.1"
    )
    
    print(f"✅ Generator initialized with model: {generator.llm_model}")
    print()
    
    # Step 1: Test normal question (should work)
    print("📋 Step 1: Testing Normal Question (Baseline)")
    normal_result = generator.generate_answer(
        query="What is Karma Yoga?",
        k=10
    )
    print(f"Normal question status: {normal_result.get('status')}")
    print(f"Normal question answer length: {len(normal_result.get('answer', ''))}")
    print(f"Normal question answer preview: {normal_result.get('answer', '')[:200]}...")
    print()
    
    # Step 2: Test follow-up WITH conversation context (the broken case)
    print("📋 Step 2: Testing Follow-up WITH Conversation Context (BROKEN)")
    
    conversation_context = f"""User: What is Karma Yoga?
Assistant: {normal_result.get('answer', '')}"""
    
    print(f"Conversation context length: {len(conversation_context)}")
    print(f"Conversation context preview: {conversation_context[:300]}...")
    print()
    
    # Test the problematic follow-up question
    followup_result = generator.generate_answer_with_context(
        query="Can you explain this in bullet points?",
        conversation_context=conversation_context,
        max_docs=5,
        k=10,
        use_mmr=True,
        diversity=0.3
    )
    
    print(f"Follow-up status: {followup_result.get('status')}")
    print(f"Follow-up fallback_triggered: {followup_result.get('fallback_triggered')}")
    print(f"Follow-up relevance_info: {followup_result.get('relevance_info')}")
    print(f"Follow-up answer: {followup_result.get('answer', '')[:300]}...")
    print()
    
    # Step 3: Debug the exact issue
    print("📋 Step 3: Analyzing Why Follow-up Triggers Fallback")
    
    if followup_result.get('fallback_triggered'):
        print("❌ PROBLEM IDENTIFIED: Follow-up is triggering fallback logic!")
        print("This means:")
        print("  1. The query 'Can you explain this in bullet points?' is being treated as irrelevant")
        print("  2. Retrieved chunks have low relevance scores")
        print("  3. Conversation context is NOT being used to enhance relevance")
        print()
        
        relevance_info = followup_result.get('relevance_info', {})
        print(f"Relevance details: {relevance_info}")
        
        chunks_used = followup_result.get('chunks_used', 0)
        print(f"Chunks used: {chunks_used}")
        
        sources = followup_result.get('sources', [])
        print(f"Sources found: {len(sources)}")
        
    else:
        print("✅ Follow-up did not trigger fallback (unexpected)")
        
    print()
    print("📋 Step 4: Testing Direct Context Enhancement")
    
    # Let's check if the context enhancement pipeline is working
    try:
        from enhanced_document_retriever import EnhancedDocumentRetriever
        retriever = EnhancedDocumentRetriever(vector_store_dir="vector_store")
        
        # Test context-aware retrieval
        chunks = retriever.retrieve_with_context(
            query="Can you explain this in bullet points?",
            conversation_history=[
                {"role": "user", "content": "What is Karma Yoga?"},
                {"role": "assistant", "content": normal_result.get('answer', '')[:500]}
            ],
            max_docs=5
        )
        
        print(f"Context-aware retrieval returned {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  Chunk {i+1}: {chunk.get('content', '')[:100]}...")
            print(f"    Relevance: {chunk.get('relevance_score', 'N/A')}")
        
    except Exception as e:
        print(f"Context enhancement test failed: {str(e)}")
        
    print()
    print("🔍 ROOT CAUSE ANALYSIS:")
    print("The issue is likely one of these:")
    print("1. Context enhancement is not working (query rewriting fails)")
    print("2. Relevance threshold is too high for follow-up queries")
    print("3. Conversation context is not being properly integrated")
    print("4. Fallback logic is incorrectly triggered for contextual queries")

if __name__ == "__main__":
    debug_conversation_context()
