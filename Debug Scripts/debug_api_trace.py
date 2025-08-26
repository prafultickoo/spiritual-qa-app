"""
Debug API trace to understand where model field becomes None
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator
import json

def test_direct_api_call():
    """Test the API logic directly to trace where model becomes None"""
    print("=" * 70)
    print("TRACING API FLOW DIRECTLY")
    print("=" * 70)
    
    # Initialize answer generator like the API does
    generator = AnswerGenerator(
        vector_store_dir="vector_store",
        llm_model="gpt-4.1"
    )
    
    print(f"\n1. Generator initialized with llm_model: {generator.llm_model}")
    
    # Test 1: Call without conversation context (should work)
    print("\n2. Testing generate_answer WITHOUT context...")
    result1 = generator.generate_answer(
        query="What is para vidya?",
        k=10,
        reasoning_effort=None
    )
    print(f"Result keys: {list(result1.keys())}")
    print(f"Model field: '{result1.get('model', 'MISSING')}'")
    print(f"Status: {result1.get('status')}")
    
    # Test 2: Call with conversation context (problematic)
    print("\n3. Testing generate_answer_with_context...")
    conversation_context = "User: What is apara vidya?\n\nAssistant: Apara vidya is worldly knowledge..."
    
    try:
        result2 = generator.generate_answer_with_context(
            query="Can you explain in bullets?",
            conversation_context=conversation_context,
            max_docs=5,
            k=10,
            use_mmr=True,
            diversity=0.3,
            reasoning_effort=None
        )
        print(f"Result keys: {list(result2.keys())}")
        print(f"Model field: '{result2.get('model', 'MISSING')}'")
        print(f"Status: {result2.get('status')}")
        
        # Check if this is a fallback response
        if result2.get('fallback_triggered'):
            print("FALLBACK WAS TRIGGERED!")
            print(f"Relevance info: {result2.get('relevance_info')}")
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_api_call()
