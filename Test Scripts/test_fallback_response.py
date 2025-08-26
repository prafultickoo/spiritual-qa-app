"""
Test fallback response for out-of-context queries
"""
import os
from answer_generator import AnswerGenerator

def test_fallback_for_pizza():
    """Test that pizza query triggers fallback response"""
    print("=" * 70)
    print("TESTING FALLBACK RESPONSE FOR OUT-OF-CONTEXT QUERIES")
    print("=" * 70)
    
    # Initialize answer generator
    vector_store_dir = "./vector_store"
    generator = AnswerGenerator(vector_store_dir)
    
    # Test queries
    test_queries = [
        "What is a pizza?",
        "How do I make a hamburger?",
        "What's the best smartphone to buy?",
        "What is karma?"  # This should NOT trigger fallback
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        try:
            # Generate answer without conversation context
            result = generator.generate_answer_with_context(
                query=query,
                conversation_context="",
                max_docs=5,
                k=10,
                use_mmr=True  # Use default MMR
            )
            
            if result.get('status') == 'success':
                fallback_triggered = result.get('fallback_triggered', False)
                
                if fallback_triggered:
                    print("✅ FALLBACK TRIGGERED")
                    relevance_info = result.get('relevance_info', {})
                    print(f"   Max Score: {relevance_info.get('max_score', 'N/A'):.3f}")
                    print(f"   Avg Score: {relevance_info.get('avg_score', 'N/A'):.3f}")
                    print(f"   Threshold: {relevance_info.get('threshold', 'N/A')}")
                else:
                    print("📚 NORMAL RESPONSE (from documents)")
                    print(f"   Chunks used: {result.get('chunks_used', 0)}")
                
                print(f"\nAnswer: {result.get('answer', '')[:200]}...")
            else:
                print(f"❌ Error: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✅ FALLBACK TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_fallback_for_pizza()
