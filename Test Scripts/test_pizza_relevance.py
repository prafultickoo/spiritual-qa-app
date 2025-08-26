"""
Test relevance scores for out-of-context queries like 'pizza'
"""
import os
from document_retriever import DocumentRetriever

def test_pizza_relevance():
    """Test what chunks and scores are retrieved for 'pizza' query"""
    print("=" * 70)
    print("TESTING RELEVANCE SCORES FOR 'PIZZA' QUERY")
    print("=" * 70)
    
    # Initialize retriever
    vector_store_dir = "./vector_store"
    retriever = DocumentRetriever(vector_store_dir)
    
    # Test query
    query = "What is a pizza?"
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    
    # Retrieve chunks with scores (using default MMR)
    result = retriever.retrieve_chunks(query, k=5)  # use_mmr=True by default
    
    if result.get('status') != 'success':
        print(f"Failed to retrieve chunks: {result.get('error', 'Unknown error')}")
        return
    
    chunks = result.get('chunks', [])
    if not chunks:
        print("No chunks retrieved!")
        return
    
    # Display results
    print(f"\nRetrieved {len(chunks)} chunks:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nCHUNK {i}:")
        print(f"Relevance Score: {chunk.get('relevance_score', 'N/A')}")
        print(f"Source: {chunk.get('source', 'Unknown')}")
        print(f"Content Preview: {chunk.get('content', '')[:200]}...")
        print(f"Has Verses: {chunk.get('has_verses', False)}")
        
    # Analyze scores
    print("\n" + "=" * 70)
    print("SCORE ANALYSIS:")
    print("-" * 50)
    
    scores = [chunk.get('relevance_score', 0) for chunk in chunks]
    if scores:
        print(f"Highest Score: {max(scores):.4f}")
        print(f"Lowest Score: {min(scores):.4f}")
        print(f"Average Score: {sum(scores)/len(scores):.4f}")
        
        # Check if any chunk mentions pizza
        pizza_mentions = sum(1 for chunk in chunks if 'pizza' in chunk.get('content', '').lower())
        print(f"\nChunks mentioning 'pizza': {pizza_mentions}/{len(chunks)}")
        
        # Check what the chunks are actually about
        print("\nActual content themes:")
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '').lower()
            themes = []
            if 'food' in content or 'eat' in content:
                themes.append('food/eating')
            if 'sattv' in content:
                themes.append('sattvic')
            if 'karm' in content:
                themes.append('karma')
            if 'yoga' in content:
                themes.append('yoga')
            print(f"  Chunk {i}: {', '.join(themes) if themes else 'unknown'}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("-" * 50)
    print("Based on these scores, we should set a relevance threshold.")
    print("If all scores are below the threshold, trigger the fallback response.")
    
if __name__ == "__main__":
    test_pizza_relevance()
