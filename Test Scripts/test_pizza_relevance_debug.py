"""
Test relevance scores for out-of-context queries like 'pizza' with debug logging
"""
import os
import json
from document_retriever import DocumentRetriever

def test_pizza_relevance_debug():
    """Test what chunks and scores are retrieved for 'pizza' query with debug info"""
    print("=" * 70)
    print("TESTING RELEVANCE SCORES FOR 'PIZZA' QUERY (DEBUG MODE)")
    print("=" * 70)
    
    # Initialize retriever
    vector_store_dir = "./vector_store"
    retriever = DocumentRetriever(vector_store_dir)
    
    # Test query
    query = "What is a pizza?"
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    
    # Retrieve chunks with scores
    result = retriever.retrieve_chunks(query, k=5)
    
    # Debug: Print raw result
    print("\nDEBUG: Raw result structure:")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    if result.get('status') != 'success':
        print(f"Failed to retrieve chunks: {result.get('error', 'Unknown error')}")
        return
    
    chunks = result.get('chunks', [])
    if not chunks:
        print("No chunks retrieved!")
        return
    
    # Debug: Print first chunk structure
    print(f"\nDEBUG: First chunk structure:")
    if chunks:
        first_chunk = chunks[0]
        print(f"Chunk type: {type(first_chunk)}")
        if isinstance(first_chunk, dict):
            print(f"Chunk keys: {list(first_chunk.keys())}")
            print(f"Has relevance_score key: {'relevance_score' in first_chunk}")
            if 'relevance_score' in first_chunk:
                print(f"Relevance score value: {first_chunk['relevance_score']}")
                print(f"Relevance score type: {type(first_chunk['relevance_score'])}")
    
    # Display results
    print(f"\nRetrieved {len(chunks)} chunks:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nCHUNK {i}:")
        if isinstance(chunk, dict):
            print(f"Relevance Score: {chunk.get('relevance_score', 'N/A')}")
            print(f"Source: {chunk.get('source', 'Unknown')}")
            print(f"Content Preview: {chunk.get('content', '')[:200]}...")
            print(f"Has Verses: {chunk.get('has_verses', False)}")
        else:
            print(f"ERROR: Chunk is not a dict, it's a {type(chunk)}")
        
    # Analyze scores
    print("\n" + "=" * 70)
    print("SCORE ANALYSIS:")
    print("-" * 50)
    
    scores = []
    for chunk in chunks:
        if isinstance(chunk, dict) and 'relevance_score' in chunk:
            scores.append(chunk['relevance_score'])
    
    if scores:
        print(f"Found {len(scores)} scores out of {len(chunks)} chunks")
        print(f"Highest Score: {max(scores):.4f}")
        print(f"Lowest Score: {min(scores):.4f}")
        print(f"Average Score: {sum(scores)/len(scores):.4f}")
    else:
        print("No relevance scores found in chunks!")
        
    # Check if any chunk mentions pizza
    pizza_mentions = sum(1 for chunk in chunks 
                        if isinstance(chunk, dict) and 'pizza' in chunk.get('content', '').lower())
    print(f"\nChunks mentioning 'pizza': {pizza_mentions}/{len(chunks)}")
    
    print("\n" + "=" * 70)
    
if __name__ == "__main__":
    test_pizza_relevance_debug()
