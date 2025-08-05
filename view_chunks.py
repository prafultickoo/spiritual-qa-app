"""
Simple tool to view random chunks from the spiritual vector database.
"""
import os
import json
import random
import sys
import unicodedata
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import document retriever
from utils.document_retriever import SpiritualDocumentRetriever

def get_random_chunks(num_chunks: int = 3) -> List[Dict[str, Any]]:
    """
    Get random chunks from the vector database.
    
    Args:
        num_chunks: Number of chunks to retrieve
        
    Returns:
        List of document dictionaries
    """
    try:
        # Initialize document retriever
        retriever = SpiritualDocumentRetriever()
        
        # Use a simple query to retrieve documents
        generic_queries = [
            "karma",
            "meditation",
            "dharma"
        ]
        
        # Use random generic query
        query = random.choice(generic_queries)
        
        # Retrieve more documents than needed
        docs = retriever.retrieve_documents(query, num_docs=10)
        
        if not docs:
            print(f"No documents found for query: {query}")
            return []
        
        # Select random subset if we have enough docs
        if len(docs) > num_chunks:
            selected_docs = random.sample(docs, num_chunks)
        else:
            selected_docs = docs
        
        return selected_docs
    
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return []

def clean_sanskrit_text(text: str) -> str:
    """
    Clean and normalize Sanskrit text for display.
    
    Args:
        text: Text containing Sanskrit/Devanagari characters
        
    Returns:
        Cleaned text
    """
    # Option 1: Replace Devanagari with transliterated version
    def replace_devanagari(match):
        # This is a simplified example - a complete implementation would use a proper transliteration library
        devanagari = match.group(0)
        # Check if surrounding text contains a transliteration already
        context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
        
        # Look for common transliteration patterns after Devanagari text
        transliteration_match = re.search(r'\n([a-zA-Z¡-ÿ\s]+)\n', context[match.end()-match.start():])
        if transliteration_match:
            return f"[Sanskrit: {transliteration_match.group(1)}]"
        return "[Sanskrit verse - transliteration not available]"
    
    # Find blocks of Devanagari text and replace them
    processed_text = re.sub(r'[\u0900-\u097F\u0981-\u09FF\u0A01-\u0A7F\u0A81-\u0AFF\u0B01-\u0B7F\u0B81-\u0BFF\u0C01-\u0C7F\u0C81-\u0CFF\u0D01-\u0D7F\u0D82-\u0DFF\u0F00-\u0FFF\u1780-\u17FF\u1900-\u197F]+',
                           replace_devanagari, text)
    
    return processed_text

def display_chunk(chunk: Dict[str, Any], chunk_num: int, display_mode: str = 'clean') -> None:
    """
    Format and display a chunk nicely.
    
    Args:
        chunk: Document chunk to display
        chunk_num: Chunk number for display
        display_mode: How to display Sanskrit text ('clean', 'raw')
    """
    print(f"\n{'='*80}\n")
    print(f"CHUNK #{chunk_num}")
    print(f"{'='*80}")
    
    # Print metadata
    print(f"SOURCE: {chunk.get('source', 'Unknown')}")
    if 'metadata' in chunk:
        metadata = chunk['metadata']
        if 'title' in metadata and metadata['title']:
            print(f"TITLE: {metadata['title']}")
        if 'author' in metadata and metadata['author']:
            print(f"AUTHOR: {metadata['author']}")
        if 'page' in metadata:
            print(f"PAGE: {metadata.get('page', 'Unknown')}")
        if 'verse_count' in metadata and metadata['verse_count'] > 0:
            print(f"VERSES: {metadata['verse_count']}")
    
    print(f"\nRELEVANCE SCORE: {chunk.get('relevance_score', 'N/A')}")
    
    # Get content and process if needed
    content = chunk.get('content', '')
    if display_mode == 'clean':
        content = clean_sanskrit_text(content)
    
    print(f"\nCONTENT:\n{'-'*40}\n{content}\n{'-'*40}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Set display mode
    display_mode = 'clean'  # Options: 'clean', 'raw'
    
    # Check if options are provided
    if len(sys.argv) > 1:
        if sys.argv[1] == '--raw':
            display_mode = 'raw'
    
    print(f"\nRetrieving random chunks from spiritual vector database (display mode: {display_mode})...")
    chunks = get_random_chunks(3)
    
    if chunks:
        print(f"\nFound {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            display_chunk(chunk, i, display_mode)
    else:
        print("\nNo chunks found or error occurred.")
    
    print("\nUsage:")
    print("  python view_chunks.py         # Shows cleaned Sanskrit text")
    print("  python view_chunks.py --raw   # Shows raw Sanskrit characters")
