"""
Utility functions for document chunking.
"""
import re
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def split_text_into_chunks(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    Prioritizes verse integrity over strict chunk size.
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int, optional): Target size of each chunk in tokens (approx. 4 chars per token)
        chunk_overlap (int, optional): Overlap between consecutive chunks in tokens
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info(f"Splitting text into chunks (target size: {chunk_size}, overlap: {chunk_overlap})")
    
    # Approximate conversion from tokens to characters (4 chars ~= 1 token)
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    # First, try to identify verse patterns
    # Simple verse identification pattern - can be expanded based on actual document structure
    verse_pattern = r'(\d+\.\d+\s*[\|\॥].+?[\|\॥])|(\d+\s*[\|\॥].+?[\|\॥])'
    
    # Identify paragraphs
    paragraphs = re.split(r'\n\n+', text)
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed the chunk size
        if len(paragraph) + current_size <= char_size or current_size == 0:
            # If it fits or it's the first paragraph, add it to current chunk
            current_chunk += paragraph + "\n\n"
            current_size += len(paragraph) + 2
        else:
            # If it doesn't fit, start a new chunk with some overlap
            chunks.append(current_chunk)
            
            # Get the tail of the last chunk for overlap
            words = current_chunk.split()
            overlap_word_count = min(len(words), int(char_overlap / 5))  # Approx. 5 chars per word
            overlap_text = ' '.join(words[-overlap_word_count:]) if overlap_word_count > 0 else ""
            
            # Start new chunk with overlap and add current paragraph
            current_chunk = overlap_text + "\n\n" + paragraph + "\n\n"
            current_size = len(current_chunk)
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-processing: Ensure verse integrity
    chunks = ensure_verse_integrity(chunks, verse_pattern)
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def ensure_verse_integrity(chunks: List[str], verse_pattern: str) -> List[str]:
    """
    Ensure that verses aren't split across chunks.
    
    Args:
        chunks (List[str]): Initial text chunks
        verse_pattern (str): Regex pattern to identify verses
        
    Returns:
        List[str]: Chunks with preserved verse integrity
    """
    logger.info("Ensuring verse integrity across chunks")
    result_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Look for verses that might be split at the end of this chunk
        matches = re.finditer(verse_pattern, chunk)
        last_match_end = 0
        last_match = None
        
        for match in matches:
            last_match = match
            last_match_end = match.end()
        
        # If we have a match and it's near the end of the chunk
        if last_match and chunk[last_match_end:].strip() == "" and i < len(chunks) - 1:
            # The verse is the last thing in this chunk, check if it continues in the next chunk
            next_chunk = chunks[i+1]
            if re.match(verse_pattern, next_chunk.strip()):
                # It likely continues, so merge with next chunk
                chunks[i+1] = chunk + next_chunk
                continue
        
        result_chunks.append(chunk)
    
    return result_chunks

def estimate_token_count(text: str) -> int:
    """
    Estimate token count based on text length.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4

def extract_verses(text: str) -> List[str]:
    """
    Extract verses from text based on common verse formatting.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: Extracted verses
    """
    # Various verse patterns to look for
    patterns = [
        r'\d+\.\d+\s*[\|\॥].+?[\|\॥]',  # Format: 1.1 | verse text |
        r'\d+\s*[\|\॥].+?[\|\॥]',  # Format: 1 | verse text |
        r'श्लोक[^\n]+\n.+?\n\n',  # Sanskrit shloka format
        r'॥.+?॥'  # Traditional Sanskrit verse markers
    ]
    
    verses = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            verses.append(match.group(0))
    
    return verses

def format_chunk_for_embedding(chunk: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a chunk with its metadata for embedding.
    
    Args:
        chunk (str): Text chunk
        metadata (Dict[str, Any]): Document metadata
        
    Returns:
        Dict[str, Any]: Formatted chunk with metadata
    """
    # Estimate token count
    token_count = estimate_token_count(chunk)
    
    # Extract verses if present
    verses = extract_verses(chunk)
    
    return {
        "text": chunk,
        "token_count": token_count,
        "verses": verses,
        "source": metadata.get("filename", ""),
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "verse_count": len(verses)
    }
