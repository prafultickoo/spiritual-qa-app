"""
Langchain-based utility functions for document processing.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Import enhanced components
from utils.enhanced_pdf_loader import load_documents_enhanced
from utils.enhanced_verse_detection import detect_verses_enhanced, EnhancedVerseDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    """Pydantic model for document chunks."""
    content: str
    metadata: Dict[str, Any]
    has_verses: bool = False
    verses: List[str] = []


def load_documents(directory_path: str, use_enhanced_loader: bool = True) -> List[Document]:
    """
    Load all PDF documents from a directory with optional enhanced Unicode support.
    
    Args:
        directory_path (str): Path to the directory containing PDF documents
        use_enhanced_loader (bool): Whether to use enhanced PDF loader with Unicode support
        
    Returns:
        List[Document]: List of loaded documents
    """
    try:
        logger.info(f"Loading documents from directory: {directory_path}")
        
        if use_enhanced_loader:
            logger.info("Using enhanced PDF loader with Unicode support")
            documents = load_documents_enhanced(directory_path)
        else:
            logger.info("Using standard PDF loader")
            # Fallback to original method
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} document chunks from {directory_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from {directory_path}: {str(e)}")
        logger.info("Falling back to standard PDF loader...")
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Fallback successful: loaded {len(documents)} documents")
            return documents
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            return []


def identify_verses(text: str, use_enhanced_detection: bool = True) -> List[str]:
    """
    Identify verses in the text using enhanced detection patterns.
    
    Args:
        text (str): Text to analyze for verses
        use_enhanced_detection (bool): Whether to use enhanced verse detection
        
    Returns:
        List[str]: List of identified verse text
    """
    if use_enhanced_detection:
        # Use enhanced detection
        verse_metadata = detect_verses_enhanced(text)
        
        # Extract verse texts from enhanced detection
        verses = []
        if 'verses' in verse_metadata:
            for verse_info in verse_metadata['verses']:
                if isinstance(verse_info, dict) and 'text' in verse_info:
                    verses.append(verse_info['text'])
                else:
                    verses.append(str(verse_info))
        
        return verses
    else:
        # Fallback to original method
        import re
        
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


def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 1500, 
    chunk_overlap: int = 200,
    preserve_verses: bool = True
) -> List[DocumentChunk]:
    """
    Split documents into chunks using Langchain's text splitters.
    Optionally preserves verse integrity.
    
    Args:
        documents (List[Document]): Documents to chunk
        chunk_size (int, optional): Target chunk size in characters
        chunk_overlap (int, optional): Overlap between chunks
        preserve_verses (bool, optional): Whether to preserve verse integrity
        
    Returns:
        List[DocumentChunk]: List of document chunks
    """
    try:
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        # Configure text splitter with separators that respect paragraph structure
        # and are less likely to split Sanskrit verses
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
        
        # Split the documents
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} initial chunks")
        
        # Process chunks to preserve verse integrity if requested
        if preserve_verses:
            chunks = _preserve_verse_integrity(chunks)
        
        # Convert to our DocumentChunk model with enhanced verse identification
        result_chunks = []
        detector = EnhancedVerseDetector()
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Use enhanced verse detection
            structural_metadata = detector.detect_verses_and_structure(content)
            verses = [verse.text for verse in structural_metadata.verses_detected]
            
            # Create enhanced metadata
            enhanced_metadata = dict(chunk.metadata)
            
            # Add structural information
            if structural_metadata.source_text:
                enhanced_metadata['source_text'] = structural_metadata.source_text
            if structural_metadata.chapter_number:
                enhanced_metadata['chapter_number'] = structural_metadata.chapter_number
            if structural_metadata.chapter_title:
                enhanced_metadata['chapter_title'] = structural_metadata.chapter_title
            
            # Add Sanskrit/transliteration flags
            enhanced_metadata['contains_sanskrit'] = structural_metadata.contains_sanskrit
            enhanced_metadata['contains_transliteration'] = structural_metadata.contains_transliteration
            
            # Add verse-specific metadata
            if structural_metadata.verses_detected:
                verse_metadata = detector.create_verse_metadata(structural_metadata.verses_detected)
                enhanced_metadata.update(verse_metadata)
            
            result_chunks.append(DocumentChunk(
                content=content,
                metadata=enhanced_metadata,
                has_verses=len(verses) > 0,
                verses=verses
            ))
        
        logger.info(f"Final chunk count: {len(result_chunks)}")
        return result_chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {str(e)}")
        return []


def _preserve_verse_integrity(chunks: List[Document]) -> List[Document]:
    """
    Process chunks to ensure verse integrity across chunk boundaries.
    
    Args:
        chunks (List[Document]): Original document chunks
        
    Returns:
        List[Document]: Processed chunks with verse integrity preserved
    """
    from copy import deepcopy
    import re
    
    logger.info("Preserving verse integrity across chunks")
    
    # Define verse pattern
    verse_pattern = r'(\d+\.\d+\s*[\|\॥].+?[\|\॥])|(\d+\s*[\|\॥].+?[\|\॥])|॥.+?॥'
    
    # Create a copy to avoid modifying while iterating
    result_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        
        # Check if this chunk ends with a partial verse
        content = current_chunk.page_content
        
        # Look for verses that might be split at the end of this chunk
        matches = list(re.finditer(verse_pattern, content))
        
        if matches and i < len(chunks) - 1:
            last_match = matches[-1]
            last_match_end = last_match.end()
            
            # If the verse is close to the end of the chunk
            if last_match_end > len(content) - 100:
                # Check if it might continue in the next chunk
                next_chunk = chunks[i+1]
                next_content = next_chunk.page_content
                
                # If next chunk starts with something that looks like a verse continuation
                if re.match(verse_pattern, next_content[:100]):
                    # Merge the chunks
                    merged_content = content + "\n" + next_content
                    merged_metadata = deepcopy(current_chunk.metadata)
                    merged_metadata["merged"] = True
                    
                    merged_chunk = Document(
                        page_content=merged_content,
                        metadata=merged_metadata
                    )
                    
                    result_chunks.append(merged_chunk)
                    i += 2  # Skip the next chunk since we merged it
                    continue
        
        # If no merge happened, add the current chunk as is
        result_chunks.append(current_chunk)
        i += 1
    
    logger.info(f"After verse preservation: {len(result_chunks)} chunks")
    return result_chunks
