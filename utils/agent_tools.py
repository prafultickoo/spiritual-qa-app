"""
Agent tools for document processing using Langchain.
"""
import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain.schema import Document
from crewai_tools import BaseTool
from utils.langchain_utils import (
    load_documents,
    chunk_documents,
    identify_verses,
    DocumentChunk
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define tools for CrewAI agents

def document_loader_tool(directory: str) -> List[Document]:
    """
    Tool for loading documents from a directory using Langchain.
    
    Args:
        directory (str): Directory path containing documents
        
    Returns:
        List[Document]: List of loaded documents
    """
    logger.info(f"Using document_loader_tool to load from: {directory}")
    return load_documents(directory)


def document_chunker_tool(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    Tool for chunking documents using Langchain text splitters.
    
    Args:
        documents (List[Document]): Documents to chunk
        chunk_size (int, optional): Target chunk size in characters
        chunk_overlap (int, optional): Overlap between chunks
        
    Returns:
        List[DocumentChunk]: List of document chunks
    """
    logger.info(f"Using document_chunker_tool to create chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    return chunk_documents(documents, chunk_size, chunk_overlap, preserve_verses=True)


def chunk_verifier_tool(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """
    Tool for verifying chunk quality and integrity.
    
    Args:
        chunks (List[DocumentChunk]): Chunks to verify
        
    Returns:
        Dict[str, Any]: Verification results
    """
    logger.info(f"Using chunk_verifier_tool to verify {len(chunks)} chunks")
    
    verification_results = {
        "verified_chunks": [],
        "issues_found": [],
        "overall_quality": "High"
    }
    
    for i, chunk in enumerate(chunks):
        # Check for basic issues
        issues = []
        
        # Check chunk size (based on rough token estimate)
        token_count = len(chunk.content) // 4  # Rough estimation: 1 token ≈ 4 characters
        
        if token_count < 200:
            issues.append(f"Chunk {i} is too small: ~{token_count} tokens")
        elif token_count > 2000:
            issues.append(f"Chunk {i} is too large: ~{token_count} tokens")
        
        # Check verse integrity
        if chunk.has_verses:
            # Verify verses are complete
            for verse in chunk.verses:
                if verse.strip() != verse:
                    issues.append(f"Chunk {i} may have incomplete verse: {verse[:50]}...")
        
        # Add chunk to appropriate list
        if issues:
            verification_results["issues_found"].append({
                "chunk_index": i,
                "issues": issues,
                "metadata": chunk.metadata
            })
        else:
            verification_results["verified_chunks"].append(chunk)
    
    # Update overall quality based on issues found
    if len(verification_results["issues_found"]) > len(chunks) * 0.2:
        verification_results["overall_quality"] = "Low"
    elif len(verification_results["issues_found"]) > 0:
        verification_results["overall_quality"] = "Medium"
    
    logger.info(f"Verification complete: {len(verification_results['verified_chunks'])} chunks verified")
    return verification_results


def save_chunks_tool(chunks: List[DocumentChunk], output_path: str) -> Dict[str, Any]:
    """
    Tool for saving processed chunks to disk.
    
    Args:
        chunks (List[DocumentChunk]): Chunks to save
        output_path (str): Path where to save chunks
        
    Returns:
        Dict[str, Any]: Result of the save operation
    """
    try:
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert chunks to dictionaries for JSON serialization
        chunks_dict = [chunk.dict() for chunk in chunks]
        
        with open(output_path, 'w') as f:
            json.dump(chunks_dict, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        return {"status": "success", "chunks_saved": len(chunks), "path": output_path}
    except Exception as e:
        logger.error(f"Error saving chunks to {output_path}: {str(e)}")
        return {"status": "error", "error": str(e)}


# CrewAI Tool objects
document_loader = BaseTool(
    name="document_loader",
    description="Load PDF documents from a directory using Langchain",
    func=document_loader_tool
)

document_chunker = BaseTool(
    name="document_chunker",
    description="Split documents into chunks while preserving verse structure",
    func=document_chunker_tool
)

chunk_verifier = BaseTool(
    name="chunk_verifier",
    description="Verify chunk quality and integrity, especially for verses",
    func=chunk_verifier_tool
)

chunk_saver = BaseTool(
    name="chunk_saver",
    description="Save processed chunks to a JSON file",
    func=save_chunks_tool
)
