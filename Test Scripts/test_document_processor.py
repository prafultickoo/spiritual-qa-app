"""
Test script for the agentic document processor with Langchain.
"""
import os
import json
import logging
from dotenv import load_dotenv
from utils.langchain_utils import load_documents, chunk_documents, identify_verses
from utils.agent_tools import document_loader_tool, document_chunker_tool, chunk_verifier_tool, save_chunks_tool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_document_loading():
    """Test document loading functionality using Langchain"""
    logger.info("Testing document loading with Langchain...")
    
    documents_dir = os.path.join(os.getcwd(), "Documents")
    documents = load_documents(documents_dir)
    
    logger.info(f"Loaded {len(documents)} document chunks from {documents_dir}")
    
    # Print preview of a few documents to verify content
    max_preview = min(3, len(documents))
    for i in range(max_preview):
        doc = documents[i]
        preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        logger.info(f"Document chunk {i+1}/{len(documents)}")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Page: {doc.metadata.get('page', 'Unknown')}")
        logger.info(f"Preview: {preview}")
        logger.info("-" * 80)
    
    return documents

def test_chunk_creation(documents):
    """Test chunk creation functionality using Langchain"""
    logger.info("Testing chunk creation with Langchain...")
    
    # Split into chunks using Langchain
    chunks = chunk_documents(documents, chunk_size=1500, chunk_overlap=200)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Print preview of first few chunks
    max_preview = min(3, len(chunks))
    for i in range(max_preview):
        chunk = chunks[i]
        preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        logger.info(f"Chunk {i+1} preview: {preview}")
        
        # Show verses if any
        if chunk.has_verses:
            logger.info(f"Chunk contains {len(chunk.verses)} verses")
            if chunk.verses:
                logger.info(f"First verse: {chunk.verses[0][:100]}...")
        
        logger.info("-" * 80)
    
    # Save all chunks to JSON for inspection
    output_dir = os.path.join(os.getcwd(), "Processed")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "test_chunks.json")
    save_chunks_tool(chunks, output_path)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    return chunks

def verify_chunk_quality(chunks):
    """Perform chunk quality verification using the verifier tool"""
    logger.info("Verifying chunk quality using verification tool...")
    
    # Use the verification tool
    verification_results = chunk_verifier_tool(chunks)
    
    # Process verification results
    verified_count = len(verification_results.get('verified_chunks', []))
    issues_count = len(verification_results.get('issues_found', []))
    quality = verification_results.get('overall_quality', 'Unknown')
    
    logger.info(f"Verification complete: {verified_count} chunks verified, {issues_count} issues found")
    logger.info(f"Overall quality: {quality}")
    
    # Log issues if found
    if issues_count > 0:
        logger.warning("Issues found:")
        for issue_data in verification_results.get('issues_found', []):
            chunk_index = issue_data.get('chunk_index', 'Unknown')
            issues = issue_data.get('issues', [])
            for issue in issues:
                logger.warning(f"- Chunk {chunk_index}: {issue}")
    
    return verification_results

if __name__ == "__main__":
    logger.info("Starting document processor test")
    
    # Test document loading
    documents = test_document_loading()
    
    # Test chunk creation
    if documents:
        chunks = test_chunk_creation(documents)
        
        # Verify chunk quality
        if chunks:
            verify_chunk_quality(chunks)
    
    logger.info("Test complete")
