"""
Agent tools for document vectorization and vector database operations.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from crewai import Tool

# Import vectorization utilities
from utils.vectorization_utils import create_vectorizer, DocumentVectorizer
from utils.langchain_utils import DocumentChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def vectorize_chunks_tool(chunks: List[DocumentChunk], 
                          persist_directory: str,
                          embedding_model: str = "openai",
                          collection_name: str = "spiritual_texts") -> Dict[str, Any]:
    """
    Tool for vectorizing document chunks and storing them in a vector database.
    
    Args:
        chunks: List of DocumentChunk objects
        persist_directory: Directory to persist vector database
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Dict with vectorization results
    """
    logger.info(f"Vectorizing {len(chunks)} chunks using {embedding_model} embeddings")
    
    try:
        # Create vectorizer
        vectorizer = create_vectorizer(
            persist_directory=persist_directory,
            model_name=embedding_model,
            collection_name=collection_name
        )
        
        # Vectorize chunks
        result = vectorizer.vectorize_chunks(chunks)
        
        logger.info(f"Successfully vectorized {result.get('chunks_vectorized', 0)} chunks")
        return result
        
    except Exception as e:
        logger.error(f"Error in vectorize_chunks_tool: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "chunks_vectorized": 0
        }


def vectorize_from_json_tool(json_path: str,
                            persist_directory: str,
                            embedding_model: str = "openai",
                            collection_name: str = "spiritual_texts") -> Dict[str, Any]:
    """
    Tool for vectorizing document chunks from a JSON file.
    
    Args:
        json_path: Path to JSON file containing document chunks
        persist_directory: Directory to persist vector database
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Dict with vectorization results
    """
    logger.info(f"Vectorizing chunks from {json_path} using {embedding_model} embeddings")
    
    try:
        # Create vectorizer
        vectorizer = create_vectorizer(
            persist_directory=persist_directory,
            model_name=embedding_model,
            collection_name=collection_name
        )
        
        # Vectorize chunks from JSON
        result = vectorizer.vectorize_from_json(json_path)
        
        logger.info(f"Successfully vectorized {result.get('chunks_vectorized', 0)} chunks from JSON")
        return result
        
    except Exception as e:
        logger.error(f"Error in vectorize_from_json_tool: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "chunks_vectorized": 0
        }


def similarity_search_tool(query: str,
                           persist_directory: str,
                           embedding_model: str = "openai",
                           collection_name: str = "spiritual_texts",
                           k: int = 5) -> Dict[str, Any]:
    """
    Tool for performing similarity search on the vector database.
    
    Args:
        query: Query text
        persist_directory: Directory of vector database
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        collection_name: Name of the ChromaDB collection
        k: Number of results to return
        
    Returns:
        Dict with search results
    """
    logger.info(f"Performing similarity search for: '{query}'")
    
    try:
        # Create vectorizer
        vectorizer = create_vectorizer(
            persist_directory=persist_directory,
            model_name=embedding_model,
            collection_name=collection_name
        )
        
        # Load vector store
        if not vectorizer.load_vector_store():
            return {
                "status": "error",
                "error": "Failed to load vector store",
                "results": []
            }
        
        # Perform search
        results = vectorizer.similarity_search(query, k=k)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
        return {
            "status": "success",
            "results": formatted_results,
            "result_count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error in similarity_search_tool: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "results": []
        }


# CrewAI Tool objects
vectorize_chunks = Tool(
    name="vectorize_chunks",
    description="Vectorize document chunks and store them in a vector database",
    func=vectorize_chunks_tool
)

vectorize_from_json = Tool(
    name="vectorize_from_json",
    description="Vectorize document chunks from a JSON file",
    func=vectorize_from_json_tool
)

similarity_search = Tool(
    name="similarity_search",
    description="Perform similarity search on the vector database",
    func=similarity_search_tool
)
