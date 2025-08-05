"""
Agent tools for query processing and document retrieval.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from crewai import Tool

# Import our query utilities
from utils.query_utils import create_query_processor, QueryProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def process_query_tool(query: str, 
                      vector_store_dir: str,
                      embedding_model: str = "openai") -> Dict[str, Any]:
    """
    Tool for converting user queries into embeddings.
    
    Args:
        query: User query text
        vector_store_dir: Directory of the vector store
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        
    Returns:
        Dict with query embedding results
    """
    logger.info(f"Processing query: '{query}' using {embedding_model} embeddings")
    
    try:
        # Create query processor
        processor = create_query_processor(
            vector_store_dir=vector_store_dir,
            embedding_model=embedding_model
        )
        
        # Process query
        result = processor.process_query(query)
        
        # Don't log the actual embeddings (they're too verbose)
        if "embedding" in result:
            embedding_len = len(result["embedding"]) if isinstance(result["embedding"], list) else "N/A"
            logger.info(f"Successfully created embedding of dimension {embedding_len}")
            # Remove raw embeddings from result to avoid overwhelming logs
            result["embedding_dimension"] = embedding_len
            result["embedding"] = "[Embedding vector removed from logs]"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_query_tool: {str(e)}")
        return {
            "query": query,
            "status": "error",
            "error": str(e)
        }


def retrieve_relevant_chunks_tool(query: str,
                                 vector_store_dir: str,
                                 embedding_model: str = "openai",
                                 k: int = 5,
                                 filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Tool for retrieving relevant document chunks for a user query.
    
    Args:
        query: User query text
        vector_store_dir: Directory of the vector store
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        k: Number of chunks to retrieve
        filter_metadata: Optional metadata filter for retrieval
        
    Returns:
        Dict with retrieved chunks
    """
    logger.info(f"Retrieving {k} chunks for query: '{query}'")
    
    try:
        # Create query processor
        processor = create_query_processor(
            vector_store_dir=vector_store_dir,
            embedding_model=embedding_model
        )
        
        # Retrieve chunks
        result = processor.retrieve_relevant_chunks(
            query_text=query,
            k=k,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"Retrieved {result.get('chunk_count', 0)} chunks for query: '{query}'")
        return result
        
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_chunks_tool: {str(e)}")
        return {
            "query": query,
            "status": "error",
            "error": str(e),
            "chunks": []
        }


def mmr_retrieval_tool(query: str,
                      vector_store_dir: str,
                      embedding_model: str = "openai",
                      k: int = 5,
                      fetch_k: int = 15,
                      diversity: float = 0.7,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Tool for retrieving diverse document chunks using Maximum Marginal Relevance.
    
    Args:
        query: User query text
        vector_store_dir: Directory of the vector store
        embedding_model: Embedding model to use ('openai' or 'huggingface')
        k: Number of chunks to retrieve in final result
        fetch_k: Number of initial chunks to fetch before applying MMR
        diversity: MMR diversity parameter (0 = maximal diversity, 1 = maximal relevance)
        filter_metadata: Optional metadata filter for retrieval
        
    Returns:
        Dict with retrieved diverse chunks
    """
    logger.info(f"Retrieving {k} diverse chunks for query: '{query}' with diversity {diversity}")
    
    try:
        # Create query processor
        processor = create_query_processor(
            vector_store_dir=vector_store_dir,
            embedding_model=embedding_model
        )
        
        # Retrieve chunks with MMR
        result = processor.mmr_retrieval(
            query_text=query,
            k=k,
            fetch_k=fetch_k,
            diversity=diversity,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"Retrieved {result.get('chunk_count', 0)} diverse chunks for query: '{query}'")
        return result
        
    except Exception as e:
        logger.error(f"Error in mmr_retrieval_tool: {str(e)}")
        return {
            "query": query,
            "status": "error",
            "error": str(e),
            "chunks": []
        }


# CrewAI Tool objects
process_query = Tool(
    name="process_query",
    description="Convert user query into embeddings",
    func=process_query_tool
)

retrieve_chunks = Tool(
    name="retrieve_chunks",
    description="Retrieve relevant document chunks for a user query",
    func=retrieve_relevant_chunks_tool
)

retrieve_diverse_chunks = Tool(
    name="retrieve_diverse_chunks",
    description="Retrieve diverse document chunks using Maximum Marginal Relevance",
    func=mmr_retrieval_tool
)
