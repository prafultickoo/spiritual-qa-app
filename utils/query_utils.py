"""
Utilities for processing user queries and retrieving relevant document chunks.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Langchain imports
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

# Import our vectorization utilities to reuse embedding models
from utils.vectorization_utils import DocumentVectorizer, EMBEDDING_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class QueryProcessor:
    """Class for processing user queries and retrieving relevant document chunks."""

    def __init__(self,
                 vector_store_dir: str,
                 embedding_model: str = "openai",
                 embedding_model_kwargs: Optional[Dict[str, Any]] = None,
                 collection_name: str = "spiritual_texts"):
        """
        Initialize the query processor.
        
        Args:
            vector_store_dir: Directory of the vector store
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            embedding_model_kwargs: Additional keyword arguments for embedding model
            collection_name: Name of the ChromaDB collection
        """
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        if embedding_model not in EMBEDDING_MODELS:
            logger.warning(f"Unknown embedding model: {embedding_model}. Defaulting to OpenAI.")
            self.embedding_model_name = "openai"
            
        # Default kwargs for embedding models
        default_kwargs = {
            "openai": {"model": "text-embedding-ada-002"},
            "huggingface": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        }
        
        # Use provided kwargs or defaults
        model_kwargs = embedding_model_kwargs or default_kwargs.get(self.embedding_model_name, {})
        
        # Initialize the embedding model
        try:
            EmbeddingClass = EMBEDDING_MODELS[self.embedding_model_name]
            self.embedding_model = EmbeddingClass(**model_kwargs)
            logger.info(f"Initialized {self.embedding_model_name} embedding model for query processing")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Load vector store
        self.vector_store = self._load_vector_store()
    
    def _load_vector_store(self) -> Optional[Chroma]:
        """
        Load the vector store from disk.
        
        Returns:
            Chroma vector store or None if it fails to load
        """
        try:
            if not os.path.exists(os.path.join(self.vector_store_dir, "chroma.sqlite3")):
                logger.error(f"Vector store not found at {self.vector_store_dir}")
                return None
            
            vector_store = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            logger.info(f"Successfully loaded vector store from {self.vector_store_dir}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def process_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query_text: The user's query text
            
        Returns:
            Dict containing the embedded query and metadata
        """
        try:
            # Create embedding for the query
            query_embedding = self.embedding_model.embed_query(query_text)
            
            return {
                "query": query_text,
                "embedding_model": self.embedding_model_name,
                "embedding": query_embedding,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query_text,
                "status": "error",
                "error": str(e)
            }
    
    def retrieve_relevant_chunks(self, 
                                query_text: str, 
                                k: int = 5,
                                filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant document chunks for a user query.
        
        Args:
            query_text: The user's query text
            k: Number of chunks to retrieve
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            Dict containing the retrieved chunks and metadata
        """
        try:
            if not self.vector_store:
                return {
                    "status": "error",
                    "error": "Vector store not loaded",
                    "chunks": []
                }
            
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(
                query_text, 
                k=k, 
                filter=filter_metadata
            )
            
            # Format the results
            formatted_chunks = []
            for doc in relevant_docs:
                chunk = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                formatted_chunks.append(chunk)
            
            return {
                "query": query_text,
                "status": "success",
                "chunks": formatted_chunks,
                "chunk_count": len(formatted_chunks)
            }
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            return {
                "query": query_text,
                "status": "error",
                "error": str(e),
                "chunks": []
            }
            
    def mmr_retrieval(self, 
                     query_text: str, 
                     k: int = 5,
                     fetch_k: int = 15,
                     diversity: float = 0.7,
                     filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant document chunks using Maximum Marginal Relevance (MMR).
        MMR diversifies results while maintaining relevance.
        
        Args:
            query_text: The user's query text
            k: Number of chunks to retrieve in final result
            fetch_k: Number of initial chunks to fetch before applying MMR
            diversity: MMR diversity parameter (0 = maximal diversity, 1 = maximal relevance)
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            Dict containing the retrieved chunks and metadata
        """
        try:
            if not self.vector_store:
                return {
                    "status": "error",
                    "error": "Vector store not loaded",
                    "chunks": []
                }
            
            # Get relevant documents using MMR
            relevant_docs = self.vector_store.max_marginal_relevance_search(
                query_text, 
                k=k, 
                fetch_k=fetch_k,
                lambda_mult=diversity,
                filter=filter_metadata
            )
            
            # Format the results
            formatted_chunks = []
            for doc in relevant_docs:
                chunk = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                formatted_chunks.append(chunk)
            
            return {
                "query": query_text,
                "status": "success",
                "chunks": formatted_chunks,
                "chunk_count": len(formatted_chunks),
                "retrieval_method": "mmr",
                "diversity": diversity
            }
        except Exception as e:
            logger.error(f"Error retrieving chunks with MMR: {str(e)}")
            return {
                "query": query_text,
                "status": "error",
                "error": str(e),
                "chunks": []
            }


def create_query_processor(vector_store_dir: str, 
                          embedding_model: str = "openai",
                          collection_name: str = "spiritual_texts", 
                          **kwargs) -> QueryProcessor:
    """
    Create a query processor with the specified parameters.
    
    Args:
        vector_store_dir: Directory of the vector store
        embedding_model: Name of embedding model ('openai', 'huggingface')
        collection_name: Name of the ChromaDB collection
        **kwargs: Additional keyword arguments for the embedding model
        
    Returns:
        QueryProcessor instance
    """
    return QueryProcessor(
        vector_store_dir=vector_store_dir,
        embedding_model=embedding_model,
        embedding_model_kwargs=kwargs,
        collection_name=collection_name
    )
