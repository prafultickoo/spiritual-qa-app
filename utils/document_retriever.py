"""
Document retriever utility for spiritual Q&A system.
Handles document retrieval from the verified vector store.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiritualDocumentRetriever:
    """
    Retrieves relevant spiritual documents from the vector store.
    """
    
    def __init__(self, vector_store_path: str = None):
        """
        Initialize the document retriever.
        
        Args:
            vector_store_path: Path to the vector store directory
        """
        self.vector_store_path = vector_store_path or os.getenv("VECTOR_STORE_DIR", "./vector_store")
        self.embeddings = None
        self.vectorstore = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store with correct collection name."""
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name="spiritual_texts"
            )
            logger.info("Document retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document retriever: {str(e)}")
            raise
    
    def retrieve_documents(self, query: str, num_docs: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: User's spiritual question
            num_docs: Number of documents to retrieve
            
        Returns:
            List of document dictionaries with content, metadata, and relevance scores
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
            
            # Perform similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=num_docs)
            
            if not docs_with_scores:
                logger.warning(f"No documents found for query: {query}")
                return []
            
            # Format results
            retrieved_docs = []
            for doc, score in docs_with_scores:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "has_verses": doc.metadata.get("has_verses", False)
                }
                retrieved_docs.append(doc_dict)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def get_context_for_query(self, query: str, max_context_length: int = 4000) -> str:
        """
        Get formatted context from retrieved documents for LLM processing.
        
        Args:
            query: User's spiritual question
            max_context_length: Maximum length of context to return
            
        Returns:
            Formatted context string for LLM
        """
        try:
            # Retrieve documents
            retrieved_docs = self.retrieve_documents(query, num_docs=8)
            
            if not retrieved_docs:
                return "No relevant spiritual texts found for this query."
            
            # Build context
            context_parts = []
            current_length = 0
            
            for i, doc in enumerate(retrieved_docs, 1):
                # Format document entry
                doc_entry = f"\n--- Spiritual Text {i} ---\n"
                doc_entry += f"Source: {doc['source']}\n"
                if doc['page'] != "Unknown":
                    doc_entry += f"Page: {doc['page']}\n"
                doc_entry += f"Relevance Score: {doc['relevance_score']:.4f}\n"
                doc_entry += f"Content: {doc['content']}\n"
                
                # Check if adding this document would exceed max length
                if current_length + len(doc_entry) > max_context_length:
                    break
                
                context_parts.append(doc_entry)
                current_length += len(doc_entry)
            
            context = "".join(context_parts)
            logger.info(f"Generated context of {len(context)} characters from {len(context_parts)} documents")
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            return f"Error retrieving spiritual context: {str(e)}"
    
    def search_by_source(self, query: str, source_filter: str, num_docs: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents from a specific source.
        
        Args:
            query: Search query
            source_filter: Source file name to filter by
            num_docs: Number of documents to retrieve
            
        Returns:
            List of documents from the specified source
        """
        try:
            # Get all relevant documents
            all_docs = self.retrieve_documents(query, num_docs=20)
            
            # Filter by source
            filtered_docs = [
                doc for doc in all_docs 
                if source_filter.lower() in doc['source'].lower()
            ]
            
            # Return top results
            return filtered_docs[:num_docs]
            
        except Exception as e:
            logger.error(f"Error searching by source: {str(e)}")
            raise
    
    def get_random_spiritual_quote(self) -> Dict[str, Any]:
        """
        Get a random spiritual quote/passage.
        
        Returns:
            Random spiritual passage with metadata
        """
        try:
            # Use a broad spiritual query to get diverse results
            docs = self.retrieve_documents("spiritual wisdom divine truth", num_docs=10)
            
            if docs:
                # Return a random one from the top results
                import random
                selected_doc = random.choice(docs[:5])
                return {
                    "quote": selected_doc['content'][:300] + "..." if len(selected_doc['content']) > 300 else selected_doc['content'],
                    "source": selected_doc['source'],
                    "page": selected_doc['page']
                }
            else:
                return {
                    "quote": "The divine wisdom is within you. Seek and you shall find.",
                    "source": "Universal Truth",
                    "page": "Heart"
                }
                
        except Exception as e:
            logger.error(f"Error getting random quote: {str(e)}")
            return {
                "quote": "In silence, the soul finds its way to truth.",
                "source": "Universal Wisdom",
                "page": "Inner Journey"
            }

# Global retriever instance
_retriever_instance = None

def get_document_retriever() -> SpiritualDocumentRetriever:
    """
    Get the global document retriever instance.
    
    Returns:
        SpiritualDocumentRetriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SpiritualDocumentRetriever()
    return _retriever_instance

def retrieve_spiritual_context(query: str, max_length: int = 4000) -> str:
    """
    Helper function to retrieve spiritual context for a query.
    
    Args:
        query: User's spiritual question
        max_length: Maximum context length
        
    Returns:
        Formatted context string
    """
    retriever = get_document_retriever()
    return retriever.get_context_for_query(query, max_length)

def search_spiritual_documents(query: str, num_docs: int = 5) -> List[Dict[str, Any]]:
    """
    Helper function to search spiritual documents.
    
    Args:
        query: Search query
        num_docs: Number of documents to return
        
    Returns:
        List of relevant documents
    """
    retriever = get_document_retriever()
    return retriever.retrieve_documents(query, num_docs)
